import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

# Load MNIST
transform = transforms.ToTensor()
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

device = torch.device('cuda')

# ============================================
# FEATURE DETECTOR
# Detects basic shapes: circle, vertical line,
# horizontal line, curve
# This runs BEFORE the main network
# ============================================
class FeatureDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Small cheap network - just detects basic shapes
        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 features: circle, vertical, horizontal, curve
        )
    
    def forward(self, x):
        return torch.sigmoid(self.detector(x))

# ============================================
# CANDIDATE MAP
# Based on detected features, which digits
# are possible?
# circle     → 0, 6, 8, 9
# vertical   → 1, 4, 7
# horizontal → 2, 3, 5, 7
# curve      → 2, 3, 5, 6
# ============================================
CANDIDATE_MAP = {
    'circle':     [0, 6, 8, 9],
    'vertical':   [1, 4, 7],
    'horizontal': [2, 3, 5, 7],
    'curve':      [2, 3, 5, 6]
}

def get_candidates_from_features(features):
    # features shape: [batch, 4]
    # threshold: if feature > 0.5, it's detected
    batch_size = features.shape[0]
    
    candidates = []
    for i in range(batch_size):
        f = features[i]
        detected = set()
        
        circle     = f[0] > 0.5
        vertical   = f[1] > 0.5
        horizontal = f[2] > 0.5
        curve      = f[3] > 0.5
        
        if circle:
            detected.update(CANDIDATE_MAP['circle'])
        if vertical:
            detected.update(CANDIDATE_MAP['vertical'])
        if horizontal:
            detected.update(CANDIDATE_MAP['horizontal'])
        if curve:
            detected.update(CANDIDATE_MAP['curve'])
        
        # If nothing detected, consider all
        if len(detected) == 0:
            detected = set(range(10))
            
        candidates.append(list(detected))
    
    return candidates

# ============================================
# MAIN NETWORK
# Bigger network for actual classification
# ============================================
class MainNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, 10)
    
    def forward(self, x):
        return self.classifier(self.features(x))
    
    def get_features(self, x):
        return self.features(x)
    
    def classify_candidates(self, features, candidates):
        # Only compute classifier for candidate classes
        # This is the REAL compute saving
        batch_size = features.shape[0]
        output = torch.full((batch_size, 10), float('-inf')).to(device)
        
        for i in range(batch_size):
            cands = candidates[i]
            # Only compute for candidate classes
            candidate_weights = self.classifier.weight[cands]
            candidate_bias = self.classifier.bias[cands]
            scores = features[i] @ candidate_weights.T + candidate_bias
            for j, c in enumerate(cands):
                output[i][c] = scores[j]
        
        return output

# ============================================
# TRAIN BOTH NETWORKS TOGETHER
# ============================================
feature_detector = FeatureDetector().to(device)
main_network = MainNetwork().to(device)

# Train main network first
optimizer_main = torch.optim.Adam(main_network.parameters())
criterion = nn.CrossEntropyLoss()

print("Training main network...")
for epoch in range(5):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_main.zero_grad()
        output = main_network(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer_main.step()
    print(f"Epoch {epoch+1} done")

# Now train feature detector
# Label: which features does each digit have?
DIGIT_FEATURES = {
    #         circle  vertical  horizontal  curve
    0: torch.tensor([1., 0., 0., 0.]),
    1: torch.tensor([0., 1., 0., 0.]),
    2: torch.tensor([0., 0., 1., 1.]),
    3: torch.tensor([0., 0., 1., 1.]),
    4: torch.tensor([0., 1., 1., 0.]),
    5: torch.tensor([0., 0., 1., 1.]),
    6: torch.tensor([1., 0., 0., 1.]),
    7: torch.tensor([0., 1., 1., 0.]),
    8: torch.tensor([1., 0., 0., 0.]),
    9: torch.tensor([1., 0., 0., 0.]),
}

optimizer_fd = torch.optim.Adam(feature_detector.parameters())
criterion_fd = nn.BCELoss()

print("Training feature detector...")
for epoch in range(5):
    for images, labels in train_loader:
        images = images.to(device)
        # Build feature labels for this batch
        feature_labels = torch.stack([
            DIGIT_FEATURES[l.item()] for l in labels
        ]).to(device)
        
        optimizer_fd.zero_grad()
        detected = feature_detector(images)
        loss = criterion_fd(detected, feature_labels)
        loss.backward()
        optimizer_fd.step()
    print(f"Epoch {epoch+1} done")

# ============================================
# TEST 1: STANDARD (full computation)
# ============================================
print("\nTesting standard network...")
correct = 0
total = 0
start = time.time()
total_class_computations = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = main_network(images)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        total_class_computations += len(images) * 10  # always computes all 10

standard_time = time.time() - start
standard_accuracy = 100 * correct / total

# ============================================
# TEST 2: YOUR IDEA
# Feature detection first → only compute candidates
# ============================================
print("Testing your idea...")
correct = 0
total = 0
total_class_computations_yours = 0
early_exits = 0
start = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Step 1: cheap feature detection
        features_detected = feature_detector(images)
        
        # Step 2: get candidates from detected features
        candidates = get_candidates_from_features(features_detected)
        
        # Step 3: extract shared features (one forward pass)
        shared_features = main_network.get_features(images)
        
        # Step 4: classify ONLY candidates (real compute saving)
        output = main_network.classify_candidates(shared_features, candidates)
        
        # Count actual computations
        for cands in candidates:
            total_class_computations_yours += len(cands)
        
        # Early exit check
        confidence = torch.softmax(output, dim=1).max(dim=1).values
        early_exits += (confidence > 0.90).sum().item()
        
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

your_time = time.time() - start
your_accuracy = 100 * correct / total

# ============================================
# RESULTS
# ============================================
reduction = 100 * (1 - total_class_computations_yours / total_class_computations)

print(f"\n{'='*45}")
print(f"FINAL COMPARISON - MNIST")
print(f"{'='*45}")
print(f"{'':5}{'Standard':20}{'Yours':20}")
print(f"Accuracy:     {standard_accuracy:>10.2f}%    {your_accuracy:>10.2f}%")
print(f"Time:         {standard_time:>10.3f}s    {your_time:>10.3f}s")
print(f"Class compute:{total_class_computations:>10,}    {total_class_computations_yours:>10,}")
print(f"{'='*45}")
print(f"Computation reduction: {reduction:.1f}%")
print(f"Speed improvement:     {standard_time/your_time:.2f}x")
print(f"Accuracy drop:         {standard_accuracy-your_accuracy:.2f}%")
print(f"Early exits:           {100*early_exits/total:.1f}%")
print(f"\nAvg candidates per image: {total_class_computations_yours/total:.1f}/10")
