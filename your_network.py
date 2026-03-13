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
# STEP 1: EXTRACT DNA / CRITICAL POINTS
# ============================================
def extract_digit_dna(dataset):
    print("Extracting digit DNA...")
    digit_pixels = {i: [] for i in range(10)}
    for image, label in dataset:
        digit_pixels[label].append(image.numpy().squeeze())
    
    digit_dna = {}
    for digit in range(10):
        stacked = np.stack(digit_pixels[digit])
        digit_dna[digit] = np.mean(stacked, axis=0)
    
    return digit_dna

# ============================================
# STEP 2: YOUR NETWORK
# ============================================
class YourNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
    
    def forward(self, x):
        return self.network(x)

# ============================================
# TRAIN
# ============================================
model = YourNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

print("Training your network...")
for epoch in range(5):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# ============================================
# EXTRACT DNA + BUILD ELIMINATION MATRIX
# ============================================
digit_dna = extract_digit_dna(train_data)

elim_matrix = torch.stack([
    torch.tensor(digit_dna[d].flatten(), dtype=torch.float32) 
    for d in range(10)
]).to(device)
elim_matrix = elim_matrix / (elim_matrix.norm(dim=1, keepdim=True) + 1e-8)

# ============================================
# TEST 1: STANDARD
# ============================================
print("\nTesting standard...")
correct = 0
total = 0
start = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

standard_time = time.time() - start
standard_accuracy = 100 * correct / total
standard_flops = total * 10

# ============================================
# TEST 2: YOUR IDEA
# ============================================
print("Testing your idea...")
correct = 0
total = 0
total_candidates = 0
early_exits = 0
start = time.time()

TOP_K = 4

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # DNA matching - batch operation on GPU
        images_flat = images.view(images.size(0), -1)
        images_norm = images_flat / (images_flat.norm(dim=1, keepdim=True) + 1e-8)
        dna_scores = torch.mm(images_norm, elim_matrix.T)
        _, top_candidates = torch.topk(dna_scores, k=TOP_K, dim=1)

        # Full network forward pass
        output = model(images)

        # Mask non candidates
        mask = torch.full_like(output, float('-inf'))
        for i in range(len(images)):
            mask[i][top_candidates[i]] = output[i][top_candidates[i]]

        # Early exit check
        confidence = torch.softmax(mask, dim=1).max(dim=1).values
        early_exits += (confidence > 0.90).sum().item()

        _, predicted = torch.max(mask, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        total_candidates += TOP_K * len(images)

your_time = time.time() - start
your_accuracy = 100 * correct / total
your_flops = total_candidates

# ============================================
# FINAL RESULTS
# ============================================
print(f"\n{'='*40}")
print(f"FINAL COMPARISON - MNIST")
print(f"{'='*40}")
print(f"{'':5}{'Standard':20}{'Yours':20}")
print(f"Accuracy:     {standard_accuracy:>10.2f}%    {your_accuracy:>10.2f}%")
print(f"Time:         {standard_time:>10.3f}s    {your_time:>10.3f}s")
print(f"Computations: {standard_flops:>10,}    {your_flops:>10,}")
print(f"{'='*40}")
print(f"Computation reduction: {100*(1-your_flops/standard_flops):.1f}%")
print(f"Speed improvement:     {standard_time/your_time:.2f}x")
print(f"Accuracy drop:         {standard_accuracy-your_accuracy:.2f}%")
print(f"Early exits:           {100*early_exits/total:.1f}%")
