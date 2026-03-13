import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

# Load MNIST
transform = transforms.ToTensor()
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# Standard plain network (exactly like 3b1b video)
class StandardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, 10)
        )
    
    def forward(self, x):
        return self.network(x)

# Train
device = torch.device('cuda')
model = StandardNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

print("Training standard network...")
for epoch in range(5):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# Test accuracy
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
end = time.time()

print(f"\n--- BASELINE RESULTS ---")
print(f"Accuracy: {100*correct/total:.2f}%")
print(f"Inference time: {end-start:.3f}s")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
