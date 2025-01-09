import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt


batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10

device = torch.device('cuda')

class CustomMNISTDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label = int(img_name.split('_label_')[-1].split('.')[0])  # Extract label from filename
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path)  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10),
    transforms.Normalize(mean = (0.1307,), std = (0.3081,))  # Normalization as per the essay
])

# Load the train and test datasets
train_dataset = CustomMNISTDataset(folder_path="output/train", transform=transform)
test_dataset = CustomMNISTDataset(folder_path="output/test", transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


model = LeNet5(num_classes).to(device)

cost = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)

train_error_rates = []
test_error_rates = []

for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = cost(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()


    train_error_rate = 100 * (1 - train_correct / train_total)
    train_error_rates.append(train_error_rate)

    with torch.no_grad():
        test_correct = 0
        test_total = 0
        all_labels = []
        all_predictions = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        test_error_rate = 100 * (1 - test_correct / test_total)
        test_error_rates.append(test_error_rate)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Error Rate: {train_error_rate:.2f}%, Test Error Rate: {test_error_rate:.2f}%')


plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_error_rates, label='Train Error Rate')
plt.plot(range(1, num_epochs + 1), test_error_rates, label='Test Error Rate')
plt.xlabel('Epoch')
plt.ylabel('Error Rate (%)')
plt.title('Training and Test Error Rates')
plt.legend()
plt.grid()
plt.show()





torch.save(model.state_dict(), "LeNet5_2.pth")



final_train_error_rate = train_error_rates[-1]
final_test_error_rate = test_error_rates[-1]
print(f"Final Train Error Rate: {final_train_error_rate:.2f}%")
print(f"Final Test Error Rate: {final_test_error_rate:.2f}%")
