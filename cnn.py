import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from math import ceil
writer = SummaryWriter("tf-logs")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Super parameter
batch_size = 512
lr = 0.04
momentum = 0.5
total_epoch = 20
# Prepare dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=4)
# Design model
class Net(nn.Module):
    def __init__(self,num_classes=10):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(20),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, num_classes)
        )
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out
model = Net().to(device)
# Construct loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# Train and Test
def train(epoch):
    for i ,(images, labels) in enumerate(train_loader):
        correct, total = 0, 0
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Visualization of training
        _, predicted = torch.max(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        batch_num = ceil(len(train_dataset)/batch_size)*epoch+i
        writer.add_scalar("acc_train/batch_num", 100 *correct / total,batch_num)
def test():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i ,(images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('[%d]: %.2f %%' % (epoch + 1, 100 * correct / total))
# Start train and Test
print('Accuracy on test set(epoch=%d):' % (total_epoch))
start=time.time()
for epoch in range(total_epoch):
    train(epoch)
    test()
end = time.time()
print('Average time per epoch:%.2f s'%((end-start)/total_epoch))
writer.close()
