import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
 
import matplotlib.pyplot as plt
train_loader=torch.utils.data.DataLoader(torchvision.datasets.MNIST('/files/',train=True,download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),batch_size=4, shuffle=True)
Test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/files/',train=False,download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),batch_size=4, shuffle=True)
#torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv_trans1 = nn.ConvTranspose2d(6, 3, 4, 2, 1)
        self.conv_trans2 = nn.ConvTranspose2d(3, 1, 4, 2, 1)
        
    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))        
        x = F.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        return x
 
#dataset = datasets.MNIST(
 #   root='PATH',
  #
#loader = DataLoader(
 #   dataset,
  #  num_workers=2,
   # batch_size=8,
    #shuffle=True
 
 
model = MyModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
 
epochs = 1
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        
        print('Epoch {}, Batch idx {}, loss {}'.format(
            epoch, batch_idx, loss.item()))
 
 
def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img
 
# Plot some images
idx = torch.randint(0, output.size(0), ())
pred = normalize_output(output[idx, 0])
img = data[idx, 0]
 
fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(img.detach().numpy())
axarr[1].imshow(pred.detach().numpy())
 
# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
 
model.conv1.register_forward_hook(get_activation('conv1'))
data = train_loader[0]
data.unsqueeze_(0)
output = model(data)
 
act = activation['conv1'].squeeze()
fig, axarr = plt.subplots(act.size(0))
for idx in range(act.size(0)):
    axarr[idx].imshow(act[idx])
 
# Visualize conv filter
kernels = model.conv1.weight.detach()
fig, axarr = plt.subplots(kernels.size(0))
for idx in range(kernels.size(0)):
    axarr[idx].imshow(kernels[idx].squeeze())
 
 
#feature extractor 
class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential( 
  nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5,                 stride=1, padding=2),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2),
 
         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
          nn.ReLU(True),
          nn.MaxPool2d(kernel_size=2),
 
          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0),
          nn.ReLU(True),
          nn.MaxPool2d(kernel_size=2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 6, 2)
        )
 
    def forward(self, x):
        x = self.extractor(x)
        x = self.decoder(x)
        return x
 
 
model = Extractor()
model.extractor[0].register_forward_hook(get_activation('ext_conv1'))
x = torch.randn(1, 3, 96, 96)
output = model(x)
print(output.shape)
 
 
act = activation['ext_conv1'].squeeze()
num_plot = 4
fig, axarr = plt.subplots(min(act.size(0), num_plot))
for idx in range(min(act.size(0), num_plot)):
    axarr[idx].imshow(act[idx])
 
 
# grid of the extracted feature 
from torchvision.utils import make_grid
 
kernels = model.extractor[0].weight.detach().clone()
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
img = make_grid(kernels)
plt.imshow(img.permute(1, 2, 0))
