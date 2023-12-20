import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_feature_maps(model, layer_num, input_image):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    layer = list(model.children())[layer_num]
    layer.register_forward_hook(get_activation('activation'))
    with torch.no_grad():
        model(input_image)
    return activation['activation']

def plot_feature_maps(feature_maps, original_image, cols=8):
    num_feature_maps = feature_maps.shape[1]

    # Calculate the number of rows needed, adding 1 for the original image
    rows = (num_feature_maps + 1) // cols
    rows += 0 if (num_feature_maps + 1) % cols == 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Plot the original image in the first subplot
    ax = axes[0, 0]
    ax.imshow(np.transpose(original_image.squeeze().cpu().numpy(), (1, 2, 0)))
    ax.axis('off')
    ax.set_title('Original Image')

    # Plot feature maps
    for i in range(1, rows * cols):
        if i <= num_feature_maps:
            ax = axes[i // cols, i % cols]
            ax.imshow(feature_maps[0, i - 1].cpu().numpy(), cmap='gray')
            ax.axis('off')
        else:
            axes[i // cols, i % cols].axis('off')  # Turn off axis for empty subplots

    plt.tight_layout()
    plt.show()

# Net init, load model
net = Net()
net.load_state_dict(torch.load('cfar_net_brain.pth'))
net.eval()

# Data reader/loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

def main():
    input_image, _ = next(iter(testloader)) 
    input_image = input_image[0].unsqueeze(0) 

    feature_maps = get_feature_maps(net, layer_num=0, input_image=input_image)
    plot_feature_maps(feature_maps, input_image)


if __name__ == "__main__":
    main()