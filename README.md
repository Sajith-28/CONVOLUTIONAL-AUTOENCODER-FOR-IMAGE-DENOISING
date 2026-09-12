# Experiment 7 - Convolutional Autoencoder for Image Denoising

## Aim

To develop an autoencoder model using PyTorch to remove noise from MNIST handwritten digit images.

## Algorithm

1. Import the required PyTorch, torchvision, NumPy and Matplotlib libraries.
2. Select CUDA if available, otherwise use CPU.
3. Convert the MNIST images into tensors.
4. Load the MNIST training and testing datasets.
5. Create DataLoaders with a batch size of 128.
6. Add random noise to the input images using a noise factor of 0.5.
7. Create an autoencoder with an encoder and decoder.
8. Flatten the 28 × 28 image before passing it to the encoder.
9. Train the model using Mean Squared Error (MSE) loss and Adam optimizer.
10. Train the model for 5 epochs.
11. Pass noisy test images through the trained model.
12. Display the original, noisy and denoised images.

## Program

```python
# Autoencoder for Image Denoising using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])


# Load MNIST dataset
dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)


# Denoising Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x


# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Print model summary
summary(model, input_size=(1, 28, 28))


# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for data in loader:
            inputs, _ = data
            inputs = inputs.to(device)

            noisy_inputs = add_noise(inputs).to(device)

            optimizer.zero_grad()

            outputs = model(noisy_inputs)

            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(loader.dataset)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')


# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:                   ")
    print("Register Number:                  ")

    plt.figure(figsize=(18, 6))

    for i in range(num_images):

        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)
```

## Output

### Model Summary

```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                  [-1, 128]         100,480
              ReLU-2                  [-1, 128]               0
            Linear-3                   [-1, 64]           8,256
              ReLU-4                  [-1, 64]               0
            Linear-5                   [-1, 32]           2,080
            Linear-6                   [-1, 64]           2,112
              ReLU-7                  [-1, 64]               0
            Linear-8                  [-1, 128]           8,320
              ReLU-9                  [-1, 128]               0
           Linear-10                  [-1, 784]         101,136
          Sigmoid-11                  [-1, 784]               0
================================================================
Total params: 222,384
Trainable params: 222,384
Non-trainable params: 0
----------------------------------------------------------------
```

### Training Output

```text
Epoch 1/5, Loss: 0.0685
Epoch 2/5, Loss: 0.0432
Epoch 3/5, Loss: 0.0347
Epoch 4/5, Loss: 0.0313
Epoch 5/5, Loss: 0.0291
```

### Image Output

The output displays 10 images in three rows:

```text
Original → Noisy → Denoised
```


<img width="540" height="342" alt="image" src="https://github.com/user-attachments/assets/b4a7a7ad-2ad6-44d5-b276-057b4dd3e16b" />


<img width="1535" height="610" alt="image" src="https://github.com/user-attachments/assets/d2abd13f-77fc-4c81-b13e-f0c3c395af06" />

## Result

Thus, the denoising autoencoder was successfully developed using PyTorch and trained on the MNIST dataset. The model reduced the training loss from **0.0685 to 0.0291** in 5 epochs and produced denoised images from noisy input images.
