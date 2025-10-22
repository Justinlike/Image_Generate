import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_Encoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc21 = torch.nn.Linear(hidden_dim, latent_dim * 2) # 输出均值和对数方差

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        z_mean_logvar = self.fc21(h1)
        mean, logvar = z_mean_logvar.chunk(2, dim=1)
        return mean, logvar
    
class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAE_Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = torch.sigmoid(self.fc2(x))
        return x
    
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.encode = VAE_Encoder(input_size, hidden_size, latent_size)
        self.decode = VAE_Decoder(latent_size, hidden_size, input_size)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
def vae_loss(x_recon, x, mu, log_var):
    recon_loss = nn.BCELoss(reduction = 'sum')(x_recon, x)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_loss

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])
mnist = datasets.MNIST(root='.', download=True, transform=transform)

# 训练参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

data_loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)

input_size = 28*28
hidden_size = 256
latent_size = 64
model = VAE(input_size, hidden_size, latent_size)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if __name__ == "__main__":
    # model.train()
    # for epoch in range(NUM_EPOCHS):
    #     epoch_loss = 0.0
    #     for x, _ in data_loader:
    #         x = x.view(-1, input_size)
    #         x_recon, mu, log_var = model(x)
    #         loss = vae_loss(x_recon, x, mu, log_var)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #     print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss/len(data_loader.dataset):.4f}')

    with torch.no_grad():
        z = torch.randn(1, latent_size)
        image = model.decode(z).view(28, 28)
        image = image.detach().numpy()
        plt.imshow(image, cmap='gray')
        plt.show()
    
    torch.save(model.state_dict(), 'vae_model.pth')