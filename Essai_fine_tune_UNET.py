import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F

# U-Net Modifié pour prédire un champ de déplacement (vx, vy)
class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(UNet, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = conv_block(256, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)  # 2 canaux pour (vx, vy)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.final(d1)

# Fonction de perte avec lissage spatial et incompressibilité
def loss_function(pred, target):
    l1_loss = F.l1_loss(pred, target)
    
    # Lissage spatial (régularisation sur le gradient)
    smoothness = torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])) + \
                 torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]))
    
    # Contraintes d'incompressibilité (divergence nulle)
    div = torch.mean(torch.abs(
        torch.gradient(pred[:, 0], dim=-1)[0] + torch.gradient(pred[:, 1], dim=-2)[0]
    ))
    
    return l1_loss + 0.01 * smoothness + 0.01 * div

# Initialisation du modèle
model = UNet().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Exemple d'entraînement (données factices pour l'instant)
for epoch in range(10):  # Remplace 10 par le nombre réel d'époques
    # Simule des données (batch_size, channels=2, H, W)
    images = torch.rand(4, 2, 128, 128).cuda()  # Deux images en entrée
    ground_truth = torch.rand(4, 2, 128, 128).cuda()  # Champ de vitesse vrai
    
    optimizer.zero_grad()
    pred = model(images)
    loss = loss_function(pred, ground_truth)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
