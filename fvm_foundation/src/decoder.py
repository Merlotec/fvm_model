import torch.nn as nn
import torch.nn.functional as F

class FluidDecoder(nn.Module):
    def __init__(self, emb_dim=768, out_channels=3):
        super().__init__()

        # Every conv2d step doubles the size of the patches and halves the number of dimensions per patch.
        # Kernel size gives the dimensionality of the network.
        
        self.up1 = nn.ConvTranspose2d(emb_dim, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)        

        self.final_proj = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
                
        # [batch, 768, 196]
        x = x.transpose(1, 2)

        # Convert into grid for the CNN.
        n = x.shape[2]
        grid = int(n ** 0.5)
        x = x.unflatten(2, (grid, grid))
        
        # Use relu activation between upsampling steps to learn non linear fluid boundaries.
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        x = F.relu(self.up4(x))
        
        # No activation function here, as fluid variables (velocity) can be negative.
        # [batch, 3, 224, 224]
        output = self.final_proj(x)      
        
        return output
