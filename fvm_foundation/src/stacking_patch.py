import torch.nn as nn

# Does a channel stacking early fusion patch embedding.
# This means that data from the previous timesteps is stacked into a single encoded patch.
# NOTE: perhaps this is not optimal because what is actually relevant is the transport between patches.
# If we were able to do some expanding thing where we add more attention to further away patches we could do better.
class StackingPatchEmbedding(nn.Module):
    # `patch_size` should be a power of 2 
    def __init__(self, num_obs: int, in_channels: int, patch_size: int = 32, emb_dim: int = 5000):
        super().__init__()

        self.total_channels = num_obs * in_channels
        
        self.projection = nn.Conv2d(
            self.total_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # [batch, total_channels, height, width] -> [batch, emb_size, H/patch_size, W/patch_size]
        x = self.projection(x)

        # Removes separate width and height and just makes every patch a flat vector.
        # [batch, emb_size, no_patches]
        x = x.flatten(2)

        # Transpose so that it is ready for the transformer.
        # [batch, no_patches, emb_size]
        x = x.transpose(1, 2)

        return x
