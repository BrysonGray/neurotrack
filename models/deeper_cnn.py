import torch
import torch.nn as nn

class ConvNet(nn.Module):
    ''' Upgraded CNN class with 5 layers and increased capacity 
        Input: (B, C, 35, 35, 35)
    '''
    def __init__(self, chin=4, ch0=32, chout=4, use_layer_norm=True, activation='relu'):
        super().__init__()        
        k = 3
        p = 1
        
        self.use_layer_norm = use_layer_norm
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        else:
            self.activation = nn.ReLU()
        
        # Layer 1: Input -> ch0 (32), stride 2. Output: 18
        self.c0 = nn.Conv3d(chin, ch0, k, stride=2, padding=p)
        self.b0 = self._make_norm(ch0)
        
        # Layer 2: ch0 -> 2*ch0 (64), stride 2. Output: 9
        self.c1 = nn.Conv3d(ch0, 2*ch0, k, stride=2, padding=p)
        self.b1 = self._make_norm(2*ch0)
        
        # Layer 3: 2*ch0 -> 4*ch0 (128), stride 2. Output: 5
        self.c2 = nn.Conv3d(2*ch0, 4*ch0, k, stride=2, padding=p)
        self.b2 = self._make_norm(4*ch0)
        
        # Layer 4: 4*ch0 -> 8*ch0 (256), stride 1. Output: 5
        self.c3 = nn.Conv3d(4*ch0, 8*ch0, k, stride=1, padding=p)
        self.b3 = self._make_norm(8*ch0)
        
        # Layer 5: 8*ch0 -> 8*ch0 (256), stride 1. Output: 5
        self.c4 = nn.Conv3d(8*ch0, 8*ch0, k, stride=1, padding=p)
        self.b4 = self._make_norm(8*ch0)
        
        # Pooling to reduce parameters while keeping some spatial info
        # Pools to 2x2x2 volume from 5x5x5
        self.pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        # Linear layers
        # Input: 8*ch0 * 2*2*2 = 256 * 8 = 2048
        flat_features = 8 * ch0 * 8
        self.l0 = nn.Linear(flat_features, 512)
        self.ln0 = nn.LayerNorm(512)
        self.l1 = nn.Linear(512, chout)
        
        self._initialize_weights()
    
    def _make_norm(self, channels):
        if self.use_layer_norm:
            return nn.GroupNorm(1, channels)
        else:
            return nn.BatchNorm3d(channels)

    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.c0(x)
        x = self.b0(x)
        x = self.activation(x)
        
        x = self.c1(x)
        x = self.b1(x)
        x = self.activation(x)
        
        x = self.c2(x)
        x = self.b2(x)
        x = self.activation(x)
        
        x = self.c3(x)
        x = self.b3(x)
        x = self.activation(x)
        
        x = self.c4(x)
        x = self.b4(x)
        x = self.activation(x)
        
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        
        x = self.l0(x)
        x = self.ln0(x)
        x = self.activation(x)
        
        x = self.l1(x)
        
        # Optional: Add a final check for NaN values
        if torch.isnan(x).any():
            print("WARNING: ConvNet output contains NaN values!")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        return x
    
if __name__ == "__main__":
    pass
