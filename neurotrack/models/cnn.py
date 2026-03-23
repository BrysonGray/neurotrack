import torch


class ConvNet(torch.nn.Module):
    ''' Base CNN class

    '''
    def __init__(self, chin=4, ch0=16, chout=4, use_layer_norm=True, activation='relu'):
        super().__init__()        
        k = 3
        p = (k-1)//2
        s = 2
        self.use_layer_norm = use_layer_norm
        
        # Choose activation function
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU()
        elif activation == 'swish':
            self.activation = torch.nn.SiLU()  # Swish activation
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU(0.01)
        else:
            self.activation = torch.nn.ReLU()  # Default fallback
        
        # Input normalization
        # self.n0 = torch.nn.InstanceNorm3d(chin, affine=True)
        
        # Convolutional layers
        self.c0 = torch.nn.Conv3d(chin, ch0, k, s, p)
        self.c1 = torch.nn.Conv3d(ch0, 2*ch0, k, s, p)
        self.c2 = torch.nn.Conv3d(2*ch0, 4*ch0, k, s, p)
        
        # Normalization layers - choose between BatchNorm and LayerNorm
        if use_layer_norm:
            # LayerNorm is more stable than BatchNorm
            self.b0 = torch.nn.GroupNorm(1, ch0)  # GroupNorm with 1 group = LayerNorm
            self.b1 = torch.nn.GroupNorm(1, 2*ch0)
            self.b2 = torch.nn.GroupNorm(1, 4*ch0)
        else:
            # Original BatchNorm (less stable)
            self.b0 = torch.nn.BatchNorm3d(ch0)
            self.b1 = torch.nn.BatchNorm3d(2*ch0)
            self.b2 = torch.nn.BatchNorm3d(4*ch0)
        
        # Linear layers with normalization
        self.l0 = torch.nn.Linear(5**3*64, 64)
        self.ln0 = torch.nn.LayerNorm(64)  # Normalize linear layer output
        self.l1 = torch.nn.Linear(64, chout)
        
        # Initialize weights to prevent NaN issues
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.Linear)):
                # He initialization for ReLU activations
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, (torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
    def forward(self, x):
        # Input normalization to stabilize training
        # x = self.n0(x)
        
        x = self.c0(x)
        x = self.b0(x)
        x = self.activation(x)
        
        x = self.c1(x)
        x = self.b1(x)
        x = self.activation(x)
        
        x = self.c2(x)
        x = self.b2(x)
        x = self.activation(x)
        
        # Flatten and linear layers
        x = self.l0(x.reshape(x.shape[0], -1))
        x = self.ln0(x)  # Normalize before activation
        x = self.activation(x)
        
        x = self.l1(x)
        
        # Optional: Add a final check for NaN values
        if torch.isnan(x).any():
            print("WARNING: ConvNet output contains NaN values!")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        return x
    
if __name__ == "__main__":
    pass