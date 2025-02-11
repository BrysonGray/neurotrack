import torch


class ConvNet(torch.nn.Module):
    ''' Base CNN class

    '''
    def __init__(self,chin=4,ch0=16,chout=4):
        super().__init__()        
        k = 3
        p = (k-1)//2
        s = 2
        self.c0 = torch.nn.Conv3d(chin,ch0,k,s,p)
        self.b0 = torch.nn.BatchNorm3d(ch0)
        self.c1 = torch.nn.Conv3d(ch0,2*ch0,k,s,p)
        self.b1 = torch.nn.BatchNorm3d(2*ch0)
        self.c2 = torch.nn.Conv3d(2*ch0,4*ch0,k,s,p)
        self.b2 = torch.nn.BatchNorm3d(4*ch0)
        self.l0 = torch.nn.Linear(5**3*64,64)        
        self.l1 = torch.nn.Linear(64,chout)
    def forward(self,x):
        x = self.c0(x)
        x = self.b0(x)
        x = torch.relu(x)
        
        x = self.c1(x)
        x = self.b1(x)
        x = torch.relu(x)
        
        x = self.c2(x)
        x = self.b2(x)
        x = torch.relu(x)
        
        
        x = self.l0(x.reshape(x.shape[0],-1))
        x = torch.relu(x)
        
        x = self.l1(x)
        return x
    
if __name__ == "__main__":
    pass