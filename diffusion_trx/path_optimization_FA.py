import numpy as np
import emlddmm
import torch
from torch.nn.functional import grid_sample
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

def interp(x,I,phii,**kwargs):
    '''
    Interpolate a 3D image with specified regular voxel locations at specified sample points.
    
    Interpolate the 3D image I, with regular grid positions stored in x (1d arrays),
    at the positions stored in phii (3D arrays with first channel storing component)
    
    Parameters
    ----------
    x : list of numpy arrays
        x[i] is a numpy array storing the pixel locations of imaging data along the i-th axis.
        Note that this MUST be regularly spaced, only the first and last values are queried.
    I : array
        Numpy array or torch tensor storing 3D imaging data.  I is a 4D array with 
        channels along the first axis and spatial dimensions along the last 3 
    phii : array
        Numpy array or torch tensor storing positions of the sample points. phii is a 4D array
        with components along the first axis (e.g. x0,x1,x1) and spatial dimensions 
        along the last 3.
    kwargs : dict
        keword arguments to be passed to the grid sample function. For example
        to specify interpolation type like nearest.  See pytorch grid_sample documentation.
    
    Returns
    -------
    out : torch tensor
        4D array storing a 3D image with channels stored along the first axis. 
        This is the input image resampled at the points stored in phii.
    
    type(qdot_)
    '''
    # first we have to normalize phii to the range -1,1    
    I = torch.as_tensor(I)
    phii = torch.as_tensor(phii)
    phii = torch.clone(phii)
    for i in range(2):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    # note the above maps to 0,1
    phii *= 2.0
    # to 0 2
    phii -= 1.0
    # done
        
    if 'padding_mode' not in kwargs:
        kwargs['padding_mode'] = 'border' # note that default is zero
    out = grid_sample(I[None],phii.flip(0).permute((1,2,0))[None],align_corners=True,**kwargs)
    # note align corners true means square voxels with points at their centers
    # post processing, get rid of batch dimension
    out = out[0]
    return out


# define objective function
def energy(q, I, sigma):
#     qdot = torch.gradient(q,dim=1)[0]
    qdot = q[:,1:] - q[:,:-1]
    # interpolate image at q points
    xI = [torch.arange(x, dtype=torch.double) for x in I.shape]
    q_ = q[:,:-1] + qdot/2
    I_q = interp(xI,I[None],q_[:,None])
#     I_q = interp(xI,I[None],q[:,None])
    E = 0.5 * (torch.sum(qdot**2) + (1/sigma**2)*torch.sum(-I_q.squeeze()*torch.sqrt(torch.sum(qdot**2,0))))

    return E


# load FA
FA = np.load('outputs/amyg_dti_FA.npy')
print(' FA shape: ', FA.shape,'\n',
      'FA dtype: ', FA.dtype,'\n',
      'FA range: ', np.min(FA), np.max(FA))

fig = plt.figure(figsize=(12,12))
im = plt.imshow(FA[100])

plt.scatter(x=(80,134),y=(125,44), c='r', s=25)

col = torch.linspace(80,134,1000, dtype=torch.double)
row = torch.linspace(125,44,1000, dtype=torch.double)
plt.scatter(x=col.detach(),y=row.detach(), c='r',s=0.1)
plt.show()

col = torch.linspace(80,150,1000, dtype=torch.double)
row = torch.linspace(125,100,1000, dtype=torch.double)
q = torch.stack((row,col))
q.requires_grad = True
I = torch.tensor(FA[100])

niter = 10000
sigma = 2
eps = 5e-1 * sigma**2
Esave = []
for i in range(niter):
    E = energy(q,I,sigma)
    E.backward()
    with torch.no_grad():
        q[:,1:-1] -= q.grad[:,1:-1]*eps
        q.grad.zero_()
    Esave.append(E.item())

fig = plt.figure(figsize=(12,12))
im = plt.imshow(I)
plt.scatter(x=(80,150),y=(125,100), c='r', s=25)
plt.scatter(x=q[1].detach(),y=q[0].detach(), c='r',s=0.1)
# plt.plot(q[1].detach(),q[0].detach(), c='r')
plt.show()