import torch
import numpy as np

class SphericalSampler():

    def __init__(self, input_shape, radii, resolution=(180,360)):
        """
        Initialize the interpolation grid for spherical sampling.
        This method sets up a grid of points on concentric spheres with specified radii,
        and precomputes the weights and indices needed for trilinear interpolation.

        Parameters
        ----------
        input_shape : tuple
            The shape of the input volume, with the last three dimensions representing
            the spatial dimensions (z, y, x).
        radii : array_like
            List or array of radii for the concentric spheres.
        resolution : tuple, optional
            Resolution of the spherical grid as (theta_res, phi_res), where theta_res
            is the number of points along the polar angle (0 to π) and phi_res is the
            number of points along the azimuthal angle (0 to 2π). Default is (180, 360).

        Notes
        -----
        The method computes the following attributes:
        - weights000, weights001, ..., weights111: Interpolation weights for the 8 corners
          of each voxel containing a sample point.
        - points_i000, points_i001, ..., points_i111: Linear indices of the 8 corners for
          each sample point, flattened to 1D for efficient access.
        The coordinate system is set up with the origin at the corner of the volume, and
        the points are scaled to match the specified radii.
        """

        
        self.input_shape = input_shape
        self.radii = radii
        self.resolution = resolution

        # compute sample points
        theta_res, phi_res = resolution
        theta = torch.linspace(0, np.pi, theta_res)
        phi = torch.linspace(0, 2*np.pi, phi_res)
        theta_grid, phi_grid = torch.meshgrid(theta, phi)

        # Convert to cartesian coordinates (points on a unit sphere)
        x = (torch.sin(theta_grid) * torch.cos(phi_grid) + 1) # ndim is 2, i.e. shape is equal to resolution
        x = torch.stack((x,) * len(radii))  
        x = x * radii[:,None,None] # ndim is 3 with radius varying along the first dimension
        y = (torch.sin(theta_grid) * torch.sin(phi_grid) + 1)
        y = torch.stack((y,) * len(radii))
        y = y * radii[:,None,None]
        z = (torch.cos(theta_grid) + 1)
        z = torch.stack((z,) * len(radii))
        z = z * radii[:,None,None]

        points_f000 = torch.stack((z,y,x), dim=-1).reshape(-1,3)
        points_i000 = torch.floor(points_f000).long()
        p000 = points_f000 - points_i000

        # compute sample weights
        self.weights000 = (1.0 - p000[...,0])*(1.0 - p000[...,1])*(1.0 - p000[...,2])
        self.weights001 = (1.0 - p000[...,0])*(1.0 - p000[...,1])*(      p000[...,2])
        self.weights010 = (1.0 - p000[...,0])*(      p000[...,1])*(1.0 - p000[...,2])
        self.weights011 = (1.0 - p000[...,0])*(      p000[...,1])*(      p000[...,2])
        self.weights100 = (      p000[...,0])*(1.0 - p000[...,1])*(1.0 - p000[...,2])
        self.weights101 = (      p000[...,0])*(1.0 - p000[...,1])*(      p000[...,2])
        self.weights110 = (      p000[...,0])*(      p000[...,1])*(1.0 - p000[...,2])
        self.weights111 = (      p000[...,0])*(      p000[...,1])*(      p000[...,2])

        # Vectorize points
        n = input_shape[-3:]
        self.points_i000 = points_i000[:,0]*n[1]*n[2] + points_i000[:,1]*n[2] + points_i000[:,2]

        # Find new sample points
        self.points_i001 = self.points_i000 + 1
        self.points_i010 = self.points_i000 + n[2]
        self.points_i011 = self.points_i000 + n[2] + 1
        self.points_i100 = self.points_i000 + n[1]*n[2]
        self.points_i101 = self.points_i000 + n[1]*n[2] + 1
        self.points_i110 = self.points_i000 + n[1]*n[2] + n[2]
        self.points_i111 = self.points_i000 + n[1]*n[2] + n[2] + 1
    

    def map_coordinates(self,img):
        """

        Parameters
        ----------
        img : torch.Tensor
            The input image with shape (N, C, D, H, W).

        Returns
        -------
        Iout : torch.Tensor
            Output stack of 2D spherical projections of the input image with shape (N, N_radii, C, H, W). 
        """

        if img.shape[-3:] != self.input_shape[-3:]:
            raise Exception(f"Input image shape along last 3 dimensions ({img.shape[-3:]}) does not match expected shape ({self.input_shape[-3:]}).")
        
        if img.ndim != 5:
            raise Exception(f"Input image must have 5 dimensions (N, C, D, H, W) but found {img.ndim}")
        
        Ivec = img.reshape(img.shape[0], img.shape[1], -1)
        Iout = torch.zeros((img.shape[0], img.shape[1], len(self.points_i000)))
        Iout.addcmul_(Ivec[...,self.points_i000], self.weights000 )
        Iout.addcmul_(Ivec[...,self.points_i001], self.weights001 )
        Iout.addcmul_(Ivec[...,self.points_i010], self.weights010 )
        Iout.addcmul_(Ivec[...,self.points_i011], self.weights011 )
        Iout.addcmul_(Ivec[...,self.points_i100], self.weights100 )
        Iout.addcmul_(Ivec[...,self.points_i101], self.weights101 )
        Iout.addcmul_(Ivec[...,self.points_i110], self.weights110 )
        Iout.addcmul_(Ivec[...,self.points_i111], self.weights111 )
        Iout = Iout.reshape(img.shape[0], len(self.radii), img.shape[1], self.resolution[0], self.resolution[1])

        return Iout
        
        
        