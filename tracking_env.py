#!/usr/bin/env python

"""
Deep reinforcement learning tractography environment

Author: Bryson Gray
2024
"""

import numpy as np
import torch
import scipy
from skimage.draw import line_nd
from skimage.filters import gaussian
from image import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def crop(img, center, radius):
#     """ Crop an image around a center point (rounded to the nearest pixel center).
#         The cropped image will be smaller than the given radius if it overlaps with the image boundary.

#         Parameters
#         ----------
#         img : ndarray
#             Image to be cropped with channels along the first axis (c x h x w x d).
#         center : list or tuple
#             The center of the cropped image in slice-row-col coordinates. This will be rounded to the nearest pixel index.
#         radius : int
#             The radius of the cropped image. The total width is 2*radius + 1  in each dimension assuming it doesn't intersect with a boundary.
        
#         Returns
#         -------
#         cropped_img :  ndarray
#             Cropped image
#         padding : ndarray
#             1d array of six integers specifying the distance in pixels that the cropped image overlaps with boundaries,
#             two for each dimension, the first being the overlap with the smallest image index and the second with the largest index.
#     """
#     i,j,k = [int(x) for x in center]
#     shape = img.shape[1:]
#     zpad_top = zpad_btm = ypad_front = ypad_back = xpad_left = xpad_right = 0

#     if (i + radius) > shape[0]-1:
#         zpad_btm = i + radius - (shape[0]-1)
#     if (i - radius) < 0:
#         zpad_top = radius - i
#     if (j + radius) > shape[1]-1: # back is the max y idx
#         ypad_back = j + radius - (shape[1]-1) # number of zeros to append in the y dim
#     if (j - radius) < 0: # front is zeroth idx
#         ypad_front = radius - j
#     if (k + radius) > shape[2]-1:
#         xpad_right = k + radius - (shape[2]-1) # number of zeros to append in the x dim
#     if (k - radius) < 0:
#         xpad_left = radius - k
    
#     padding = np.array([zpad_top, zpad_btm, ypad_front, ypad_back, xpad_left, xpad_right])
#     zrmd_top, zrmd_btm, yrmd_front, yrmd_back, xrmd_left, xrmd_right = np.array([radius]*6) - padding
    
#     cropped_img = img[:, i-zrmd_top:i+zrmd_btm+1, j-yrmd_front:j+yrmd_back+1, k-xrmd_left:k+xrmd_right+1] # slicing img creates a view (not a copy of img)

#     return cropped_img, padding


# def pad(img, padding, value=0):
#     """ Pad an array along each dimension with specified width and value.

#         Parmeters
#         ---------
#         img : ndarray
#             Image to be padded with channels in the first dimension.
#         padding : int or list
#             Width of padding. If int, pads equally on all sides, if a list, it must contain six ints in order of corresponding dimension,
#             two for each, the first being for the smallest index side and second for the greatest side.
#         value : int or float
#             Value to pad edges with.
        
#         Returns
#         -------
#         padded_img : torch.Tensor
#             The padded image.
#     """
#     zpad_top, zpad_btm, ypad_front, ypad_back, xpad_left, xpad_right = padding
#     padded_shape = np.array(img.shape)
#     padded_shape[1:] += np.array([zpad_top+zpad_btm, ypad_front+ypad_back, xpad_left+xpad_right])
#     padded_img = torch.ones(tuple(padded_shape)) * value
#     padded_img[:, zpad_top:padded_shape[1] - zpad_btm, ypad_front:padded_shape[2] - ypad_back, xpad_left:padded_shape[3] - xpad_right] = img

#     return padded_img


# def add_bundle_point(img, point, ball):
#     """ Add a blurred point (ball) to the last image channel.

#         This is done in-place to avoid copying arrays.
    
#         Parameters
#         ----------
#         img : torch.Tensor
#             Volume image array with channels along the first axis (c x h x w x d).
#         point : list or tuple
#             The point to add in slice-row-col coordiates. This will be rounded to the nearest int.
#         ball : ndarray
#             Blurred point to add to img.
            
#     """
#     p = [int(x) for x in point]
#     shape = ball.shape
#     radius = (shape[0] - 1)//2
#     patch, padding = crop(img, p, radius) # patch is a cropped view of the original array
#     zpad_top, zpad_btm, ypad_front, ypad_back, xpad_left, xpad_right = padding
#     # The in-place addition to patch changes the original bundle_density array since patch is a view
#     patch[-1] += torch.Tensor(ball[zpad_top:shape[0] - zpad_btm, ypad_front:shape[1] - ypad_back, xpad_left:shape[2] - xpad_right])

#     return


# def draw_line_segment(img, segment, width, dx=[1.0,1.0,1.0], binary=False):
#     """ Draw a line segment with width.

#     Parameters
#     ----------
#     segment : array_like
#         array with two three dimensional points (shape: 2x3)
    
#     width : scalar
#         segment width
#     """
#     # get the center of the patch from the segment endpoints
#     center = segment.sum(axis=0) / 2
#     direction = segment[0] - segment[1]
#     segment_length = torch.sqrt(torch.sum(direction**2))

#     # unit normalize direction
#     direction = direction / segment_length


#     # the patch should contain both line end points plus some blur
#     L = int(torch.ceil(segment_length/2)) # half the line length, rounded up
#     overhang = int(2*width) # include space for 3 standard deviations beyond the line
#     patch_radius = L + overhang
#     patch, padding = crop(center, patch_radius, pad=False) # patch is a view of self.data (c x h x w x d)

#     patch_size = 2*patch_radius + 1
#     X = torch.zeros((patch_size,patch_size,patch_size))
#     # get endpoints
#     c = torch.Tensor([patch_radius]*3)
#     start = torch.round(segment_length*direction + c).to(int)
#     end = torch.round(-segment_length*direction + c).to(int)
#     line = line_nd(start, end, endpoint=True)
#     X[line] = 1.0
#     dx = []
#     sigma = [d*width/2 for d in dx]
#     X = torch.Tensor(gaussian(X, sigma=sigma))
#     new_patch = X[padding[0]:X.shape[0]-padding[1], padding[2]:X.shape[1]-padding[3], padding[4]:X.shape[2]-padding[5]]
#     new_patch /= new_patch.max()
    
#     # add segment to patch
#     patch[-1] = torch.maximum(new_patch, patch[-1])

#     if binary:
#         patch[-1] = torch.where(patch[-1] > 0.68, 1.0, 0.0)

#     return


def density_error_change(true_density, old_density, new_density):
    """ Change in the error between true and estimated density maps. Error is mean absolute difference between density maps
        after normalizing by their sums.

        Parameters
        ----------
        true_density : torch.Tensor
            Tensor with shape 1 x h x w x d
        old_density: torch.Tensor
            Tensor with shape 1 x h x w x d
        new_density: torch.Tensor
            Tensor with shape 1 x h x w x d
        
        Returns
        -------
        error_change : scalar
    
    """
    true_density = true_density / (true_density.sum() + np.finfo(float).eps)
    old_density = old_density / (old_density.sum() + np.finfo(float).eps)
    new_density = new_density / (new_density.sum() + np.finfo(float).eps)

    old_mad = torch.mean(torch.abs(true_density - old_density))
    new_mad = torch.mean(torch.abs(true_density - new_density))

    return new_mad - old_mad


class Environment():
    """ Loads a volume. Takes a list of seeds which
    which will be cycled through on every call of reset.

    Parameters
    ----------
    img : torch.Tensor
        The image to perform tracking on. Tensor with shape c x h x w x d.
    radius : int
        The radius of a state input image patch along each axis (total width is 2*radius + 1).
    seeds : ndarray
        N x 3 array of starting streamline coordinates
    mask : torch.Tensor
        Mask of the true neuron. Tensor  with shape 1 x h x w x d.
    true_denstiy : torch.Tensor
        True neuron position as a density. Tensor with shape 1 x h x w x d
    actions : ndarray
        N x 3 array of possible step directions.
    step_size : float
        Distance taken in each step in units of pixels.
    max_len : int
        Maximum number of steps allowed per streamline
    alpha : float
        Density matching weight
    beta : float
        Smoothness weight
    
    Attributes
    ----------
    head_id : int
        Index of the current streamline head out of the whole tracking bundle.
    img : torch.Tensor
        Tensor with shape c x h x w x d where the first channels are the input image and the last channel is the streamline density.
    radius : int
        The radius of a state input image patch along each axis (total width is 2*radius + 1).
    paths : torch.Tensor
        Tensor with shape t x N x 3, where t is the length of the streamlines.
    mask : torch.Tensor
        Mask of the true neuron. Tensor  with shape 1 x h x w x d.
    true_denstiy : torch.Tensor
        True neuron position as a density. Tensor with shape 1 x h x w x d
    action_space : torch.Tensor
        N x 3 tensor of possible step directions.
    step_size : float
        Distance taken in each step in units of pixels.
    max_len : int
        Maximum number of steps allowed per streamline
    cos_path_angle : list
        Cosine angle between step directions. A list of N floats, one for each path.
    n_resets : int
        Count of the number of episode resets during training.

    """

    def __init__(self, img, radius, seeds, mask, true_density, actions, n_seeds=1, step_size=1.0,\
                 step_width=1.0, pixelsize=[1.0,1.0,1.0], max_len=10000, alpha=1.0, beta=1e-3, friction=1e-4):

        self.head_id = 0
        self.n_resets = 0 # count number of resets
        self.img = Image(img, dx=pixelsize)
        self.radius = radius
        self.seeds = seeds        
        self.mask = mask
        self.true_density = Image(true_density, dx=pixelsize)
        self.n_seeds = n_seeds # the number of paths per bundle
        self.step_size = torch.Tensor([step_size]) / torch.Tensor(pixelsize)
        self.step_width = step_width
        self.max_len = max_len
        self.action_space = actions
        if not isinstance(self.action_space, torch.Tensor):
            self.action_space = torch.tensor(self.action_space)

        seed_id = self.n_resets % len(self.seeds)
        center = torch.Tensor(self.seeds[seed_id])
        self.r = 5.0 # radius around center to randomly place starting points
        g = torch.Generator()
        g.manual_seed(0)
        offsets = self.r * torch.rand((self.n_seeds, 3), generator=g)
        bundle_seeds = center[None] + offsets

        self.paths = [*torch.Tensor(bundle_seeds)[:,None]] # a list of N paths. each path is a 1 x 3 tensor
        self.cos_path_angle = [1.0] * len(self.paths) # list of N floats.       
        self.alpha = alpha
        self.beta = beta
        self.friction = friction

        # initialize last two step directions randomly for each streamline
        g = torch.Generator()
        g.manual_seed(self.n_resets)
        last_steps = [*2*torch.rand(((len(self.paths),)+(1,3)), generator=g)-1.0] # list of len N paths, of 1x3 tensors
        last_steps = [x / np.sqrt(x[0,0]**2+x[0,1]**2+x[0,2]**2) for x in last_steps] # unit normalize directions
        self.paths = [torch.cat((point - 2*step*self.step_size, point - step*self.step_size, point)) for point, step in zip(self.paths, last_steps)]

        # initialize bundle density map
        # compute gaussian ball to represent streamline points in bundle density map
        # w = 2*radius + 1
        # spike = np.zeros((w,w,w))
        # spike[radius,radius,radius] = 1.0
        # self.ball = scipy.ndimage.gaussian_filter(spike, sigma=2)

        bundle_density = torch.zeros_like(true_density)
        self.img.data = torch.cat((self.img.data, bundle_density), dim=0)
        for i in range(len(self.paths)):
            # add_bundle_point(bundle_density, self.paths[i][0], self.ball)
            for j in range(len(self.paths[i])-1):
                segment = torch.stack((self.paths[i][j], self.paths[i][j+1]), dim=0)
                self.img.draw_line_segment(segment, width=self.step_width)


    def get_state(self):
        """ Get the state for the current step at streamline 'head_id'. The state consists of an image patch and
        streamline density patch centered on the streamline head plus the last three streamline positions.

        Returns
        -------
        patch : torch.Tensor
            Tensor with shape (n x c x h x w x d) where the first channels are the input image and the last channel is the streamline density.
        last_steps : torch.Tensor
         Tensor with shape 3 x 3 (3 step positions by 3 euclidean coordinates)
        """
        patch, _ = self.img.crop(self.paths[self.head_id][-1], self.radius, pad=True, value=0.0)
        patch = patch.detach().clone()
        
        last_steps = self.paths[self.head_id][-3:].detach().clone() # 3 x 3 tensor of last three streamline positions

        return patch[None], last_steps[None]


    def get_reward(self, terminated, delta_density_diff, verbose=False):
        """ Get the reward for the current state. The reward depends on the streamline smoothness and
        the change in distance between the streamline density and true denisty maps.

        Parameters
        ----------
        terminated : bool
        alpha : float
            Weight for density matching.
        beta : float
            Weight for streamline smoothness.
        friction = 
            
        Returns
        -------
        reward : torch.Tensor
            Tensor with shape 1
        """
        if terminated:
            reward = 0
        else:
            prev_step = (self.paths[self.head_id][-2]-self.paths[self.head_id][-3]) / self.step_size
            current_step = (self.paths[self.head_id][-1]-self.paths[self.head_id][-2]) / self.step_size
            cos_angle = torch.dot(current_step, prev_step).to(float) #/ torch.sqrt(torch.sum(self.step_size**2))
            # note that delta density difference is a change in error,
            # so negative change is good, hence the flipped (positive) exponent which is normally negative for sigmoid function.
            # it is also shifted down so that zero change yields zero.
            sigmoid_diff = 2e4 / (1 + np.exp(delta_density_diff / 1e-4)) - 1e4 # calibrated so that a good step is around 1.
            # sigmoid_diff = np.max([sigmoid_diff, 0.0]) # do not give negative values for matching
            reward = self.alpha*sigmoid_diff + self.beta*(cos_angle-1) - self.friction 
            if verbose:
                print(f'sigmoid_diff: {sigmoid_diff}','\n',
                      f'matching reward: {self.alpha*sigmoid_diff}','\n',
                      f'cos_angle: {cos_angle}','\n',
                      f'smoothing reward: {self.beta*(cos_angle-1)}', '\n',
                      f'friction reward: {-self.friction}')

        return torch.tensor([reward], device=DEVICE, dtype=torch.float32)


    def step(self, action, verbose=False):
        """ Take a tracking step for one streamline. Add the new position to the path and
        and the bundle density mask, and compute the reward and new state observation.

        Parameters
        ----------
        action : torch.Tensor
            Tensor with shape 3.
        
        Returns
        -------
        observation : tuple
            The img patch centered at the new streamline head (a tensor with shape c x h x w x d), and the
            last three streamline positions.
        reward : torch.Tensor
            Tensor with shape 1
        terminated : bool

        """
        terminated = False

        # bounce agent back in bounds if it steps out
        # add new position to path
        # while True:
        #     new_position = self.paths[self.head_id][-1] + self.step_size*action
        #     out_of_bound = any([x >= y or x < 0 for x,y in zip(torch.round(new_position), self.img.data.shape[1:])])
        #     if out_of_bound:
        #         action = self.action_space[int(np.random.randint(len(self.action_space)))]
        #     else:
        #         break

        new_position = self.paths[self.head_id][-1] + self.step_size*action
        out_of_bound = any([x >= y or x < 0 for x,y in zip(torch.round(new_position), self.img.data.shape[1:])])
        out_of_mask = 1 - self.mask[(0,)+tuple([int(x) for x in self.paths[self.head_id][-1]])]
        self.paths[self.head_id] = torch.cat((self.paths[self.head_id], new_position[None]))
        # decide if path terminates
        # out_of_mask = self.mask[tuple([int(x) for x in self.paths[self.head_id][-1]])]
        too_long = len(self.paths[self.head_id]) > self.max_len
        self_terminate = not any(action)
        terminate_path = too_long or self_terminate or out_of_mask or out_of_bound

        if terminate_path:
            observation = None
            reward = torch.tensor([0.], device=DEVICE)
            # remove the path from 'paths' and add it to 'ended_paths'
            self.paths.pop(self.head_id)
            # if that was the last path in the list, then terminate the episode
            if len(self.paths) == 0:
                terminated = True
            else:
                self.head_id = (self.head_id + 1)%len(self.paths)
        else:
            center = self.paths[self.head_id][-2]
            r = self.radius + int(np.ceil(self.step_size.max()))
            true_density_patch, _ = self.true_density.crop(center, radius=r) # patch centered at previous step position
            old_density_patch, _ = self.img.crop(center, radius=r)
            old_density_patch = old_density_patch.detach().clone()[-1][None] # need to make a copy or else this will be modified by adding a point to img
            # add_bundle_point(self.img, self.paths[self.head_id][-1], self.ball)
            segment = self.paths[self.head_id][-2:, :3]
            self.img.draw_line_segment(segment, width=self.step_width)
            new_density_patch, _ = self.img.crop(center, radius=r)
            new_density_patch = new_density_patch[-1][None]
            delta_density_diff = density_error_change(true_density_patch, old_density_patch, new_density_patch)
            observation = self.get_state()
            reward = self.get_reward(terminated, delta_density_diff, verbose=verbose)

            self.head_id = (self.head_id + 1)%len(self.paths)

        return observation, reward, terminated


    def reset(self):
        # reset the bundle with new random previous step directions.
        self.n_resets += 1

        # start with next seed point
        seed_id = self.n_resets % len(self.seeds)
        center = torch.Tensor(self.seeds[seed_id])
        self.r = 5.0 # radius around center to randomly place starting points
        g = torch.Generator()
        g.manual_seed(0)
        offsets = self.r * torch.rand((self.n_seeds, 3), generator=g)
        bundle_seeds = center[None] + offsets

        self.paths = [*torch.Tensor(bundle_seeds)[:,None]] # a list of N paths. each path is a 1 x 3 tensor
        self.cos_path_angle = [1.0] * len(self.paths) # list of N floats.       

        # initialize last two step directions randomly for each streamline
        g = torch.Generator()
        g.manual_seed(self.n_resets)
        last_steps = [*2*torch.rand(((len(self.paths),)+(1,3)), generator=g)-1.0] # list of len N paths, of 1x3 tensors
        last_steps = [x / np.sqrt(x[0,0]**2+x[0,1]**2+x[0,2]**2) for x in last_steps] # unit normalize directions
        self.paths = [torch.cat((point - 2*step*self.step_size, point - step*self.step_size, point)) for point, step in zip(self.paths, last_steps)]

        # reset bundle density
        self.img.data[-1] = torch.zeros_like(self.true_density.data[0])
        for i in range(len(self.paths)):
            for j in range(len(self.paths[i])-1):
                segment = torch.stack((self.paths[i][j], self.paths[i][j+1]), dim=0)
                self.img.draw_line_segment(segment, width=self.step_width)

        return