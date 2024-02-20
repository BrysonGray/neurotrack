#!/usr/bin/env python

"""
Train deep Q-network reinforcement learning tractography model

Author: Bryson Gray
2024
"""


import torch
import argparse
from argparse import RawTextHelpFormatter
import json
import numpy as np

from tracking_env import Environment
from dqn_tracker import DQNModel
from load_data import load_data


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    """ Train a deep q network.

    Parameters
    ----------
    args : json
    """
    img_file = args["image"]
    label_file = args["label"]
    seeds = args["seeds"]
    model = args["model"] if "model" in args else None
    n_seeds = args["n_seeds"] if "n_seeds" in args else 1
    step_size = args["step_size"] if "step_size" in args else 1.0
    step_width = args["step_width"] if "step_width" in args else 1.0
    batch_size = args["batchsize"] if "batchsize" in args else 128
    gamma = args["gamma"] if "gamma" in args else 0.99
    eps_start = args["eps_start"] if "eps_start" in args else 0.9
    eps_end = args["eps_end"] if "eps_end" in args else 0.01
    eps_decay = args["eps_decay"] if "eps_decay" in args else 1000
    tau = args["tau"] if "tau" in args else 0.001
    lr = args["lr"] if "lr" in args else 5e-4
    alpha = args["alpha"] if "alpha" in args else 1.0
    beta = args["beta"] if "beta" in args else 1e-3
    friction = args["friction"] if "friction" in args else 1e-4
    num_episodes = args["num_episodes"] if "num_episodes" in args else 100
    pixelsize = args["pixelsize"] if "pixelsize" in args else [1.0, 1.0, 1.0]
    patch_radius = 17
    actions = np.load('/home/brysongray/tractography/neuron_trx/action_space_6_dir.npy')

    img, density, mask = load_data(img_file, label_file, pixelsize=pixelsize, inverse=True)


    env = Environment(img,
                      patch_radius,
                      seeds,
                      mask,
                      density,
                      actions,
                      n_seeds=n_seeds,
                      step_size=step_size,
                      step_width=step_width,
                      max_len=10000,
                      alpha=alpha,
                      beta=beta,
                      friction=friction)

    dqn_model = DQNModel(in_channels=4,
                         n_actions=len(actions),
                         input_size=(2*patch_radius+1),
                         lr=lr,
                         step_size=torch.tensor(step_size))

    if model:
        dqn_model.load_model(torch.load(model))

    dqn_model.train(env,
                    episodes=num_episodes,
                    batch_size=batch_size,
                    gamma=gamma,
                    tau=tau,
                    eps_start=eps_start,
                    eps_end=eps_end,
                    eps_decay=eps_decay,
                    save_snapshots=True,
                    show=False,
                    name=f'alpha-{alpha:.1f}_b-{beta:.1f}_f-{friction:.1f}_n-{n_seeds}')

    return

if __name__ == "__main__":

    help_string = "Arg parser looks for one argument, \'--config\', which is a JSON file with the following entries: \n\
    Required\n\
    --------\n\
    \"image\": The path to the input image.\n\
    \"label\": The path to the label file with extension 'swc'.\n\
    \"seed\": A list of scalars with length four. First three values are image coordinates and the last is the radius within which to sample N seeds.\n\
    Optional\n\
    --------\n\
    \"model\": The path to pretrained model weights.\n\
    \"n_seeds\": The number of seed points to use, randomly placed inside the radius centered around the seed point. Default: 1\n\
    \"batchsize\": Number of transitions sampled from the replay buffer. Default: 128\n\
    \"gamma\": Return discount factor. Takes values between 0 and 1. Larger values give more weight to long-term reward. Default: 0.99\n\
    \"eps_start\": Starting value of epsilon, which controls the likelihood of taking random actions. Epsilon decays\n\
                   exponentially over the course of training. Default: 0.9\n\
    \"eps_end\": The value that epsilon converges to as the number of steps taken increases over training. Default: 0.01\n\
    \"eps_decay\": The rate of exponetial decay of epsilon. A greater value means slower decay. Default: 1000\n\
    \"tau\":  The update rate of the target network. Default: 1e-3\n\
    \"lr\": The learning rate of the optimizer. Default: 1e-4"

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--config', nargs=1,
                        help=help_string,
                        type=argparse.FileType('r'))
    
    args = json.load(parser.parse_args().config[0])

    main(args)
