import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import torch
from torch.optim.adamw import AdamW
from torch.optim.adam import Adam

sys.path.append(str(Path(__file__).parents[1]))
from environments.sac_tracking_env import Environment
from memory.buffer import PrioritizedReplayBuffer
from models.resblock import ResidualBlock
from models.resnet import ResNet
from models.cnn import ConvNet
from solvers import sac

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
date = datetime.now().strftime("%m-%d-%y")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--json', type=str, help='Path to input parameters json file.')
    args = parser.parse_args()
    args_json = args.json
    with open(args_json) as f:
        params = json.load(f)
    
    img_path = params["img_path"]
    outdir = params["outdir"]
    name = params["name"]
    model = params["model"] if "model" in params else None
    n_seeds = params["n_seeds"] if "n_seeds" in params else 1
    step_size = params["step_size"] if "step_size" in params else 1.0
    step_width = params["step_width"] if "step_width" in params else 1.0
    batch_size = params["batchsize"] if "batchsize" in params else 256
    tau = params["tau"] if "tau" in params else 0.005
    gamma = params["gamma"] if "gamma" in params else 0.99
    lr = params["lr"] if "lr" in params else 0.001
    alpha = params["alpha"] if "alpha" in params else 1.0
    beta = params["beta"] if "beta" in params else 1e-3
    friction = params["friction"] if "friction" in params else 1e-4
    n_episodes = params["n_episodes"] if "n_episodes" in params else 100
    init_temperature = params["init_temperature"] if "init_temperature" in params else 0.005
    target_entropy = params["target_entropy"] if "target_entropy" in params else 0.0
    patch_radius = 17

    if "classifier_weights" in params:
        classifier_path = params["classifier_weights"]
        classifier_state_dict = torch.load(classifier_path, weights_only=True)
        classifier = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=1)
        classifier = classifier.to(device=DEVICE, dtype=dtype)
        classifier.load_state_dict(classifier_state_dict)
        classifier.eval()
    else:
        classifier = None

    env = Environment(img_path,
                    radius=patch_radius,
                    step_size=step_size,
                    step_width=step_width,
                    max_len=10000,
                    alpha=alpha,
                    beta=beta,
                    friction=friction,
                    classifier=classifier)
    
    in_channels = 4
    input_size = 2*patch_radius+1
    init_temperature = 0.005
    actor = ConvNet(chin=in_channels, chout=4)
    actor = actor.to(device=DEVICE,dtype=dtype)

    Q1 = ConvNet(chin=in_channels+3,chout=1)
    Q1 = Q1.to(device=DEVICE,dtype=dtype)
    Q2 = ConvNet(chin=in_channels+3,chout=1)
    Q2 = Q2.to(device=DEVICE,dtype=dtype)
    Q1_target = ConvNet(chin=7,chout=1)
    Q1_target = Q1_target.to(device=DEVICE,dtype=dtype)
    Q2_target = ConvNet(chin=7,chout=1)
    Q2_target = Q2_target.to(device=DEVICE,dtype=dtype)

    if "sac_weights" in params:
        sac_path = params["sac_weights"]
        state_dicts = torch.load(sac_path, weights_only=True)
        actor.load_state_dict(state_dicts["policy_state_dict"])
        Q1.load_state_dict(state_dicts["Q1_state_dict"])
        Q2.load_state_dict(state_dicts["Q2_state_dict"])

    Q1_target.load_state_dict(Q1.state_dict())
    Q2_target.load_state_dict(Q2.state_dict())

    log_alpha = torch.log(torch.tensor(init_temperature).to(DEVICE))
    log_alpha.requires_grad = True
    Q1_optimizer = AdamW(Q1.parameters(), lr=lr)
    Q2_optimizer = AdamW(Q2.parameters(), lr=lr)
    actor_optimizer = AdamW(actor.parameters(), lr=lr)
    log_alpha_optimizer = Adam([log_alpha], lr=lr)

    criterion = torch.nn.MSELoss()
    memory = PrioritizedReplayBuffer(100000, obs_shape=(in_channels,input_size,input_size,input_size), action_shape=(3,), alpha=0.8)

    sac.train(env, actor, Q1, Q2, Q1_target, Q2_target, log_alpha,
          actor_optimizer, Q1_optimizer, Q2_optimizer, log_alpha_optimizer,
          memory, target_entropy, batch_size, gamma, tau, outdir, name,
          show_states=True, save_snapshots=False, update_after=256,
          updates_per_step=1, update_every=1, n_episodes=n_episodes, n_trials=5)
    
    print("Done!")