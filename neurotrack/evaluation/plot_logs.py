from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_logs(logs_csv: Path, output_dir: Path):
    # CSV fields expected: episode,image_file,num_branches,episode_avg_reward,episode_avg_var,episode_return,episode_avg_policy_loss,moving_avg_reward,complexity
    # We want to plot moving_avg_reward, episode_return and episode_avg_policy_loss over episodes
    logs = []
    with open(logs_csv, "r") as f:
        next(f)  # Skip header
        for line in f:
            fields = line.strip().split(",")
            log = {
                "episode": int(fields[0]),
                "image_file": fields[1],
                "num_branches": int(fields[2]),
                "episode_avg_reward": float(fields[3]),
                "episode_avg_var": float(fields[4]),
                "episode_return": float(fields[5]),
                "episode_avg_policy_loss": float(fields[6]),
                "moving_avg_reward": float(fields[7]),
                "complexity": float(fields[8])
            }
            logs.append(log)

    # Sort logs by episode
    logs.sort(key=lambda x: x["episode"])
    episodes = [log["episode"] for log in logs]
    moving_avg_rewards = [log["moving_avg_reward"] for log in logs]
    episode_returns = [log["episode_return"] for log in logs]
    episode_avg_policy_losses = [log["episode_avg_policy_loss"] for log in logs]

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(episodes, moving_avg_rewards, label="Moving Avg Reward")
    plt.xlabel("Episode")
    plt.ylabel("Moving Avg Reward")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(episodes, episode_returns, label="Episode Return", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(episodes, episode_avg_policy_losses, label="Episode Avg Policy Loss", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Episode Avg Policy Loss")
    plt.legend()
    plt.tight_layout()
    output_file = output_dir / f"training_logs_plot_{logs_csv.stem}.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Plot training logs")
    parser.add_argument("--logs_csv", type=Path, required=True, help="Path to the logs CSV file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the output plot")
    args = parser.parse_args()

    plot_logs(args.logs_csv, args.output_dir)