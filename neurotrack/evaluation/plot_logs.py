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
                "episode": int(fields[2]),
                "image_file": fields[3],
                "episode_avg_reward": float(fields[4]),
                "episode_avg_loss": float(fields[5]),
                "false_stop_rate": float(fields[8]),
                "false_continue_rate": float(fields[9])
            }
            logs.append(log)

    # Sort logs by episode
    logs.sort(key=lambda x: x["episode"])
    episodes = [log["episode"] for log in logs]
    episode_avg_rewards = [log["episode_avg_reward"] for log in logs]
    update_episodes = [log["episode"] for log in logs if log["episode_avg_loss"] > 0.0]
    episode_avg_loss = [log["episode_avg_loss"] for log in logs if log["episode_avg_loss"] > 0.0]
    false_stop_rates = [log["false_stop_rate"] for log in logs if "false_stop_rate" in log]
    false_continue_rates = [log["false_continue_rate"] for log in logs if "false_continue_rate" in log]


    plt.figure(figsize=(16, 8))
    plt.subplot(4, 1, 1)
    plt.plot(episodes, episode_avg_rewards, label="Episode Avg Reward", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Moving Avg Reward")
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(update_episodes, episode_avg_loss, label="Episode Avg Loss", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Episode Avg Loss")
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.scatter(episodes, false_stop_rates, label="False Stop Rate", color="red", s=5)
    plt.xlabel("Episode")
    plt.ylabel("False Stop Rate")
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.scatter(episodes, false_continue_rates, label="False Continue Rate", color="purple", s=5)
    plt.xlabel("Episode")
    plt.ylabel("False Continue Rate")
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