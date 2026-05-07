from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_logs(logs_csv: Path, output_dir: Path):
    """
    Plot training logs with separate plots for each metric.
    
    An epoch is defined as all episodes until the first image file repeats.
    Each point represents the average value across all episodes in that epoch.
    Creates 4 separate plots:
    - episode_avg_loss
    - episode_avg_step_mse
    - false_stop_rate
    - false_continue_rate
    """
    # Parse CSV: episode,image_file,episode_avg_reward,episode_avg_loss,episode_avg_step_mse,
    #            episode_avg_expert_step_norm,false_stop_rate,false_continue_rate,steps_done,complexity
    epochs_data = []
    current_epoch = {
        "losses": [],
        "step_mses": [],
        "false_stops": [],
        "false_continues": []
    }
    seen_images = set()
    
    with open(logs_csv, "r") as f:
        next(f)  # Skip header
        for line in f:
            fields = line.strip().split(",")
            image_file = fields[1]
            episode_avg_loss = float(fields[3])
            episode_avg_step_mse = float(fields[4])
            false_stop_rate = float(fields[6])
            false_continue_rate = float(fields[7])
            
            # Check if we've seen this image before (marks start of new epoch)
            if image_file in seen_images:
                # Save current epoch and start a new one
                epochs_data.append(current_epoch)
                current_epoch = {
                    "losses": [],
                    "step_mses": [],
                    "false_stops": [],
                    "false_continues": []
                }
                seen_images = set()
            
            seen_images.add(image_file)
            current_epoch["losses"].append(episode_avg_loss)
            current_epoch["step_mses"].append(episode_avg_step_mse)
            current_epoch["false_stops"].append(false_stop_rate)
            current_epoch["false_continues"].append(false_continue_rate)
    
    # Don't forget the last epoch
    if current_epoch["losses"]:
        epochs_data.append(current_epoch)
    
    # Aggregate: calculate mean for each epoch
    epoch_numbers = list(range(1, len(epochs_data) + 1))
    avg_losses = [np.mean(e["losses"]) for e in epochs_data]
    avg_step_mses = [np.mean(e["step_mses"]) for e in epochs_data]
    avg_false_stops = [np.mean(e["false_stops"]) for e in epochs_data]
    avg_false_continues = [np.mean(e["false_continues"]) for e in epochs_data]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    # Plot 1: episode_avg_loss
    axes[0].plot(epoch_numbers, avg_losses, label="Episode Avg Loss", color="blue", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Episode Avg Loss")
    axes[0].set_title("Average Loss per Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: episode_avg_step_mse
    axes[1].plot(epoch_numbers, avg_step_mses, label="Episode Avg Step MSE", color="green", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Episode Avg Step MSE")
    axes[1].set_title("Average Step MSE per Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: false_stop_rate
    axes[2].plot(epoch_numbers, avg_false_stops, label="False Stop Rate", color="red", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("False Stop Rate")
    axes[2].set_title("Average False Stop Rate per Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: false_continue_rate
    axes[3].plot(epoch_numbers, avg_false_continues, label="False Continue Rate", color="purple", linewidth=2)
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("False Continue Rate")
    axes[3].set_title("Average False Continue Rate per Epoch")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f"training_logs_plot_{logs_csv.stem}.png"
    plt.savefig(output_file, dpi=100)
    plt.close()
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Plot training logs")
    parser.add_argument("--logs_csv", type=Path, required=True, help="Path to the logs CSV file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the output plot")
    args = parser.parse_args()

    plot_logs(args.logs_csv, args.output_dir)