import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_dataset(npz_file_path):
    """
    Analyzes the Pelican dataset from a .npz file, calculates stats,
    and generates histograms for data distribution.

    Args:
        npz_file_path (str): The path to the .npz file.
    """
    try:
        with np.load(npz_file_path, allow_pickle=True) as data:
            flights = data['datas']
    except FileNotFoundError:
        print(f"Error: The file {npz_file_path} was not found.")
        return

    # Aggregate data from all flights
    aggregated_data = {}
    for flight in flights:
        for key, value in flight.items():
            if key not in ['len']:  # We don't need stats for 'len'
                if key not in aggregated_data:
                    aggregated_data[key] = []
                aggregated_data[key].append(value.flatten())

    # Consolidate lists of arrays into single arrays
    for key in aggregated_data:
        aggregated_data[key] = np.concatenate(aggregated_data[key])

    # Calculate and print stats
    print("--- Dataset Statistics ---")
    for key, value in aggregated_data.items():
        min_val = np.min(value)
        max_val = np.max(value)
        mean_val = np.mean(value)
        std_val = np.std(value)
        print(f"\nField: {key}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std Dev: {std_val:.4f}")

    # Create a directory for plots
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Generate and save histograms
    print("\n--- Generating Histograms ---")
    for key, value in aggregated_data.items():
        plt.figure(figsize=(10, 6))
        plt.hist(value, bins=50)
        plt.title(f'Distribution of {key}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plot_path = os.path.join(plot_dir, f'{key}_histogram.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved histogram for {key} to {plot_path}")

if __name__ == '__main__':
    npz_file = 'sim_10HZ.npz'
    analyze_dataset(npz_file)
