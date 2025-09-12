import numpy as np
import matplotlib.pyplot as plt

def plot_distribution(data, var_cfg, title="Distribution", output_path=None, log_scale=False, color="skyblue"):
    """Plot a flat distribution (histogram + boxplot) for one variable."""
    unit = var_cfg["unit"]
    vmin = var_cfg["vmin"]
    vmax = var_cfg["vmax"]

    # flatten data if it's multi-dimensional
    if len(data.shape) > 1:
        data = data.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram
    axes[0].hist(data, bins=30, range=(vmin, vmax), color=color, edgecolor="black", log=log_scale)
    axes[0].set_title(f"{title} - Histogram")
    axes[0].set_xlabel(f"{var_cfg['long_name']} [{unit}]")
    axes[0].set_ylabel("Frequency")

    # Boxplot
    axes[1].boxplot(data, vert=True)
    axes[1].set_title(f"{title} - Boxplot")
    axes[1].set_ylabel(f"{var_cfg['long_name']} [{unit}]")
    axes[1].set_ylim(vmin, vmax)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_grouped_distribution(data, var_cfg, n_frame=8, title="Grouped Distribution"):
    """
    Group data into samples of n_frame, then collect values for each frame index across samples.
    Plot distributions (boxplots).
    """
    unit = var_cfg["unit"]
    vmin = var_cfg["vmin"]
    vmax = var_cfg["vmax"]

    n_samples = len(data) // n_frame
    reshaped = np.array(data[:n_samples * n_frame]).reshape(n_samples, n_frame)

    frame_distributions = [reshaped[:, i] for i in range(n_frame)]

    plt.figure(figsize=(10, 5))
    plt.boxplot(frame_distributions, positions=range(1, n_frame + 1))
    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel(f"{var_cfg['long_name']} [{unit}]")
    plt.ylim(vmin, vmax)
    plt.show()


def plot_vertical_profile(data_by_level, var_cfg, levels, title="Vertical Profile"):
    """
    Plot mean ± std across pressure levels.
    data_by_level should be dict {level: array of values}.
    """
    unit = var_cfg["unit"]

    means = []
    stds = []
    ordered_levels = sorted(levels, reverse=True)  # High altitude (low pressure) at top

    for lvl in ordered_levels:
        values = np.array(data_by_level[lvl])
        means.append(values.mean())
        stds.append(values.std())

    means = np.array(means)
    stds = np.array(stds)

    plt.figure(figsize=(6, 6))
    plt.plot(means, ordered_levels, marker="o", label="Mean")
    plt.fill_betweenx(ordered_levels, means - stds, means + stds, alpha=0.3, label="±1 std")
    plt.gca().invert_yaxis()  # pressure decreases upwards
    plt.title(title)
    plt.xlabel(f"{var_cfg['long_name']} [{unit}]")
    plt.ylabel("Pressure level (hPa)")
    plt.legend()
    plt.show()
