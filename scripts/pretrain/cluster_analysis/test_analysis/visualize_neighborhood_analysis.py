import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import entropy
from matplotlib.lines import Line2D
import sys
import seaborn as sns
import cmcrameri.cm as cmc

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from utils.plotting.class_colors import CLOUD_CLASS_INFO


# =========================
# PATHS
# =========================
run_name = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
base_path = f"/data1/fig/{run_name}/epoch_800/test_traj"
output_dir = os.path.join(base_path, "hypersphere_analysis")

neighbors_csv = os.path.join(
    output_dir,
    "test_vectors_local_label_composition.csv"
)

tsne_csv = os.path.join(
    base_path,
    "tsne_all_vectors_with_centroids.csv"
)

df = pd.read_csv(neighbors_csv)
df_tsne = pd.read_csv(tsne_csv)
df_tsne_test = df_tsne[(df_tsne["vector_type"] != "TRAIN") & (df_tsne["vector_type"] != "CENTROID")]

# merge tsne coords
df = df.merge(
    df_tsne_test[["filename", "tsne_dim_1", "tsne_dim_2"]],
    on="filename",
    how="left"
)

labels_ordered = sorted(
    CLOUD_CLASS_INFO.keys(),
    key=lambda x: CLOUD_CLASS_INFO[x]["order"]
)
print(labels_ordered)

colors = {l: CLOUD_CLASS_INFO[l]["color"] for l in labels_ordered}
short = {l: CLOUD_CLASS_INFO[l]["short"] for l in labels_ordered}
print(colors)
print(short)

perc_cols = [f"wperc_label_{l}" for l in labels_ordered]
print(perc_cols)

#df["dominant_label"] = df[perc_cols].idxmax(axis=1).str.replace("perc_label_", "")
#df["dominance"] = df[perc_cols].max(axis=1)

df["entropy"] = df[perc_cols].apply(
    lambda x: entropy(x / 100.0) if np.isfinite(x).all() else np.nan,
    axis=1
)


datetimes = [ fname.split("_")[1] for fname in df["filename"] ]
df["datetime"] = pd.to_datetime(datetimes, format="%Y-%m-%dT%H-%M")

df["hour"] = df["datetime"].dt.hour
df["day_night"] = np.where(df["hour"].between(8, 16), "DAY", "NIGHT")
df["region"] = np.where(df["lat"] >= 47, "NORTH", "SOUTH")



INTERESTING_CLASSES = ["EC", "DC", "OA"]
LABEL_NAME_MAP = {
    "EC": "Early Convection",
    "DC": "Deep Convection",
    "OA": "Overcast Anvil"
}
#get labels ordered interesting from short dict {3: 'CS', 0: 'SC', 6: 'MC1', 5: 'MC2', 1: 'EC', 2: 'DC', 4: 'OA'}
label_interesting_short = [
    key for key, lbl in short.items()
    if lbl in INTERESTING_CLASSES
]
colors_interesting = {k: colors[k] for k in label_interesting_short}
print(label_interesting_short)
print(colors_interesting)


print(df.columns.tolist())

palette = {
    f"wperc_label_{k}": v
    for k, v in colors.items()
}

DIVIDE_BY = None #REGION or DAY_NIGHT or ALL

EVENT_GROUPS = ['ALL']#['PRECIP', 'HAIL', 'MIXED']

if DIVIDE_BY == 'REGION':
    #create a dict with df for norht and south
    df_groups = {
        'NORTH': df[df['region'] == 'NORTH'],
        'SOUTH': df[df['region'] == 'SOUTH']
    }
elif DIVIDE_BY == 'DAY_NIGHT':
    df_groups = {
        'DAY': df[df['day_night'] == 'DAY'],
        'NIGHT': df[df['day_night'] == 'NIGHT']
    }
else:
    df_groups = {
        'ALL': df
    }   

for group_name, df_group in df_groups.items():
    for event in EVENT_GROUPS:
        if event == 'ALL':
            df_event = df_group
        else:
            df_event = df_group[df_group['storm_type'] == event]

        n_classes = len(INTERESTING_CLASSES)
        fig, axes = plt.subplots(
            n_classes, 1,
            figsize=(2, 1.5 * n_classes),
            sharex=True,
        )
        #increase space between subplots
        fig.subplots_adjust(hspace=0.4)

        fig_sc, axes_sc = plt.subplots(
            1, n_classes,
            figsize=(3 * n_classes, 2.5),
            sharey=True
        )

        if n_classes == 1:
            axes = [axes]
            axes_sc = [axes_sc]

        df_meltered = df_event[df_event["label"].isin(label_interesting_short)].melt(
            id_vars=["filename", "label"],
            value_vars=perc_cols,
            var_name="neighbor_label",
            value_name="probability"
        )
        print(df_meltered)


        for i, cls in enumerate(label_interesting_short):
            ax = axes[i]
            ax_sc = axes_sc[i]

            df_cls = df_meltered[df_meltered["label"] == cls]
            df_cls_scatter = df_event[df_event["label"] == cls]

            x = df_cls_scatter["dominance"].values
            y = df_cls_scatter["entropy"].values

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]


            sns.boxplot(
                data=df_cls,
                x="neighbor_label",
                y="probability",
                hue="neighbor_label",
                ax=ax,
                palette=palette,
                legend=False,
                showfliers=False,
                width=0.6
            )

            sns.scatterplot(
                    data=df_cls_scatter,
                    x="dominance",
                    y="entropy",
                    hue="crop_type",
                    palette=["#66c2a5", "#fc8d62", "#8da0cb"], #set manually 3 colors
                    ax=ax_sc,
                    legend=False,
                    s=10,
                    alpha=0.6
                )

            #set the legend to the last axis
            if i == n_classes - 1:
                #get unique crop types in the data
                crop_types = ['observed', 'interpolated', 'extrapolated']
                #handles just points with the color of each crop type
                handles = [
                    Line2D(
                        [0], [0],
                        marker="o",
                        color="w",
                        markerfacecolor=["#66c2a5", "#fc8d62", "#8da0cb"][j],
                        markersize=6,
                        label=crop_type
                    )
                    for j, crop_type in enumerate(crop_types)
                ]
                #labels_legend = crop_types.tolist()
                print("handles:", handles)
                print("labels_legend:", crop_types)
                ax_sc.legend(
                    handles,
                    crop_types,
                    title="Crop Type",
                    fontsize=8,
                    title_fontsize=10,
                    loc="upper right",
                    bbox_to_anchor=(1.3, 1)
                )

            if len(x) > 10:
                print("enough points for quadratic fit")
                # 1st-degree polynomial fit
                coef = np.polyfit(x, y, deg=1)
                a, b = coef  # y = a x^2 + b x + c

                x_line = np.linspace(x.min(), x.max(), 300)
                y_line = a * x_line + b

                ax_sc.plot(
                    x_line,
                    y_line,
                    color="black",
                    linestyle="--",
                    linewidth=2,
                    label="1st-degree fit"
                )

                y_pred = a * x + b
                residuals = y - y_pred
                std_res = residuals.std()


                ax_sc.fill_between(
                    x_line,
                    y_line - std_res,
                    y_line + std_res,
                    color="black",
                    alpha=0.15,
                    label="±1σ"
                )


                text = (
                    #f"2nd-deg fit\n"
                    #f"Curvature: {curvature}\n"
                    f"⟨slope⟩ = {a:.2f}\n"
                    f"σ(res) = {std_res:.2f}"
                )
            #put text in the lower left corner of the scatter plot
            ax_sc.text(
                0.05, 0.05,
                text,
                transform=ax_sc.transAxes,
                va="bottom",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            ax.set_title(
                LABEL_NAME_MAP.get(short[cls], short[cls]),
                fontsize=11
            )

            ax_sc.set_title(
                LABEL_NAME_MAP.get(short[cls], short[cls]),
                fontsize=11, fontweight="bold"
            )

            ax_sc.set_ylim(0,1.6)
            ax_sc.set_xlim(30,100)
            
            ax.set_xlabel("")
            if i == n_classes - 1:
                ax.set_xticks(np.arange(len(labels_ordered)))
                ax.set_xticklabels(
                    [short[l].upper() for l in labels_ordered],
                    rotation=45, fontsize=10
                )
            else:
                ax.set_xticks(np.arange(len(labels_ordered)))
                ax.set_xticklabels([])

            if i == n_classes // 2:
                ax.set_ylabel("Dominance", fontsize=11)
            else:
                ax.set_ylabel("")
            #set y tickes label with FixedLocator to have 5 ineger values
            ax.set_yticks(np.linspace(0, 100, 5))
            #ax.set_yticklabels(np.arange(0, 110, 20), fontsize=10)

            ax.grid(True, axis="y", alpha=0.3)

            #add an horizontal line at y=75 to highlight the dominance threshold
            ax.axhline(
                75, color="black", linestyle="--", linewidth=1, alpha=0.7
            )
            #highlight the area above the line with a light color
            # ax.fill_between(
            #     x=[-0.5, len(labels_ordered)-0.5],
            #     y1=75, y2=100,
            #     color="black", alpha=0.05
            # )

        # fig.suptitle(
        #     f"Event: {event}",
        #     fontsize=14, y=1.07
        # )
        #reduce space between subplots
        #fig.subplots_adjust(wspace=0.15)

        #add title to the figure with the group name and event
        fig.suptitle(
            f"c) Percentage of neighbor labels",
            fontsize=11, fontweight="bold", y=1.
        )

        fig.savefig(
            os.path.join(output_dir, f"neighborhood_prob_boxplots_{DIVIDE_BY}_{group_name}_{event}.png"),
            dpi=200, bbox_inches="tight"
        )
        

        fig_sc.suptitle(
            f"Event: {event}",
            fontsize=14, y=1.07
        )

        fig_sc.savefig(
            os.path.join(output_dir, f"dominance_entropy_{DIVIDE_BY}_{group_name}_{event}.png"),
            dpi=200, bbox_inches="tight"
        )
        plt.close()

"""
plt.figure(figsize=(8, 7))

for lbl in labels_ordered:
    mask = df["dominant_label"] == lbl
    plt.scatter(
        df.loc[mask, "tsne_dim_1"],
        df.loc[mask, "tsne_dim_2"],
        c=colors[lbl],
        s=5,
        alpha=df.loc[mask, "dominance"] / 100,
        label=short[lbl]
    )

plt.legend(markerscale=3)
plt.title("t-SNE colored by dominant neighbor label\n(alpha = dominance)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tsne_dominance.png"), dpi=300)
plt.close()


plt.figure(figsize=(8, 7))
sc = plt.scatter(
    df["tsne_dim_1"],
    df["tsne_dim_2"],
    c=df["entropy"],
    s=6,
    cmap="viridis"
)

plt.colorbar(sc, label="Neighborhood entropy")
plt.title("Local label entropy in embedding space")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tsne_entropy.png"), dpi=300)
plt.close()


def boxplot_metric(metric, ylabel, fname):
    data = [
        df.loc[df["label"] == l, metric].dropna()
        for l in labels_ordered
    ]

    plt.figure(figsize=(9, 4))
    bp = plt.boxplot(data, showfliers=False)

    for patch, l in zip(bp["boxes"], labels_ordered):
        patch.set_color(colors[l])

    plt.xticks(
        range(1, len(labels_ordered) + 1),
        [short[l] for l in labels_ordered]
    )
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()


boxplot_metric("dominance", "Dominance (%)", "dominance_per_label.png")
boxplot_metric("entropy", "Entropy", "entropy_per_label.png")


for metric in ["dominance", "entropy"]:
    plt.figure(figsize=(7, 4))

    for dn, ls in zip(["DAY", "NIGHT"], ["solid", "dashed"]):
        vals = [
            df.loc[
                (df["label"] == l) & (df["day_night"] == dn),
                metric
            ].median()
            for l in labels_ordered
        ]
        plt.plot(
            labels_ordered,
            vals,
            label=dn,
            linestyle=ls,
            marker="o"
        )

    plt.xticks(labels_ordered, [short[l] for l in labels_ordered])
    plt.ylabel(metric)
    plt.legend()
    plt.title(f"{metric} – Day vs Night")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_day_night.png"), dpi=300)
    plt.close()



for metric in ["dominance", "entropy"]:
    plt.figure(figsize=(7, 4))

    for hemi in ["NORTH", "SOUTH"]:
        vals = [
            df.loc[
                (df["label"] == l) & (df["hemisphere"] == hemi),
                metric
            ].median()
            for l in labels_ordered
        ]
        plt.plot(vals, label=hemi, marker="o")

    plt.xticks(range(len(labels_ordered)), [short[l] for l in labels_ordered])
    plt.ylabel(metric)
    plt.legend()
    plt.title(f"{metric} – North vs South")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_north_south.png"), dpi=300)
    plt.close()


df = df.sort_values(["filename", "datetime"])

traj_metrics = (
    df.groupby("filename")
    .agg(
        mean_entropy=("entropy", "mean"),
        max_entropy=("entropy", "max"),
        entropy_trend=("entropy", lambda x: x.iloc[-1] - x.iloc[0]),
        mean_dominance=("dominance", "mean"),
        duration=("datetime", lambda x: (x.max() - x.min()).total_seconds() / 3600)
    )
)

traj_metrics.to_csv(
    os.path.join(output_dir, "trajectory_metrics.csv")
)
"""

# “Hail cores have high dominance but low spatial coverage”

# “Nighttime convective systems show higher embedding entropy”

# “Storms that intensify show decreasing entropy before peak”

# “Mixed class is not a cluster but a transition regime”