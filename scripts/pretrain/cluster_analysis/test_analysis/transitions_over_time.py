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
from scripts.pretrain.transitions.compute_transitions_utils import (
    check_continuous_timestamps
)


# =========================
# PATHS
# =========================
run_name = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
base_path = f"/data1/fig/{run_name}/epoch_800/test_traj"
output_dir = os.path.join(base_path, "hypersphere_analysis")
plot_dir = os.path.join(output_dir, "trajectory_plots")
os.makedirs(plot_dir, exist_ok=True)

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

df["dominance"] = df[perc_cols].max(axis=1) 


datetimes = [ fname.split("_")[1] for fname in df["filename"] ]
df["datetime"] = pd.to_datetime(datetimes, format="%Y-%m-%dT%H-%M")

df["hour"] = df["datetime"].dt.hour
df["day_night"] = np.where(df["hour"].between(8, 16), "DAY", "NIGHT")
df["region"] = np.where(df["lat"] >= 47, "NORTH", "SOUTH")
print(df.columns.tolist())

#make storm_id column removing timestamp (first part after first underscore)
df["storm_id"] = df["filename"].str.split("_").str[0]

# ---------------------------------
# trajectory-aligned time
# ---------------------------------

aligned_times = []

for storm_id, g in df.groupby("storm_id"):
    g = g.sort_values("datetime")
    t0 = g["datetime"].iloc[0]
    t1 = g["datetime"].iloc[-1]
    t_center = t0 + (t1 - t0) / 2

    aligned = (g["datetime"] - t_center).dt.total_seconds() / 3600.0
    aligned_times.append(aligned)

df["t_aligned"] = pd.concat(aligned_times).sort_index()

BIN_HOURS = 1.0
df["t_bin"] = (df["t_aligned"] / BIN_HOURS).round() * BIN_HOURS

#get max and min t_bin
t_bin_min = df["t_bin"].min()
t_bin_max = df["t_bin"].max()
print(f"t_bin_min: {t_bin_min}, t_bin_max: {t_bin_max}")

#add duration column (in hours) for each row
#fileter out extrapolated crop type 
df_no_extrap = df[df['crop_type'] != 'extrapolated'].copy()
df_no_extrap["duration"] = df_no_extrap.groupby("storm_id")["datetime"].transform(lambda x: (x.max() - x.min()).total_seconds() / 3600.0)


def compute_transitions_over_time(df):

    #build transition probability over aligned time bins
    df = df.sort_values(["storm_id", "datetime"])

    df["next_label"] = df.groupby("storm_id")["label"].shift(-1)
    df["next_t"] = df.groupby("storm_id")["t_aligned"].shift(-1)

    # transition time = midpoint
    df["t_trans"] = 0.5 * (df["t_aligned"] + df["next_t"])

    df_trans = df.dropna(subset=["next_label", "t_trans"])

    df_trans["t_bin"] = (df_trans["t_trans"] / BIN_HOURS).round() * BIN_HOURS

    trans_counts = (
        df_trans
        .groupby(["t_bin", "label", "next_label"])
        .size()
        .rename("count")
        .reset_index()
    )

    #normalize only if enough samples in the bin, otherwise set prob to nan
    tot_counts_bin = (
        trans_counts
        .groupby(["t_bin", "label"])["count"]
        .transform("sum")
    )
    mask_enough = tot_counts_bin >= 20
    trans_counts = trans_counts[mask_enough].copy()
    trans_counts["prob"] = (
        trans_counts["count"]
        / trans_counts.groupby(["t_bin", "label"])["count"].transform("sum")
    )

    return trans_counts



def compute_transitions_over_time_with_CI(
    df,
    bin_hours,
    min_count=20,
    n_bootstrap=300,
    ci_level=95,
):

    # --- prepare base dataframe ---
    df = df.sort_values(["storm_id", "datetime"]).copy()

    df["next_label"] = df.groupby("storm_id")["label"].shift(-1)
    df["next_t"] = df.groupby("storm_id")["t_aligned"].shift(-1)
    df["t_trans"] = 0.5 * (df["t_aligned"] + df["next_t"])

    df = df.dropna(subset=["next_label", "t_trans"])

    df["t_bin"] = (df["t_trans"] / bin_hours).round() * bin_hours

    storm_ids = df["storm_id"].unique()

    # ==========================
    # Point estimate
    # ==========================
    base_counts = (
        df.groupby(["t_bin", "label", "next_label"])
        .size()
        .rename("count")
        .reset_index()
    )

    base_tot = (
        base_counts
        .groupby(["t_bin", "label"])["count"]
        .transform("sum")
    )

    base_counts = base_counts[base_tot >= min_count].copy()
    base_counts["prob"] = (
        base_counts["count"]
        / base_counts.groupby(["t_bin", "label"])["count"].transform("sum")
    )

    # ==========================
    # Bootstrap
    # ==========================
    boot_probs = []

    for _ in range(n_bootstrap):
        sampled_ids = np.random.choice(storm_ids, len(storm_ids), replace=True)
        df_b = df[df["storm_id"].isin(sampled_ids)]

        counts_b = (
            df_b.groupby(["t_bin", "label", "next_label"])
            .size()
            .rename("count")
            .reset_index()
        )

        tot_b = (
            counts_b
            .groupby(["t_bin", "label"])["count"]
            .transform("sum")
        )

        counts_b = counts_b[tot_b >= min_count].copy()
        counts_b["prob"] = (
            counts_b["count"]
            / counts_b.groupby(["t_bin", "label"])["count"].transform("sum")
        )

        boot_probs.append(
            counts_b[["t_bin", "label", "next_label", "prob"]]
        )

    boot_df = pd.concat(boot_probs, keys=range(n_bootstrap), names=["boot"])

    # ==========================
    # CI computation
    # ==========================
    alpha = (100 - ci_level) / 2

    ci_df = (
        boot_df
        .groupby(["t_bin", "label", "next_label"])["prob"]
        .agg(
            prob_mean="mean",
            prob_ci_low=lambda x: np.nanpercentile(x, alpha),
            prob_ci_high=lambda x: np.nanpercentile(x, 100 - alpha),
            n_samples="count",
        )
        .reset_index()
    )

    return ci_df




def mean_std_by_time(df, value_col):
    return (
        df.groupby(["t_bin", "label"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

def population_by_time(df):
    #total = df.groupby("t_bin").size()
    counts = df.groupby(["t_bin", "label"]).size()
    #print(counts.reset_index(name="counts"))
    #pop = (counts / total).reset_index(name="fraction")
    return counts.reset_index(name="counts")




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

df_sum = check_continuous_timestamps(df, id_col="storm_id", time_col="datetime", freq='15min')
print(df_sum)
#howm many entries have is_continuous == True
n_continuous = df_sum['is_continuous'].sum()
print(f"Number of continuous trajectories: {n_continuous} out of {len(df_sum)}")
#filter out non continuous trajectories
df = df[df['storm_id'].isin(df_sum[df_sum['is_continuous']==True]['storm_id'])]
print(df)

palette = {
    f"wperc_label_{k}": v
    for k, v in colors.items()
}

DIVIDE_BY = 'ALL' #REGION or DAY_NIGHT or ALL

EVENT_GROUPS = ['ALL', 'PRECIP', 'HAIL', 'MIXED']

if DIVIDE_BY == 'REGION':
    #create a dict with df for norht and south
    df_groups = {
        'NORTH': df[df['region'] == 'NORTH'],
        'SOUTH': df[df['region'] == 'SOUTH']
    }
    df_groups_no_extrap = {
        'NORTH': df_no_extrap[df_no_extrap['region'] == 'NORTH'],
        'SOUTH': df_no_extrap[df_no_extrap['region'] == 'SOUTH']
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
    df_groups_no_extrap = {
        'ALL': df_no_extrap
    }  

for group_name, df_group in df_groups.items():
    for event in EVENT_GROUPS:
        if event == 'ALL':
            df_event = df_group
            df_event_no_extrap = df_groups_no_extrap[group_name]
        else:
            df_event = df_group[df_group['storm_type'] == event]
            df_event_no_extrap = df_groups_no_extrap[group_name][df_groups_no_extrap[group_name]['storm_type'] == event]

        n_classes = len(INTERESTING_CLASSES)
        fig, axes = plt.subplots(
            3, 1,
            figsize=(7, 5), #reduce height
            sharex=True,#make ratio 0.5, 1, 1
            gridspec_kw={'height_ratios': [0.75, 1, 1]}
        )

        fig_dur, ax_dur = plt.subplots(
            figsize=(4, 3)
        )

        fig_trans, axes_trans = plt.subplots(
            2, 1,
            figsize=(4, 3),
            sharex=True,    
            gridspec_kw={'height_ratios': [1, 1]}
        )
        

        if n_classes == 1:
            axes = [axes]

        #plot duration histogram
        #get only one duration per storm_id
        df_event_unique = df_event_no_extrap.drop_duplicates(subset=["storm_id"])
        ax_dur.hist(
            df_event_unique["duration"],
            bins=np.arange(0, df_no_extrap["duration"].max() + 1, 1),
            color='blue',
            alpha=0.7
        )
        #set axis labels
        ax_dur.set_xlabel("Storm Duration (hours)", fontsize=12)
        ax_dur.set_ylabel("Counts", fontsize=12)
        ax_dur.set_title(f"{event} storms - {group_name} group", fontsize=12, fontweight="bold")
        #set y ticks label fontsize
        ax_dur.tick_params(axis='y', labelsize=10)
        #set x ticks label fontsize
        ax_dur.tick_params(axis='x', labelsize=10)
        #set y lim
        y_max_dur = 250
        ax_dur.set_ylim(0, y_max_dur)

        df_event_trans = compute_transitions_over_time_with_CI(df_event, BIN_HOURS, min_count=20, n_bootstrap=300, ci_level=95)
        #print(df_event_trans.head())

        persist = df_event_trans[
                (df_event_trans["label"] == df_event_trans["next_label"]) &
                (df_event_trans["label"].isin(label_interesting_short))
            ]

        for cls in label_interesting_short:
            d = persist[persist["label"] == cls]

            axes_trans[0].plot(
                d["t_bin"],
                d["prob_mean"],
                color=colors_interesting[cls],
                lw=2
            )

            axes_trans[0].fill_between(
                d["t_bin"],
                d["prob_ci_low"],
                d["prob_ci_high"],
                color=colors_interesting[cls],
                alpha=0.25
            )

        axes_trans[0].axvline(0, ls="--", c="k")
        axes_trans[0].set_ylabel("Persistence \n probability", fontsize=12)
        axes_trans[0].set_ylim(0.5, 1.0)
        #set y ticks to 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        axes_trans[0].set_yticks(np.arange(0.5, 1.00, 0.1))
        #set y ticks label fontsize
        axes_trans[0].tick_params(axis='y', labelsize=10)
        axes_trans[0].set_title(f"{event}", fontsize=12, fontweight="bold")
        axes_trans[0].grid(axis='both', alpha=0.3)
        #set legend with class names and not legend titles
        handles = [
            Line2D(
                [0], [0],
                color=colors_interesting[cls],
                lw=2
            )
            for cls in label_interesting_short
        ]


        axes_trans[0].legend(handles,INTERESTING_CLASSES, title=None, fontsize=8, title_fontsize=10, loc='lower right') #no legend title

        mask = (df_event_trans["label"] == 1) & (df_event_trans["next_label"] == 2)
        d = df_event_trans[mask]

        axes_trans[1].plot(
            d["t_bin"],
            d["prob_mean"],
            color="red",
            lw=2,
            label="EC → DC"
        )

        axes_trans[1].fill_between(
            d["t_bin"],
            d["prob_ci_low"],
            d["prob_ci_high"],
            color="red",
            alpha=0.25
        )


        mask = (df_event_trans["label"] == 2) & (df_event_trans["next_label"] == 4)
        d = df_event_trans[mask]

        axes_trans[1].plot(
            d["t_bin"],
            d["prob_mean"],
            color="blue",
            lw=2,
            label="DC → OA"
        )

        axes_trans[1].fill_between(
            d["t_bin"],
            d["prob_ci_low"],
            d["prob_ci_high"],
            color="blue",
            alpha=0.25
        )


        axes_trans[1].set_ylabel("Transition \n probability", fontsize=12)
        #set y lim to 0 to 0.3
        axes_trans[1].set_ylim(0, 0.5)
        #set tickes to 0, 0.1, 0.2, 0.3, 0.4, 0.5
        axes_trans[1].set_yticks(np.arange(0, 0.5, 0.1))
        #finsize of ticklabels
        axes_trans[1].tick_params(axis='y', labelsize=10)

        #set x tickes ana labels (from -6 to 6 every 2)
        axes_trans[1].set_xlabel("Aligned time (hours)", fontsize=12)
        axes_trans[1].set_xlim(-6, 6)
        axes_trans[1].set_xticks(np.arange(-6, 7, 2))
        axes_trans[1].set_xticklabels([str(x) for x in np.arange(-6, 7, 2)])
        axes_trans[1].tick_params(axis='x', labelsize=10)
        axes_trans[1].axvline(0, ls="--", c="k")
        axes_trans[1].legend(fontsize=8, loc='upper right')
        axes_trans[1].grid(axis='both', alpha=0.3)


        for cls in  label_interesting_short:

            color = colors[cls]
            df_cls = df_event[df_event["label"] == cls]

            # --------------------
            # Population
            # --------------------
            pop = population_by_time(df_event)

            pop_cls = pop[pop["label"] == cls]

            axes[0].plot(
                pop_cls["t_bin"],
                pop_cls["counts"],
                color=color,
                lw=2
            )
            axes[0].set_title(f"{event} storms - {group_name} group", fontsize=12, fontweight="bold")
            axes[0].set_ylabel("Counts", fontsize=12)
            #set 5 y ticks
            y_max = 1200
            axes[0].set_ylim(0, y_max)
            axes[0].set_yticks(np.linspace(0, y_max, 4))
            #set y ticks label fontsize
            axes[0].tick_params(axis='y', labelsize=10)

            # --------------------
            # Entropy
            # --------------------
            ent = mean_std_by_time(df_cls, "entropy")
            #plot only if the count in the bin is > 50
            ent = ent[ent["count"] >= 30]

            axes[1].plot(
                ent["t_bin"],
                ent["mean"],
                color=color,
                lw=2,
                alpha=0.8
            )
            axes[1].fill_between(
                ent["t_bin"],
                ent["mean"] - ent["std"],
                ent["mean"] + ent["std"],
                color=color,
                alpha=0.15
            )
            axes[1].set_ylabel("Entropy", fontsize=12)
            axes[1].set_ylim(0,1.1)
            #set y ticks label fontsize
            axes[1].tick_params(axis='y', labelsize=10)
            #draw 4 ticks in y with FixedLocator
            axes[1].yaxis.set_major_locator(mpl.ticker.FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))

            # --------------------
            # Dominance
            # --------------------
            dom = mean_std_by_time(df_cls, "dominance")
            dom = dom[dom["count"] >= 30]
            axes[2].plot(
                dom["t_bin"],
                dom["mean"],
                color=color,
                lw=2,
                alpha=0.8
            )
            axes[2].fill_between(
                dom["t_bin"],
                dom["mean"] - dom["std"],
                dom["mean"] + dom["std"],
                color=color,
                alpha=0.15
            )
            axes[2].set_ylabel("Dominance", fontsize=12) 
            axes[2].set_ylim(50,101)
            #set y ticks label fontsize
            axes[2].tick_params(axis='y', labelsize=10)
            axes[2].yaxis.set_major_locator(mpl.ticker.FixedLocator([50, 60, 70, 80, 90, 100]))

        # --------------------
        # Final cosmetics
        # --------------------
        axes[2].set_xlabel("Aligned time (hours)", fontsize=12)
        axes[2].set_xlim(-5 - BIN_HOURS, 5 + BIN_HOURS)
        #set ticks every 2 hours
        axes[2].set_xticks(np.arange(-5, 6, 2))
        #set x ticks label fontsize
        axes[2].tick_params(axis='x', labelsize=10)

        for ax in axes.flat:
            ax.axvline(0, color="k", lw=1, ls="--", alpha=0.5)
            ax.grid(alpha=0.3)

        output_path = os.path.join(
            plot_dir,
            f"trajectory_analysis_{group_name}_{event}.png"
        )
        fig.savefig(
            output_path,
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        output_path_dur = os.path.join(
            plot_dir,
            f"trajectory_duration_hist_{group_name}_{event}.png"
        )
        fig_dur.savefig(
            output_path_dur,
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig_dur)\

        output_path_trans = os.path.join(
            plot_dir,
            f"trajectory_transitions_{group_name}_{event}.png"
        )
        fig_trans.savefig(
            output_path_trans,
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig_trans)