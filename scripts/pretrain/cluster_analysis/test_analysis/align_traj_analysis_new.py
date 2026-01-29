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

#open the variable csv stats
stats_csv = os.path.join(
    base_path,"crops_stats_vars-cth-cma-precipitation-euclid_msg_grid_stats-50-99-25-75_frames-1_coords-datetime_dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc_all_0.csv"
)

df_stats = pd.read_csv(stats_csv)
#print(df_stats.columns.tolist())    
#print(df_stats)
#change name of column from time to datetime
df_stats = df_stats.rename(columns={"time": "datetime"})
df_stats = df_stats.rename(columns={"crop": "filename"})

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



def mean_std_by_time(df, value_col):
    return (
        df.groupby(["t_bin", "label"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

def population_by_time(df, remove_extrapolated=False):
    if remove_extrapolated:
        df = df[df['crop_type'] != 'extrapolated']
    counts = df.groupby(["t_bin", "label"]).size()
    return counts.reset_index(name="counts")


#merge from df_stats the coluns var==precipitation and column==99, and the var==euclid_msg_grid and columns None
#using the filename column
df = df.merge(
    df_stats[
        (df_stats["var"] == "precipitation")
        & (df_stats["99"].notnull())
    ][["filename", "99"]],
    on="filename",
    how="left",
    suffixes=("", "_precip_99")
)

df = df.merge(
    df_stats[
        (df_stats["var"] == "euclid_msg_grid")
        & (df_stats["None"].notnull())
    ][["filename", "None"]],
    on="filename",
    how="left",
    suffixes=("", "_euclid_msg_grid_None")
)

#rename column None to euclid_msg_grid and 99 to precipitation_99
df = df.rename(columns={"None": "lightning_count"})
df = df.rename(columns={"99": "rain_rate_p99"})



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

DIVIDE_BY = 'ALL' #REGION or DAY_NIGHT or ALL

EVENT_GROUPS = ['ALL']#['PRECIP', 'HAIL', 'MIXED']

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
            4, n_classes,
            figsize=(10, 5), #reduce height
            sharex=True,#make ratio 0.5, 1, 1
            gridspec_kw={'height_ratios': [0.5, 1, 1, 1]}
        )
        #reduce horizontal space between subplots
        fig.subplots_adjust(hspace=0.1, wspace=0.08)

        
        if n_classes == 1:
            axes = [axes]


        for i, cls in enumerate(INTERESTING_CLASSES):
            print(f"Processing event {event} - class {cls}...")
            #get key from short names: {3: 'CS', 0: 'SC', 6: 'MC1', 5: 'MC2', 1: 'EC', 2: 'DC', 4: 'OA'}
            label = [key for key, lbl in short.items() if lbl == cls][0]
            print(f"Label: {label}")
            #get color from the key (label) {3: '#dbe9ff', 0: '#88a4cf', 6: '#78fba0', 5: '#239b4d', 1: '#fdae61', 2: '#d73027', 4: '#8b46a1'}
            color = colors[label]
            print(color)

            df_cls = df_event[df_event["label"] == label]

            # --------------------
            # Population
            # --------------------
            pop_cls = population_by_time(df_cls)
            #pop_cls_no_extrap = population_by_time(df_cls, remove_extrapolated=True)
            #pop_cls = pop[pop["label"] == label]

            axes[0,i].plot(
                pop_cls["t_bin"],
                pop_cls["counts"],
                color=color,
                lw=2
            )

            # axes[0,i].plot(
            #     pop_cls_no_extrap["t_bin"],
            #     pop_cls_no_extrap["counts"],
            #     color=color,
            #     lw=2,
            #     ls="--"
            # )

            y_max = 5000
            axes[0,i].set_ylim(10, y_max)
            axes[0,i].set_yscale("log")
            #remove y label and ticks for y axis except for first column
           
            axes[0,i].set_title(f"Class {cls}", fontsize=12, fontweight="bold")
            axes[0,i].grid(which='both', axis='both', linestyle='--', alpha=0.3)
            if i != 0:
                axes[0, i].set_ylabel("")
                axes[0, i].set_yticklabels([])
            else:
                axes[0,i].set_ylabel("Counts", fontsize=10)
                #set 5 y ticks
                
                axes[0,i].yaxis.set_major_locator(mpl.ticker.FixedLocator([10, 100, 1000]))
                #set y ticks label fontsize
                axes[0,i].tick_params(axis='y', labelsize=10)

            # --------------------
            # Weighted Percentage of Interest Classes
            # --------------------
            for lbl in INTERESTING_CLASSES:
                label_2 = [key for key, l in short.items() if l == lbl][0]
                perc = mean_std_by_time(df_cls, f"wperc_label_{label_2}")
                color_2 = colors[label_2]
                #plot only if the count in the bin is > 50
                perc = perc[perc["count"] >= 30]

                axes[1, i].plot(
                    perc["t_bin"],
                    perc["mean"],
                    color=color_2,
                    lw=2,
                    alpha=0.8
                )
                axes[1, i].fill_between(
                    perc["t_bin"],
                    perc["mean"] - perc["std"],
                    perc["mean"] + perc["std"],
                    color=color_2,
                    alpha=0.15
                )
            axes[1, i].set_ylim(0,105)
            axes[1,i].grid(which='both', axis='both', linestyle='--', alpha=0.3)
            axes[1, 0].yaxis.set_major_locator(mpl.ticker.FixedLocator([0, 25, 50, 75, 100]))
            #remove label and ticks for y axis except for first column
            if i != 0:
                axes[1, i].set_ylabel("")
                axes[1, i].set_yticklabels([])
            else:
                axes[1, 0].set_ylabel("Weighted \n Percentage", fontsize=10)
                #set y ticks label fontsize
                axes[1, 0].tick_params(axis='y', labelsize=10)
                #draw 4 ticks in y with FixedLocator
                
            # --------------------
            # Add precipitation 99th percentile and lightning count in the same plot (twin y axis)
            # --------------------
            rr99 = mean_std_by_time(df_cls, "rain_rate_p99")
            rr99 = rr99[rr99["count"] >= 30]
            axes[2, i].plot(
                rr99["t_bin"],
                rr99["mean"],
                color="blue",
                lw=2,
                alpha=0.8
            )
            axes[2, i].fill_between(
                rr99["t_bin"],
                rr99["mean"] - rr99["std"],
                rr99["mean"] + rr99["std"],
                color="blue",
                alpha=0.15
            )
            #remove label and ticks for y axis except for first column
            axes[2, i].set_ylim(0,25)
            axes[2, i].grid(which='both', axis='both', linestyle='--', alpha=0.3)
            axes[2, i].yaxis.set_major_locator(mpl.ticker.FixedLocator([0, 5, 10, 15, 20]))
            if i == 0:
                axes[2, i].set_ylabel("Rain Rate \n p99 (mm/hr)", fontsize=10) 
                #set y ticks label fontsize
                axes[2, i].tick_params(axis='y', labelsize=10)#, colors="blue")
            else:
                axes[2, i].set_ylabel("")
                axes[2, i].set_yticklabels([])
           

            #create twin axis for lightning count
            
            lc = mean_std_by_time(df_cls, "lightning_count")
            lc = lc[lc["count"] >= 30]
            axes[3, i].plot(
                lc["t_bin"],
                lc["mean"],
                color="green",
                lw=2,
                alpha=0.8
            )
            axes[3, i].fill_between(
                lc["t_bin"],
                lc["mean"] - lc["std"],
                lc["mean"] + lc["std"],
                color="green",
                alpha=0.15
            )
            axes[3, i].set_ylim(50, 2000)
            axes[3, i].set_yscale("log")
            axes[3, i].grid(which='both', axis='both', linestyle='--', alpha=0.3)
            axes[3, i].yaxis.set_major_locator(mpl.ticker.FixedLocator([100, 500, 1000]))
            #remove label and ticks for y axis except for first column
            if i != 0:
                axes[3, i].set_ylabel("")
                axes[3, i].set_yticklabels([])
            else:
                axes[3, i].set_ylabel("Lightning \n Count", fontsize=10)#, color="orange")
                #set y ticks label fontsize
                axes[3, i].tick_params(axis='y', labelsize=10)#, colors="orange")
                axes[3, i].set_yticklabels([100, 500, 1000])

        # --------------------
        # Final cosmetics
        # --------------------
        axes[3,1].set_xlabel("Aligned time (hours)", fontsize=12)
        axes[3,i].set_xlim(-6 - BIN_HOURS, 6 + BIN_HOURS)
        #set ticks every 2 hours
        axes[3,i].set_xticks(np.arange(-6, 7, 2))
        #set x ticks label fontsize
        axes[3,i].tick_params(axis='x', labelsize=10)
        #set overall title
        fig.suptitle(
            f"{event} - Trajectory-aligned analysis",
            fontsize=12,
            fontweight="bold", y=0.99
        )

        for ax in axes.flat:
            ax.axvline(0, color="k", lw=1, ls="--", alpha=0.5)
            #put both horizontal and vertical grid lines to all axes
            #ax.grid(which='both', axis='both', linestyle='--', alpha=0.3)

        output_path = os.path.join(
            plot_dir,
            f"trajectory_analysis_with_stats_{event}.png"
        )
        fig.savefig(
            output_path,
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        print(f"Saved figure: {output_path}")