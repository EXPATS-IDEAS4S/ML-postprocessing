import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.sankey import Sankey
import networkx as nx



def plot_persistence_bar_multiplot(ax, results, labels, vmax=None, colors=None, class_names=None):
    lbls = results["labels"]
    persistence = results["persistence_prob"]

    ax.bar(lbls, persistence, color="tab:blue", alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Persistence prob." if ax.is_first_col() else "")



def plot_persistence_time_multiplot(ax, results, labels, vmax=None, colors=None, class_names=None):
    durations = results["persistence_durations"]
    lbls = list(durations.keys())
    #preparate data (set to Nan if n_values <50)
    data = [durations[l] if len(durations[l]) >= 50 else [np.nan] for l in lbls]
    #unpack colors to a list (dict with label and color)
    box = ax.boxplot(data, patch_artist=True, showfliers=False)
    if colors is None:
        colors = ["tab:blue"] * len(lbls)
    for i, (patch, label) in enumerate(zip(box['boxes'], labels)):
        patch.set_facecolor(colors[i])
        #set edge color and mean line
        patch.set_edgecolor("black")
        box['medians'][lbls.index(label)].set_color("black")
    
    #ax.set_yscale("log")
    ax.set_xticks(range(1, len(class_names) + 1))
    #set x ticks only on last row\
    #rewrite class_names adding also the split.{'_'}0 of the labels to the class nema
    class_names_modified = []
    for suffix, name in zip(labels, class_names):
        if '_' in suffix:
            class_names_modified.append(name +  '_' + suffix.split('_')[1])
        else:
            class_names_modified.append(name)
    ax.set_xticklabels(class_names_modified if ax.is_last_row() else [], fontsize=12, rotation=45, ha="right")
    #ax.set_xlabel("Class label" if ax.is_last_row() else "", fontsize=12)
    ax.set_ylabel("Persistence \n time (hours)" if ax.is_first_col() else "", fontsize=12)
    #set y ticks to be fixed along plots, to have 4 values from 0 to vmax
    #ax.set_yticks(np.arange(0, vmax + 1, vmax / 5))
    #ax.set_yticklabels(np.arange(0, vmax + 1, vmax / 5).astype(int), fontsize=12)
    #use only 5 ticks usinf FIXED locator
    
    ax.set_ylim(0.1, vmax)
    ax.set_yscale("log")
    #ax.yaxis.set_major_locator(plt.FixedLocator([0,2,4,6,8]))
    ax.yaxis.set_major_locator(plt.FixedLocator([0.1,1,10]))
    #ax.yaxis.tick_params(labelsize=12)
    #set grid
    ax.yaxis.grid(True)
    


def plot_transition_heatmap_multiplot(
    ax,
    results,
    labels,
    vmax=1.0,
    colors=None,
    class_names=None,
):
 

    P = results["transition_matrix"]
    P_low = results.get("transition_ci_low", None)
    P_high = results.get("transition_ci_high", None)
    #persistence_prob = results.get("persistence_prob", None)
    #persistence_ci_low = results.get("persistence_ci_low", None)
    #persistence_ci_high = results.get("persistence_ci_high", None)
    persistence_durations = results.get("persistence_durations", None)
    

    M = P.copy()
    np.fill_diagonal(M, np.nan)
    print(colors)
    
    sns.heatmap(
        M,
        ax=ax,
        cmap=colors or "Blues",
        vmin=0,
        vmax=vmax,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    # --- off-diagonal: transition probability ± CI ---
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if i != j and not np.isnan(M[i, j]):

                # if P_low is not None and P_high is not None:
                #     ci = (P_high[i, j] - P_low[i, j]) / 2
                #     text = f"{M[i, j]:.2f}\n±{ci:.2f}"
                # else:
                text = f"{M[i, j]:.2f}"

                ax.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if M[i, j] < vmax / 2 else "black",
                )

    # --- diagonal: persistence probability ---
    if persistence_durations is not None:
        for i, cls in enumerate(class_names):
            # persistence_durations is a dict with label keys and duration lists as values
            label = labels[i]
            if label in persistence_durations and len(persistence_durations[label]) > 0:
                durations = np.array(persistence_durations[label])
                mean_duration = np.nanmean(durations)
                if np.isfinite(mean_duration):
                    text = f"{mean_duration:.1f}h"
                else:
                    text = "n/a"
            else:
                text = "n/a"

            ax.text(
                i + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="red",
            )

    # --- ticks & labels ---
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    ax.set_xticks(np.arange(len(class_names)) + 0.5)

    class_names_modified = []
    print(labels)

    for suffix, name in zip(str(labels), class_names):
        if '_' in suffix:
            class_names_modified.append(name +'_' + suffix.split('_')[1])
        else:
            class_names_modified.append(name)

    ax.set_yticklabels(
            class_names_modified, rotation=0,
        fontsize=10 if len(class_names) <= 10 else 8
    )
    ax.set_xticklabels(
        class_names_modified, rotation=45, ha="right",
        fontsize=10 if len(class_names) <= 10 else 8
    )

    if ax.is_first_col():
        ax.set_ylabel("Current class", fontsize=10)

    if ax.is_last_row():
        ax.set_xlabel("Next class", fontsize=10)
    
    #title
    ax.set_title("g)", fontsize=12, fontweight="bold")

    # --- colorbar ---
    if ax.is_last_col():
        cbar = ax.figure.colorbar(
            ax.collections[0],
            ax=ax,
            orientation="vertical",
            fraction=0.046,
            pad=0.04,
            #title of colorbar on the left side with label "Transition probability" and fontsize 10 
            #label="Transition probability",
            #fontsize=10,

        )
        #cbar.ax.yaxis.set_label_position("left")
        cbar.ax.tick_params(labelsize=10)
        #cbar.ax.yaxis.set_ticks_position('left')
        #titel on the left side of colorbar with label "Transition probability" and fontsize 10
        cbar.set_label("Transition probability", fontsize=10, labelpad=8)
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.tick_params(labelsize=10)



        

def plot_entropy_persistence_multiplot(ax, results, labels, vmax=None, colors=None, class_names=None):
    p = results["persistence_prob"]
    H = results["entropy"]
    lbls = results["labels"]

    #colors = [colors.get(str(label), "tab:blue") if colors is not None else "tab:blue" for label in labels]
    ax.scatter(p, H, s=50, c=colors, alpha=0.8, edgecolors="k")

    # for i, lbl in enumerate(lbls):
    #     ax.text(p[i], H[i], str(lbl), fontsize=10)

    ax.set_xlabel("Persistence prob." if ax.is_last_row() else "", fontsize=12)
    ax.set_ylabel("Entropy" if ax.is_first_col() else "")

    #set ticks consistently
    ax.set_xticks(np.arange(0, 1.2, 0.2), fontsize=12, rotation=45)
    ax.set_yticks(np.arange(0, 1.6, 0.2), fontsize=12)

    #set x and y ticks labels only on left and bottom plots
    if not ax.is_first_col():
        ax.set_yticklabels([])
    if not ax.is_last_row():
        ax.set_xticklabels([])



def plot_alluvial_multiplot(ax, results, labels=None, vmax=None, min_weight=0.05):
    """
    Alluvial (Sankey) plot for transition probabilities.
    Only shows transitions above min_weight.
    """

    P = results["transition_matrix"]
    lbls = results["labels"]
    K = len(lbls)

    sankey = Sankey(ax=ax, scale=1.0, offset=0.2, head_angle=120)

    for i in range(K):
        flows = []
        labels_flow = []
        orientations = []

        outflow = 0.0
        for j in range(K):
            if i != j and P[i, j] >= min_weight:
                flows.append(-P[i, j])
                labels_flow.append(f"{lbls[i]}→{lbls[j]}")
                orientations.append(0)
                outflow += P[i, j]

        if outflow > 0:
            flows.insert(0, outflow)
            labels_flow.insert(0, lbls[i])
            orientations.insert(0, 0)

            sankey.add(
                flows=flows,
                labels=labels_flow,
                orientations=orientations,
                trunklength=1.0,
                pathlengths=[0.25] * len(flows)
            )

    sankey.finish()
    ax.set_title("Alluvial transitions", fontsize=10)





def plot_transition_graph_multiplot(
    ax,
    results,
    selected_labels=None,
    min_weight=0.05,
    colors=None,
    class_names=None,
):
    """
    Directed graph visualization of transition probabilities
    for a selected subset of classes.
    """

    P = results["transition_matrix"]
    lbls = list(results["labels"])
    persistence = results["persistence_prob"]

    # -------------------------
    # Label selection
    # -------------------------
    if selected_labels is None:
        selected_labels = lbls

    selected_labels = [l for l in selected_labels if l in lbls]

    if len(selected_labels) == 0:
        ax.axis("off")
        return

    # map label -> index in matrix
    label_to_idx = {l: i for i, l in enumerate(lbls)}

    # -------------------------
    # Graph
    # -------------------------
    G = nx.DiGraph()

    for lbl in selected_labels:
        i = label_to_idx[lbl]
        size = 300 + 2000 * persistence[i] if np.isfinite(persistence[i]) else 300
        G.add_node(lbl, size=size)

    # edges only among selected classes
    for li in selected_labels:
        for lj in selected_labels:
            if li == lj:
                continue
            i, j = label_to_idx[li], label_to_idx[lj]
            if P[i, j] >= min_weight:
                G.add_edge(li, lj, weight=P[i, j])

    if len(G.nodes) == 0:
        ax.axis("off")
        return

    # -------------------------
    # Layout
    # -------------------------
    pos = nx.circular_layout(G)

    node_sizes = [G.nodes[n]["size"] for n in G.nodes]
    edge_weights = [5 * G[u][v]["weight"] for u, v in G.edges]

    # -------------------------
    # Colors and labels
    # -------------------------
    
    node_colors = colors
    node_labels = class_names
    node_labels = {
        lbl: class_names[i] if class_names is not None else str(lbl)
        for i, lbl in enumerate(selected_labels)
    }
    
    # -------------------------
    # Draw
    # -------------------------
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.8,
        alpha=0.9,
        ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_weights,
        arrows=True,
        arrowstyle="->",
        alpha=0.7,
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=10,
        #fontweight="bold",
        ax=ax
    )

    ax.set_title("Transition graph", fontsize=11)
    ax.axis("off")

