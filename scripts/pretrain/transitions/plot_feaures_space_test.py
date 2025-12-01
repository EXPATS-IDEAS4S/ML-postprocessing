import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# =========================
# CONFIG
# =========================
run_name = 'dcv2_vit_k10_ir108_100x100_2013-2020_3xrandomcrops_1xtimestamp_cma_nc'
path_csv = f"/data1/fig/{run_name}/test"
filename = f"features_{run_name}.csv"
output_path = f"/data1/fig/{run_name}/test/"

COLORS_PER_CLASS = {
    '0': 'darkgray', '1': 'darkslategrey', '2': 'peru', '3': 'orangered',
    '4': 'lightcoral', '5': 'deepskyblue', '6': 'purple', '7': 'lightblue',
    '8': 'green', '9': 'goldenrod', '10': 'magenta', '11': 'dodgerblue',
    '12': 'darkorange', '13': 'olive', '14': 'crimson'
}

VECTOR_TYPES = ["msg", "PRECIP", "HAIL"]  # expected vector_type fields


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(f"{path_csv}/{filename}", low_memory=False)
print(df.head())
print("Loaded columns:", df.columns)


# =========================
# PREPARE FIGURE
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)
axes = axes.flatten()


# =========================
# PLOT LOOP
# =========================
for ax, vtype in zip(axes, VECTOR_TYPES):

    # --- Filter by vector_type ---
    sub = df[df["vector_type"] == vtype]

    if sub.empty:
        ax.set_title(f"No data for {vtype}")
        continue

    print(f"Computing t-SNE for vector_type={vtype}   (n={len(sub)})")

    # --- Extract feature vectors only ---
    feature_cols = [c for c in sub.columns if c.startswith("dim_")]
    X = sub[feature_cols].values.astype(np.float32)

    # --- t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=50,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        random_state=42
    )
    X_emb = tsne.fit_transform(X)

    sub["tsne_1"] = X_emb[:, 0]
    sub["tsne_2"] = X_emb[:, 1]

    # --- Plot each class ---
    for label in sorted(sub["label"].unique()):
        df_class = sub[sub["label"] == label]
        color = COLORS_PER_CLASS.get(str(label), "black")

        ax.scatter(
            df_class["tsne_1"],
            df_class["tsne_2"],
            s=2,
            color=color,
            label=str(label),
            alpha=0.7
        )

    ax.set_title(f"t-SNE – {vtype}", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Optional: shrink legend size
    ax.legend(markerscale=4, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig(f"{output_path}/tsne_features_space_test.png",
            dpi=300, bbox_inches="tight", transparent=True)
plt.close()
print(f"✅ Saved t-SNE features space plot: {output_path}/tsne_features_space_test.png")
