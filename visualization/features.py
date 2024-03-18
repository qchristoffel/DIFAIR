import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def tsne(
    known_features,
    known_labels,
    classes_dict,
    nb_classes,
    unknown_features=None,
    unknown_labels=None,
    class_anchors=None,
    mean_centers=None,
    save_path=None,
    title=None,
):

    if unknown_features is not None:
        features = np.concatenate([known_features, unknown_features], axis=0)
        if unknown_labels is None:
            unknown_labels = np.full(len(unknown_features), -1)
            classes_dict[-1] = "unknown"
            to_color = nb_classes + 1
        else:
            to_color = nb_classes + len(np.unique(unknown_labels))
        labels = np.concatenate([known_labels, unknown_labels], axis=0)
    else:
        features = known_features
        labels = known_labels
        to_color = nb_classes
    print(np.unique(labels, return_counts=True))

    if class_anchors is not None:
        # print("True centers:", class_anchors)
        features = np.concatenate([class_anchors, features], axis=0)
        shift_true = nb_classes
    else:
        shift_true = 0

    if mean_centers is not None:
        # print("Mean centers:", mean_centers)
        features = np.concatenate([mean_centers, features], axis=0)
        shift_mean = nb_classes
    else:
        shift_mean = 0

    print(f"TSNE visualization of known/unknown features")
    tsne = TSNE(
        n_components=2, verbose=1, perplexity=50, n_iter=2000, init="pca", n_jobs=-1
    )
    x2d = tsne.fit_transform(features)
    print("TSNE shape:", x2d.shape)
    print("-" * 20)

    colors = plt.cm.tab20(np.linspace(0, 1, min(20, nb_classes)))

    plt.figure(figsize=(10, 7))
    for i, label in enumerate(sorted(np.unique(labels), reverse=True)):
        plt.scatter(
            x2d[shift_true + shift_mean :][labels == label, 0],
            x2d[shift_true + shift_mean :][labels == label, 1],
            color=colors[i % 20],
            s=30,
            label=classes_dict[label].rsplit(",")[0],
            marker="o" if label >= 0 else "x",
            alpha=1,
        )

    if class_anchors is not None:
        for i in range(nb_classes):
            plt.scatter(
                x2d[shift_mean + i, 0],
                x2d[shift_mean + i, 1],
                color="k",
                s=100,
                marker="*",
                label="Class anchor" if i == 0 else "",
            )
            plt.text(
                x2d[shift_mean + i, 0],
                x2d[shift_mean + i, 1] - 2,
                classes_dict[i].rsplit(",")[0],
                fontsize=14,
                fontweight="bold",
                horizontalalignment="center",
                verticalalignment="top",
            )

    if mean_centers is not None:
        for i in range(nb_classes):
            plt.scatter(
                x2d[i, 0],
                x2d[i, 1],
                color="k",
                s=100,
                marker="+",
                label="Mean center" if i == 0 else "",
                linewidths=2,
                edgecolors="k",
            )
            # plt.text(x2d[i, 0], x2d[i, 1], classes_dict[i], fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    if title is not None:
        plt.title(title, fontsize=16)
    else:
        plt.title("TSNE visualization", fontsize=16)

    plt.xlabel("Component 1", fontsize=14)
    plt.ylabel("Component 2", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)


def compute_mean_features(features, labels, nb_classes):
    # no sorting between correctly labelled and mislabelled samples
    mean_features = np.zeros((nb_classes, features.shape[1]))
    for i in range(nb_classes):
        mean_features[i] = np.mean(features[labels == i], axis=0)
    return mean_features


def plot_mean_features(
    mean_features,
    nb_features,
    class_names,
    ncols=3,
    identified_features=False,
    save_path=None,
):
    if mean_features.shape[1] != nb_features:
        nb_features = 5  # TODO: temporary fix to plot cross entropy representations

    nrows = np.ceil(len(class_names) / ncols).astype(int)
    width = int(0.5 * nb_features * ncols)
    height = int(0.5 * len(class_names) * nrows)
    fig, axs = plt.subplots(nrows, ncols, sharey=True, figsize=(width, height))
    axs = axs.flatten()

    vmin = np.min(mean_features)
    vmax = np.max(mean_features)

    for i in range(len(class_names)):
        reshaped_features = mean_features[i].reshape(-1, nb_features).copy()
        # reshaped_features[reshaped_features < THRESHOLD] = 0
        im = axs[i].imshow(
            reshaped_features,
            # cmap='rainbow',
            # cmap='viridis',
            cmap="OrRd",
            vmin=vmin,
            vmax=vmax,
        )
        axs[i].set_title(class_names[i])
        axs[i].set_xticks(range(nb_features), labels=range(nb_features))
        if identified_features:
            axs[i].set_yticks(range(len(class_names)), labels=class_names)
        # for j in range(16):
        #     for k in range(reshaped_features.shape[1]):
        #         axs[i].text(k, j, f"{reshaped_features[j, k]:.2f}",
        #                     ha="center", va="center", color="w")
    for i in range(len(class_names), len(axs)):
        axs[i].axis("off")

    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)

    print(save_path)
    if save_path:
        plt.savefig(save_path)

    return fig


def features_similarity(mean_features, class_names):
    cos_sim = cosine_similarity(mean_features).round(3)
    cos_sim_df = pd.DataFrame(cos_sim, index=class_names, columns=class_names)

    dist = euclidean_distances(mean_features).round(3)
    dist[np.diag_indices_from(dist)] = np.nan
    # norm_dist = (dist-np.min(dist)) / (np.max(dist) - np.min(dist))
    dist_df = pd.DataFrame(dist, index=class_names, columns=class_names)
    dist_df.fillna(0, inplace=True)

    print("Cosine similarity between representations:")
    print(cos_sim_df)

    print("Euclidean distance between representations:")
    print(dist_df)

    plt.figure()
    plt.imshow(cos_sim_df, cmap="OrRd")
    plt.colorbar()
    plt.title("Cosine similarity between mean representations", fontsize=16)
    plt.xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), labels=class_names)
    plt.tight_layout()

    plt.figure()
    boundaries = np.linspace(0, np.max(dist_df.values), 8)
    norm = colors.BoundaryNorm(boundaries, ncolors=256)
    im = plt.imshow(dist_df, cmap="OrRd", norm=norm)
    plt.colorbar(im)
    plt.title("Euclidean distance between mean representations", fontsize=16)
    plt.xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), labels=class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if dist_df.iloc[i, j] > np.median(dist_df.values):
                text = plt.text(
                    j,
                    i,
                    f"{dist_df.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="w",
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=2, foreground="black"),
                        path_effects.Normal(),
                    ]
                )
            else:
                text = plt.text(
                    j,
                    i,
                    f"{dist_df.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="w",
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=2, foreground="#323336"),
                        path_effects.Normal(),
                    ]
                )
    plt.tight_layout()
