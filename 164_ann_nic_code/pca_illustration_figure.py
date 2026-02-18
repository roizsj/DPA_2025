import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15
})


def pca_2d(X):
    mu = X.mean(axis=0)
    Xc = X - mu
    C = np.cov(Xc, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    Z = Xc @ eigvecs
    return mu, eigvals, eigvecs, Z


def add_side_box(ax, lines, fontsize=18):
    text = "\n".join(lines)
    ax.text(
        1.03, 0.92,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        linespacing=1.4,
        bbox=dict(
            boxstyle="round,pad=0.5",
            fc="white",
            ec="0.4",
            alpha=0.98
        )
    )


def plot_pca_original(X, save_path):
    mu, eigvals, eigvecs, _ = pca_2d(X)
    pc1, pc2 = eigvecs[:, 0], eigvecs[:, 1]

    var_x = np.var(X[:, 0], ddof=1)
    var_y = np.var(X[:, 1], ddof=1)
    var_pc1, var_pc2 = eigvals

    fig, ax = plt.subplots(figsize=(11, 5))

    fig.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.15)

    ax.scatter(X[:, 0], X[:, 1], s=30)
    ax.set_title("Original Data (x–y space)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, lw=1, alpha=0.25)
    ax.axvline(0, lw=1, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    s1 = 2.6 * np.sqrt(var_pc1)
    s2 = 2.6 * np.sqrt(var_pc2)

    ax.arrow(mu[0], mu[1], pc1[0]*s1, pc1[1]*s1,
             head_width=0.18, head_length=0.28,
             lw=2, color="tab:red", length_includes_head=True)
    ax.arrow(mu[0], mu[1], pc2[0]*s2, pc2[1]*s2,
             head_width=0.18, head_length=0.28,
             lw=2, color="tab:blue", length_includes_head=True)

    ax.text(mu[0]+pc1[0]*s1*1.05, mu[1]+pc1[1]*s1*1.05,
            "PC1", fontsize=18, fontweight="bold", color="tab:red")
    ax.text(mu[0]+pc2[0]*s2*1.05, mu[1]+pc2[1]*s2*1.05,
            "PC2", fontsize=18, fontweight="bold", color="tab:blue")

    add_side_box(ax, [
        f"Var(x) = {var_x:.2f}",
        f"Var(y) = {var_y:.2f}",
    ])

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_transformed(X, save_path):
    mu, eigvals, eigvecs, Z = pca_2d(X)

    var_pc1, var_pc2 = eigvals
    ratio1 = var_pc1 / (var_pc1 + var_pc2)
    ratio2 = var_pc2 / (var_pc1 + var_pc2)

    fig, ax = plt.subplots(figsize=(11, 5))

    fig.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.15)

    ax.scatter(Z[:, 0], Z[:, 1], s=30)
    ax.set_title("After PCA (PC1–PC2 space)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axhline(0, lw=1, alpha=0.25)
    ax.axvline(0, lw=1, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    add_side_box(ax, [
        f"Var(PC1) = {var_pc1:.2f}",
        f"Var(PC2) = {var_pc2:.2f}",
    ])

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
if __name__ == "__main__":
    np.random.seed(7)

    cov = np.array([[3.0, 2.0],
                    [2.0, 2.0]])
    X = np.random.multivariate_normal([0, 0], cov, size=80)

    plot_pca_original(X, "pca_original_ppt.png")
    plot_pca_transformed(X, "pca_after_pca_ppt.png")



