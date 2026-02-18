import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def read_fvecs(path: str, mmap: bool = True) -> np.ndarray:
    """
    Read FAISS-style .fvecs:
      each vector: int32 dim, followed by dim float32.
    Returns float32 array of shape (n, dim).
    """
    path = str(path)
    if mmap:
        data = np.memmap(path, dtype=np.int32, mode="r")
    else:
        data = np.fromfile(path, dtype=np.int32)

    if data.size == 0:
        raise ValueError(f"Empty fvecs: {path}")

    dim = int(data[0])
    stride = 1 + dim
    if data.size % stride != 0:
        raise ValueError(
            f"File size mismatch for {path}: data.size={data.size}, dim={dim}, stride={stride}"
        )
    n = data.size // stride
    x = data.view(np.float32).reshape(n, stride)[:, 1:]
    return np.asarray(x, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_fvecs", required=True, help="Path to sift_base.fvecs (or any .fvecs)")
    ap.add_argument("--sample", type=int, default=10000, help="Number of vectors to sample")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--mmap", action="store_true", help="Use memmap to read fvecs")
    ap.add_argument("--center", action="store_true", help="Explicitly center data (PCA already centers by default)")
    ap.add_argument("--out_png", default="pca_cumvar.png", help="Output plot path")
    args = ap.parse_args()

    X = read_fvecs(args.base_fvecs, mmap=args.mmap)
    n, d = X.shape
    if args.sample > n:
        raise ValueError(f"sample={args.sample} > N={n}")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=args.sample, replace=False)
    Xs = X[idx].astype(np.float32, copy=False)

    # PCA (sklearn PCA 会默认做中心化；对解释方差曲线这是标准做法)
    # full SVD 对 d<=1000 通常没问题；如果你未来 d 很大可换成 svd_solver="randomized"
    pca = PCA(n_components=d, svd_solver="full", random_state=args.seed)
    pca.fit(Xs)

    # cumulative explained variance ratio
    cum = np.cumsum(pca.explained_variance_ratio_)
    x = np.arange(1, d + 1)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, cum, linewidth=2)
    plt.xlabel("PCA components (x)")
    plt.ylabel("Cumulative explained variance ratio (1..x)")
    plt.ylim(0, 1.01)
    plt.xlim(1, d)
    plt.title("Sift1M")
    plt.grid(True, linewidth=0.5)

    # Optional: mark 80%, 90%, 95%, 99%
    for t in [0.80, 0.90, 0.95, 0.99]:
        k = int(np.searchsorted(cum, t) + 1)
        plt.text(min(k + 1, d), min(t + 0.01, 1.0), f"{int(t*100)}% @ {k}", fontsize=10)

    out = Path(args.out_png)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"[OK] saved plot to: {out.resolve()}")


if __name__ == "__main__":
    main()
