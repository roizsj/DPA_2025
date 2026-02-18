import os
import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

def read_fvecs(path: str, mmap: bool = True) -> np.ndarray:
    """
    Read .fvecs file (Faiss format):
    Each vector stored as: int32 dim, then dim float32 values.
    Returns float32 array of shape (n, dim).
    """
    if mmap:
        data = np.memmap(path, dtype=np.int32, mode="r")
    else:
        data = np.fromfile(path, dtype=np.int32)

    if data.size == 0:
        raise ValueError("Empty fvecs file.")

    dim = int(data[0])
    stride = 1 + dim  # int32 dim + dim float32 (but viewed as int32)
    if data.size % stride != 0:
        raise ValueError(
            f"File size not aligned with dim={dim}. "
            f"data.size={data.size}, stride={stride}"
        )
    n = data.size // stride

    # View the underlying buffer as float32 for the payload part
    data_f = data.view(np.float32).reshape(n, stride)
    X = data_f[:, 1:].astype(np.float32, copy=False)  # skip the dim column
    return X

def batch_predict(km: MiniBatchKMeans, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
    """Predict cluster labels in batches to reduce peak memory."""
    n = X.shape[0]
    labels = np.empty(n, dtype=np.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        labels[start:end] = km.predict(X[start:end])
    return labels

def main(
    fvecs_path: str,
    n_clusters: int = 50_000,
    batch_size: int = 50_000,
    max_iter: int = 100,
    n_init: int = 1,
    random_state: int = 42,
    reassignment_ratio: float = 0.01,
    predict_batch_size: int = 200_000,
    save_csv: bool = True,
    out_prefix: str = "sift1m_kmeans_50k"
):
    # 1) load
    X = read_fvecs(fvecs_path, mmap=True)
    n, d = X.shape
    print(f"Loaded X: n={n:,}, d={d}")

    # 2) train MiniBatchKMeans
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        reassignment_ratio=reassignment_ratio,
        verbose=1
    )
    km.fit(X)  # if memory is tight, see notes below

    # 3) assign labels (batch predict)
    labels = batch_predict(km, X, batch_size=predict_batch_size)

    # 4) counts per cluster (size n_clusters)
    counts = np.bincount(labels, minlength=n_clusters)
    assert counts.size == n_clusters

    # 5) histogram: x -> n_x
    #    x = cluster size, n_x = how many clusters have that size
    size_dist = Counter(counts.tolist())

    # Print summary
    print("\nCluster size stats:")
    print(f"  min={counts.min()}, max={counts.max()}, mean={counts.mean():.2f}, std={counts.std():.2f}")
    # (optional) how many empty clusters
    print(f"  empty clusters (x=0): {size_dist.get(0, 0)}")

    # Print full distribution sorted by x
    print("\nDistribution (x -> n_x):")
    for x in sorted(size_dist.keys()):
        print(f"{x}\t{size_dist[x]}")

    if save_csv:
        # Save cluster sizes
        cluster_csv = f"{out_prefix}_{n_clusters}_cluster_sizes.csv"
        np.savetxt(
            cluster_csv,
            np.c_[np.arange(n_clusters), counts],
            delimiter=",",
            header="cluster_id,count",
            comments="",
            fmt=["%d", "%d"]
        )
        print(f"\nSaved: {cluster_csv}")

        # Save x->n_x distribution
        dist_csv = f"{out_prefix}_size_distribution.csv"
        with open(dist_csv, "w", encoding="utf-8") as f:
            f.write("x,n_x\n")
            for x in sorted(size_dist.keys()):
                f.write(f"{x},{size_dist[x]}\n")
        print(f"Saved: {dist_csv}")

if __name__ == "__main__":
    # 修改成你的文件路径，比如: r"/path/to/sift_base.fvecs"
    fvecs_path = r"/home/zhangshujie/ann_nic/sift/sift_base.fvecs"
    main(fvecs_path)