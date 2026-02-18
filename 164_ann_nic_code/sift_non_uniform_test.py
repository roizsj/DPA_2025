import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def file_sha1(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    """Hash the raw bytes of the fvecs file for cache safety."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_cluster_offsets(labels: np.ndarray, k: int):
    """
    Given labels (N,), build:
      - perm: indices that sort vectors by (label, stable by index)
      - offsets: (k+1,) offsets so that vectors in cluster c are in
                 perm[offsets[c] : offsets[c+1]]
      - counts: (k,) bincount
    """
    labels = labels.astype(np.int32, copy=False)
    counts = np.bincount(labels, minlength=k).astype(np.int64)

    # stable sort by label then by original index (mergesort is stable)
    perm = np.argsort(labels, kind="mergesort").astype(np.int64)

    offsets = np.zeros(k + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return perm, offsets, counts


def save_ivf_artifacts(
    out_dir: Path,
    args,
    X_shape,
    Q_shape,
    base_path: str,
    query_path: str,
    centroids: np.ndarray,
    labels: np.ndarray,
    cluster_vec_counts: np.ndarray,
    cluster_probe_counts: np.ndarray,
    groups: list[dict],
    perm: np.ndarray,
    offsets: np.ndarray,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) centroids ----
    np.save(out_dir / "centroids.npy", centroids.astype(np.float32, copy=False))

    # ---- 2) labels / perm / offsets ----
    np.save(out_dir / "labels.npy", labels.astype(np.int32, copy=False))
    np.save(out_dir / "perm_by_cluster.npy", perm.astype(np.int64, copy=False))
    np.save(out_dir / "cluster_offsets.npy", offsets.astype(np.int64, copy=False))

    # ---- 3) counts / probe counts ----
    np.save(out_dir / "cluster_vec_counts.npy", cluster_vec_counts.astype(np.int64, copy=False))
    np.save(out_dir / "cluster_probe_counts.npy", cluster_probe_counts.astype(np.int64, copy=False))

    # ---- 4) cluster -> disk mapping (derived from groups) ----
    # groups is length 4, each has "clusters": np.array([...])
    cluster_to_disk = np.full(args.k, -1, dtype=np.int16)
    for disk_id, g in enumerate(groups):
        cids = g["clusters"]
        cluster_to_disk[cids] = disk_id
    if (cluster_to_disk < 0).any():
        # should not happen if groups cover all clusters; but safe-guard
        missing = np.where(cluster_to_disk < 0)[0][:20]
        raise ValueError(f"Some clusters not assigned to any disk. Example missing: {missing.tolist()}")

    np.save(out_dir / "cluster_to_disk.npy", cluster_to_disk)

    # ---- 5) metadata for safety / reproducibility ----
    meta = {
        "base_fvecs": str(base_path),
        "query_fvecs": str(query_path),
        "base_sha1": file_sha1(str(base_path)),
        "query_sha1": file_sha1(str(query_path)),
        "X_shape": list(map(int, X_shape)),
        "Q_shape": list(map(int, Q_shape)),
        "k": int(args.k),
        "nprobe": int(args.nprobe),
        "batch_size": int(args.batch_size),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "mmap": bool(args.mmap),
        "artifacts": {
            "centroids": "centroids.npy",
            "labels": "labels.npy",
            "perm_by_cluster": "perm_by_cluster.npy",
            "cluster_offsets": "cluster_offsets.npy",
            "cluster_vec_counts": "cluster_vec_counts.npy",
            "cluster_probe_counts": "cluster_probe_counts.npy",
            "cluster_to_disk": "cluster_to_disk.npy",
        },
    }
    with (out_dir / "ivf_artifacts_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved IVF artifacts to: {out_dir}")

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

    # interpret underlying buffer as float32 and reshape
    x = data.view(np.float32).reshape(n, stride)[:, 1:]
    return np.asarray(x, dtype=np.float32)


def train_kmeans(X: np.ndarray, k: int, batch_size: int, max_iter: int, seed: int) -> MiniBatchKMeans:
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=seed,
        n_init="auto",
        verbose=0,
    )
    km.fit(X)
    return km


def assign_counts(labels: np.ndarray, k: int) -> np.ndarray:
    # cluster sizes (vector counts per cluster)
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    return counts


def ivf_probe_counts(
    Q: np.ndarray,
    centroids: np.ndarray,
    nprobe: int,
    block_q: int = 4096,
) -> np.ndarray:
    """
    Count how many times each centroid is probed.
    For each query, select nprobe nearest centroids by squared L2 distance.
    Returns array of shape (k,) with probe counts.
    """
    k, d = centroids.shape
    if Q.shape[1] != d:
        raise ValueError(f"Dim mismatch: Q dim={Q.shape[1]} vs centroids dim={d}")

    # precompute centroid norms for fast distance: ||q-c||^2 = ||q||^2 + ||c||^2 - 2 q·c
    c_norm = (centroids * centroids).sum(axis=1, dtype=np.float32)  # (k,)
    probe_counts = np.zeros(k, dtype=np.int64)

    n = Q.shape[0]
    for s in range(0, n, block_q):
        qb = Q[s : s + block_q]  # (b, d)
        q_norm = (qb * qb).sum(axis=1, dtype=np.float32)  # (b,)

        # dot: (b,k)
        dots = qb @ centroids.T  # float32 matmul
        # dist^2 = q_norm[:,None] + c_norm[None,:] - 2*dots
        dist = q_norm[:, None] + c_norm[None, :] - 2.0 * dots

        # pick nprobe smallest per row using argpartition (O(k))
        # idx shape: (b, nprobe)
        idx = np.argpartition(dist, kth=nprobe - 1, axis=1)[:, :nprobe]

        # accumulate counts
        # flatten and bincount
        flat = idx.reshape(-1)
        probe_counts += np.bincount(flat, minlength=k).astype(np.int64)

    return probe_counts

def split_into_4_by_freq_balanced_vectors(cluster_order: np.ndarray, cluster_sizes: np.ndarray) -> list[dict]:
    total_vec = int(cluster_sizes.sum())
    target = total_vec / 4.0

    groups = []
    cur = []
    cur_vec = 0
    cut_targets = [target, 2 * target, 3 * target]
    cum = 0
    t_i = 0

    for cid in cluster_order:
        sz = int(cluster_sizes[cid])
        cur.append(int(cid))
        cur_vec += sz
        cum += sz

        if t_i < 3 and cum >= cut_targets[t_i]:
            groups.append({"clusters": np.array(cur, dtype=np.int32), "vector_count": int(cur_vec)})
            cur = []
            cur_vec = 0
            t_i += 1

    groups.append({"clusters": np.array(cur, dtype=np.int32), "vector_count": int(cur_vec)})

    while len(groups) < 4:
        groups.append({"clusters": np.array([], dtype=np.int32), "vector_count": 0})

    if len(groups) > 4:
        merged_clusters = np.concatenate([g["clusters"] for g in groups[3:]]) if groups[3:] else np.array([], dtype=np.int32)
        merged_vec = int(sum(g["vector_count"] for g in groups[3:]))
        groups = groups[:3] + [{"clusters": merged_clusters, "vector_count": merged_vec}]

    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_fvecs", required=True, help="Path to SIFT-1M base .fvecs (e.g., sift_base.fvecs)")
    ap.add_argument("--query_fvecs", required=True, help="Path to queries .fvecs (e.g., sift_query.fvecs)")
    ap.add_argument("--k", type=int, default=20000, help="Number of IVF clusters (nlist)")
    ap.add_argument("--nprobe", type=int, default=32, help="How many clusters probed per query")
    ap.add_argument("--batch_size", type=int, default=200000, help="MiniBatchKMeans batch size")
    ap.add_argument("--max_iter", type=int, default=200, help="MiniBatchKMeans max_iter")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--mmap", action="store_true", help="Use memmap to read fvecs")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    X = read_fvecs(args.base_fvecs, mmap=args.mmap)
    Q = read_fvecs(args.query_fvecs, mmap=args.mmap)

    print(f"Loaded X: {X.shape}, Q: {Q.shape}")

    # 2) Train kmeans (IVF coarse quantizer)
    print(f"Training MiniBatchKMeans k={args.k} ...")
    km = train_kmeans(X, k=args.k, batch_size=args.batch_size, max_iter=args.max_iter, seed=args.seed)
    centroids = km.cluster_centers_.astype(np.float32)

    # 3) Assign base vectors to clusters -> cluster vector counts
    print("Assigning base vectors to clusters ...")
    labels = km.predict(X)
    cluster_vec_counts = assign_counts(labels, args.k)

    # 4) IVF query probing -> per-cluster probe counts
    print(f"Counting probed clusters with nprobe={args.nprobe} ...")
    cluster_probe_counts = ivf_probe_counts(Q, centroids, nprobe=args.nprobe)

    # 5) Save per-cluster stats
    per_cluster_path = out_dir / "cluster_access_counts.csv"
    # CSV columns: cluster_id, vectors_in_cluster, probe_count
    with per_cluster_path.open("w", encoding="utf-8") as f:
        f.write("cluster_id,vectors_in_cluster,probe_count\n")
        for cid in range(args.k):
            f.write(f"{cid},{int(cluster_vec_counts[cid])},{int(cluster_probe_counts[cid])}\n")
    print(f"Wrote: {per_cluster_path}")

    # 6) Sort clusters by access frequency (probe_count desc), then split into 4 groups
    order = np.argsort(-cluster_probe_counts, kind="stable")  # stable keeps deterministic ties
    groups = split_into_4_by_freq_balanced_vectors(order, cluster_vec_counts)
    
    # Build perm + offsets for posting lists (required for fast reuse / writing to disks)
    perm, offsets, _ = build_cluster_offsets(labels, args.k)

    # Save all MUST artifacts for reuse (no retrain next time)
    save_ivf_artifacts(
        out_dir=out_dir,
        args=args,
        X_shape=X.shape,
        Q_shape=Q.shape,
        base_path=args.base_fvecs,
        query_path=args.query_fvecs,
        centroids=centroids,
        labels=labels,
        cluster_vec_counts=cluster_vec_counts,
        cluster_probe_counts=cluster_probe_counts,
        groups=groups,
        perm=perm,
        offsets=offsets,
    )

    total_probes = int(cluster_probe_counts.sum())
    total_vec = int(cluster_vec_counts.sum())

    group_summary = []
    for gi, g in enumerate(groups, start=1):
        cids = g["clusters"]
        probes = int(cluster_probe_counts[cids].sum()) if cids.size > 0 else 0
        vecs = int(g["vector_count"])
        group_summary.append(
            {
                "group": gi,
                "num_clusters": int(cids.size),
                "vector_count": vecs,
                "vector_share": (vecs / total_vec) if total_vec > 0 else 0.0,
                "probe_count": probes,
                "probe_share": (probes / total_probes) if total_probes > 0 else 0.0,
                "clusters": cids.tolist(),  # if too big, you can omit this
            }
        )

    summary_path = out_dir / "access_groups_4.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "k": args.k,
                "nprobe": args.nprobe,
                "total_vectors": total_vec,
                "total_probe_events": total_probes,
                "groups": group_summary,
            },
            f,
            indent=2,
        )
    print(f"Wrote: {summary_path}")

    # Also write a small CSV summary (without listing cluster IDs)
    summary_csv = out_dir / "access_groups_4_summary.csv"
    with summary_csv.open("w", encoding="utf-8") as f:
        f.write("group,num_clusters,vector_count,vector_share,probe_count,probe_share\n")
        for g in group_summary:
            f.write(
                f"{g['group']},{g['num_clusters']},{g['vector_count']},"
                f"{g['vector_share']:.6f},{g['probe_count']},{g['probe_share']:.6f}\n"
            )
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
