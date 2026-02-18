import os
import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


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
    stride = 1 + dim
    if data.size % stride != 0:
        raise ValueError(
            f"File size not aligned with dim={dim}. "
            f"data.size={data.size}, stride={stride}"
        )
    n = data.size // stride

    data_f = data.view(np.float32).reshape(n, stride)
    X = data_f[:, 1:].astype(np.float32, copy=False)
    return X


def batch_predict(km: MiniBatchKMeans, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
    """Predict cluster labels in batches to reduce peak memory."""
    n = X.shape[0]
    labels = np.empty(n, dtype=np.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        labels[start:end] = km.predict(X[start:end])
    return labels


def build_inverted_lists(labels: np.ndarray, n_clusters: int) -> list:
    """
    Build inverted lists: lists[c] = np.ndarray of base indices assigned to cluster c.
    Uses a stable + memory-friendly method (sort by label).
    """
    order = np.argsort(labels, kind="mergesort")
    sorted_labels = labels[order]
    # boundaries for each cluster in the sorted array
    counts = np.bincount(sorted_labels, minlength=n_clusters)
    offsets = np.zeros(n_clusters + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    lists = []
    for c in range(n_clusters):
        s, e = offsets[c], offsets[c + 1]
        lists.append(order[s:e].astype(np.int64, copy=False))
    return lists


def topk_candidates_l2(
    xb: np.ndarray,
    q: np.ndarray,
    cand_idx: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Return top-k indices (in xb) among candidate indices by squared L2 distance.
    Avoids constructing huge matrices.
    """
    Xcand = xb[cand_idx]  # (#cand, d)
    dist = np.sum((Xcand - q) ** 2, axis=1)  # squared L2
    kk = min(k, dist.size)
    pos = np.argpartition(dist, kth=kk - 1)[:kk]
    pos = pos[np.argsort(dist[pos])]
    return cand_idx[pos]


def ivf_topk_and_count_visits(
    xb: np.ndarray,
    xq: np.ndarray,
    centroids: np.ndarray,
    lists: list,
    k: int = 10,
    nprobe: int = 8,
    query_batch_size: int = 2000,
    dedup_candidates: bool = True,
    count_used_in_topk: bool = True,
):
    """
    IVF search (coarse -> probe nprobe lists -> exact within candidates) and count per-cluster visits.

    Returns:
      I: (nq, k) approx neighbor indices in xb
      visit_counts: (nlist,) each time a cluster is probed by a query => +1
      scanned_vecs: (nlist,) each time a cluster is probed => +len(list[c])
      used_in_topk: (nlist,) (optional) how many returned top-k points belong to each cluster
    """
    xb = np.asarray(xb, dtype=np.float32)
    xq = np.asarray(xq, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)

    nq = xq.shape[0]
    nlist = centroids.shape[0]
    nprobe = min(nprobe, nlist)

    visit_counts = np.zeros(nlist, dtype=np.int64)
    scanned_vecs = np.zeros(nlist, dtype=np.int64)
    used_in_topk = np.zeros(nlist, dtype=np.int64) if count_used_in_topk else None

    I = np.full((nq, k), -1, dtype=np.int64)

    # process queries in batches to control memory
    for qs in range(0, nq, query_batch_size):
        qe = min(qs + query_batch_size, nq)
        Q = xq[qs:qe]  # (b, d)

        # compute distances to centroids: (b, nlist)
        # If nlist is huge (e.g., 100k), this is the bottleneck.
        # But still doable in batches.
        dqc = np.sum((Q[:, None, :] - centroids[None, :, :]) ** 2, axis=2)  # squared L2
        probes = np.argpartition(dqc, kth=nprobe - 1, axis=1)[:, :nprobe]   # (b, nprobe)

        for i in range(Q.shape[0]):
            q = Q[i]
            probe_ids = probes[i]

            # count visits
            visit_counts[probe_ids] += 1
            for c in probe_ids:
                scanned_vecs[c] += len(lists[c])

            # gather candidates
            cand = []
            for c in probe_ids:
                idxs = lists[c]
                if idxs.size:
                    cand.append(idxs)
            if not cand:
                continue

            cand_idx = np.concatenate(cand)
            if dedup_candidates:
                cand_idx = np.unique(cand_idx)

            top_idx = topk_candidates_l2(xb, q, cand_idx, k=k)
            I[qs + i, :top_idx.size] = top_idx

            if count_used_in_topk:
                # need base->cluster mapping? easiest: build it once outside and pass in.
                # We'll handle it outside this function for speed/clarity.
                pass

    return I, visit_counts, scanned_vecs, used_in_topk


def plot_cache_coverage_by_cluster_size(
    list_sizes: np.ndarray,
    visit_counts: np.ndarray,
    out_prefix: str,
    nlist: int,
    step: int = 1,
    use_logx: bool = False,
):
    """
    Test relationship between #cached clusters and %requests covered.
    Sort clusters by list_size desc, cache top-m clusters, compute coverage = sum(visits in cached clusters)/sum(all visits).

    "request" here = one probe of one cluster by a query (i.e., visit_counts counts).
    """
    assert list_sizes.shape == visit_counts.shape
    total_visits = int(visit_counts.sum())
    if total_visits <= 0:
        raise ValueError("total_visits is 0; no probes counted.")

    # sort clusters by size desc (large -> small)
    order = np.argsort(list_sizes)[::-1]
    v_sorted = visit_counts[order]

    # cumulative covered requests when caching top-m clusters
    cum_visits = np.cumsum(v_sorted, dtype=np.int64)

    # sample points by step to avoid huge plot if nlist large
    m_all = np.arange(1, nlist + 1, dtype=np.int32)
    if step > 1:
        m = m_all[::step]
    else:
        m = m_all

    covered = cum_visits[m - 1] / total_visits  # m-1 because cum_visits is 0-indexed
    covered_pct = covered * 100.0

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(m, covered_pct, marker=".", linewidth=1)
    plt.xlabel("Number of cached clusters (top-m by cluster size)")
    plt.ylabel("Covered requests (%)")
    plt.title("Cache coverage vs cached clusters (sorted by cluster size)")

    if use_logx:
        plt.xscale("log")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig_path = f"{out_prefix}_nlist{nlist}_cache_coverage.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # also save csv
    csv_path = f"{out_prefix}_nlist{nlist}_cache_coverage.csv"
    out = np.c_[m, covered_pct]
    np.savetxt(csv_path, out, delimiter=",", header="cached_clusters,covered_requests_pct", comments="", fmt=["%d", "%.6f"])

    print(f"[OK] Saved plot: {fig_path}")
    print(f"[OK] Saved data: {csv_path}")

    return m, covered_pct


def main_ivf_visit_stats(
    base_fvecs_path: str,
    query_fvecs_path: str,
    nlist: int = 4096,
    nprobe: int = 16,
    k: int = 10,
    kmeans_batch_size: int = 50_000,
    max_iter: int = 200,
    n_init: int = 1,
    random_state: int = 42,
    reassignment_ratio: float = 0.01,
    predict_batch_size: int = 200_000,
    query_batch_size: int = 2000,
    out_prefix: str = "sift1m_ivf",
    save_csv: bool = True,
):
    # 1) load base + query
    xb = read_fvecs(base_fvecs_path, mmap=True)
    xq = read_fvecs(query_fvecs_path, mmap=True)
    nb, d = xb.shape
    nq = xq.shape[0]
    print(f"Loaded xb: n={nb:,}, d={d} | xq: n={nq:,}, d={xq.shape[1]}")

    # 2) train coarse quantizer (MiniBatchKMeans)
    km = MiniBatchKMeans(
        n_clusters=nlist,
        batch_size=kmeans_batch_size,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        reassignment_ratio=reassignment_ratio,
        verbose=1
    )
    km.fit(xb)

    centroids = km.cluster_centers_.astype(np.float32)

    # 3) assign base vectors to clusters
    base_labels = batch_predict(km, xb, batch_size=predict_batch_size)
    list_sizes = np.bincount(base_labels, minlength=nlist)
    print(f"Built labels. List size: min={list_sizes.min()}, max={list_sizes.max()}, mean={list_sizes.mean():.2f}")
    print(f"Empty lists: {(list_sizes==0).sum()}/{nlist}")

    # 4) build inverted lists
    lists = build_inverted_lists(base_labels, n_clusters=nlist)

    # 5) IVF search + count visits
    I, visit_counts, scanned_vecs, _ = ivf_topk_and_count_visits(
        xb=xb,
        xq=xq,
        centroids=centroids,
        lists=lists,
        k=k,
        nprobe=nprobe,
        query_batch_size=query_batch_size,
        dedup_candidates=True,
        count_used_in_topk=False,
    )

    # sanity check: total visits should be nq * nprobe (unless nprobe>nlist)
    print("\nVisit stats:")
    print(f"  total_visits={visit_counts.sum()} (expected ~ {nq*nprobe})")
    print(f"  min={visit_counts.min()}, max={visit_counts.max()}, mean={visit_counts.mean():.2f}, median={np.median(visit_counts):.2f}")
    print(f"  zero-visited clusters={(visit_counts==0).sum()}/{nlist}")

    # 6) histogram: v -> n_v  (visit count distribution)
    visit_dist = Counter(visit_counts.tolist())

    print("\nDistribution (visits v -> number of clusters n_v):")
    for v in sorted(visit_dist.keys()):
        print(f"{v}\t{visit_dist[v]}")
    
    # 7) cache coverage curve: cache top clusters by size and see %requests covered
    plot_cache_coverage_by_cluster_size(
        list_sizes=list_sizes,
        visit_counts=visit_counts,
        out_prefix=out_prefix,
        nlist=nlist,
        step=max(1, nlist // 2000),   # 自动降采样：最多~2000个点；你也可以改成 step=1
        use_logx=False
    )

    if save_csv:
        # per-cluster stats
        cluster_csv = f"{out_prefix}_nlist{nlist}_visit_stats.csv"
        # columns: cluster_id, list_size, visits, scanned_vecs
        out = np.c_[np.arange(nlist), list_sizes, visit_counts, scanned_vecs]
        np.savetxt(
            cluster_csv,
            out,
            delimiter=",",
            header="cluster_id,list_size,visits,scanned_vecs",
            comments="",
            fmt=["%d", "%d", "%d", "%d"]
        )
        print(f"\nSaved: {cluster_csv}")

        # visit distribution
        dist_csv = f"{out_prefix}_nlist{nlist}_visit_distribution.csv"
        with open(dist_csv, "w", encoding="utf-8") as f:
            f.write("visits,n_clusters\n")
            for v in sorted(visit_dist.keys()):
                f.write(f"{v},{visit_dist[v]}\n")
        print(f"Saved: {dist_csv}")

    return I, visit_counts, scanned_vecs, list_sizes


if __name__ == "__main__":
    base_path = r"/home/zhangshujie/ann_nic/sift/sift_base.fvecs"
    # SIFT1M 的 query 常见叫 sift_query.fvecs / sift_query.fvecs
    query_path = r"/home/zhangshujie/ann_nic/sift/sift_query.fvecs"

    main_ivf_visit_stats(
        base_fvecs_path=base_path,
        query_fvecs_path=query_path,
        nlist=4096,          # 你也可以换成 8192/16384/100000
        nprobe=16,
        k=10,
        query_batch_size=1000,
        out_prefix="sift1m_ivf"
    )
