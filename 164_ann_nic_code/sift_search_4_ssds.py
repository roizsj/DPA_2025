import os, time, json, argparse
from pathlib import Path
import numpy as np

def read_fvecs_mmap(path: str) -> np.ndarray:
    data = np.memmap(path, dtype=np.int32, mode="r")
    dim = int(data[0])
    stride = 1 + dim
    n = data.size // stride
    x = data.view(np.float32).reshape(n, stride)[:, 1:]
    return np.asarray(x, dtype=np.float32)

def compute_nprobe_lists(Q: np.ndarray, centroids: np.ndarray, nprobe: int, block_q: int = 4096) -> np.ndarray:
    k, d = centroids.shape
    c_norm = (centroids * centroids).sum(axis=1, dtype=np.float32)
    out = []
    for s in range(0, Q.shape[0], block_q):
        qb = Q[s:s+block_q]
        q_norm = (qb * qb).sum(axis=1, dtype=np.float32)
        dots = qb @ centroids.T
        dist = q_norm[:, None] + c_norm[None, :] - 2.0 * dots
        idx = np.argpartition(dist, kth=nprobe-1, axis=1)[:, :nprobe]
        out.append(idx.astype(np.int32))
    return np.vstack(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_fvecs", required=True)
    ap.add_argument("--layout_root", required=True, help="folder containing disk_offsets.npy, cluster_to_disk.npy, centroids.npy, layout_meta.json")
    ap.add_argument("--disk_mounts", nargs=4, required=True)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--block_q", type=int, default=4096)
    ap.add_argument("--read_mode", choices=["pread", "seekread"], default="pread")
    ap.add_argument("--merge_adjacent", action="store_true", help="merge adjacent cluster ranges on same disk to reduce syscalls")
    args = ap.parse_args()

    root = Path(args.layout_root)
    disk_offsets = np.load(root / "disk_offsets.npy")          # (k,2) in vectors
    c2d = np.load(root / "cluster_to_disk.npy")                # (k,)
    centroids = np.load(root / "centroids.npy").astype(np.float32)

    d = int(centroids.shape[1])
    vec_bytes = d * 4

    # open fds
    fds = []
    for mp in args.disk_mounts:
        p = Path(mp) / "sift1m_ivf" / "base_vectors.f32"
        fds.append(os.open(str(p), os.O_RDONLY))

    Q = read_fvecs_mmap(args.query_fvecs)
    probe = compute_nprobe_lists(Q, centroids, args.nprobe, block_q=args.block_q)  # (nq,nprobe)

    def empty_stats():
        return {"io_s": 0.0, "bytes": 0, "vecs": 0, "reads": 0}

    totals = [empty_stats() for _ in range(4)]

    for r in range(args.repeat):
        stats = [empty_stats() for _ in range(4)]
        wall0 = time.perf_counter()

        for qi in range(probe.shape[0]):
            # collect ranges per disk
            per = [[] for _ in range(4)]  # list of (start_vec, end_vec) in disk file
            for cid in probe[qi]:
                disk = int(c2d[cid])
                s, e = disk_offsets[cid]
                if e > s:
                    per[disk].append((int(s), int(e)))

            for disk in range(4):
                if not per[disk]:
                    continue
                ranges = per[disk]

                # optional: merge adjacent/overlapping ranges (reduces syscalls)
                if args.merge_adjacent:
                    ranges.sort()
                    merged = []
                    cs, ce = ranges[0]
                    for s, e in ranges[1:]:
                        if s <= ce:  # overlap
                            ce = max(ce, e)
                        elif s == ce:  # adjacent
                            ce = e
                        else:
                            merged.append((cs, ce))
                            cs, ce = s, e
                    merged.append((cs, ce))
                    ranges = merged

                fd = fds[disk]
                for s_vec, e_vec in ranges:
                    nvec = e_vec - s_vec
                    off = s_vec * vec_bytes
                    nbytes = nvec * vec_bytes

                    t0 = time.perf_counter()
                    if args.read_mode == "pread":
                        _ = os.pread(fd, nbytes, off)
                    else:
                        os.lseek(fd, off, os.SEEK_SET)
                        _ = os.read(fd, nbytes)
                    t1 = time.perf_counter()

                    stats[disk]["io_s"] += (t1 - t0)
                    stats[disk]["bytes"] += nbytes
                    stats[disk]["vecs"] += nvec
                    stats[disk]["reads"] += 1

        wall1 = time.perf_counter()
        print(f"[Run {r+1}] wall={wall1-wall0:.3f}s")

        for disk in range(4):
            for k in totals[disk]:
                totals[disk][k] += stats[disk][k]

    for disk in range(4):
        for k in totals[disk]:
            totals[disk][k] /= args.repeat

    out = {
        "nq": int(Q.shape[0]),
        "d": int(d),
        "nprobe": int(args.nprobe),
        "repeat": int(args.repeat),
        "read_mode": args.read_mode,
        "merge_adjacent": bool(args.merge_adjacent),
        "per_disk": totals,
    }
    print(json.dumps(out, indent=2))

    for fd in fds:
        os.close(fd)

if __name__ == "__main__":
    main()