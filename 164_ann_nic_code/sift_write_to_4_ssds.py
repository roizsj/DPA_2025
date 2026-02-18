import argparse, json
from pathlib import Path
import numpy as np

def read_fvecs_mmap(path: str) -> np.ndarray:
    data = np.memmap(path, dtype=np.int32, mode="r")
    if data.size == 0:
        raise ValueError("empty fvecs")
    dim = int(data[0])
    stride = 1 + dim
    if data.size % stride != 0:
        raise ValueError("fvecs size mismatch")
    n = data.size // stride
    x = data.view(np.float32).reshape(n, stride)[:, 1:]
    return np.asarray(x, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_fvecs", required=True)
    ap.add_argument("--artifacts_dir", required=True, help="contains perm_by_cluster.npy, cluster_offsets.npy, cluster_to_disk.npy, centroids.npy")
    ap.add_argument("--out_root", required=True, help="metadata output dir (can be on any disk)")
    ap.add_argument("--disk_mounts", nargs=4, required=True, help="e.g. /mnt/nvme1 /mnt/nvme2 /mnt/nvme3 /mnt/nvme4")
    ap.add_argument("--write_chunk", type=int, default=50000, help="vectors per chunk write")
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    perm = np.load(art / "perm_by_cluster.npy")        # (N,)
    offsets = np.load(art / "cluster_offsets.npy")     # (k+1,)
    c2d = np.load(art / "cluster_to_disk.npy")         # (k,)
    centroids = np.load(art / "centroids.npy").astype(np.float32)

    X = read_fvecs_mmap(args.base_fvecs)
    N, d = X.shape
    k = int(c2d.shape[0])

    assert perm.shape[0] == N, f"perm size {perm.shape[0]} != N {N}"
    assert offsets.shape[0] == k + 1, f"offsets size {offsets.shape[0]} != k+1"
    assert centroids.shape[1] == d, f"centroids dim {centroids.shape[1]} != d {d}"

    disk_dirs = []
    for mp in args.disk_mounts:
        dd = Path(mp) / "sift1m_ivf"
        dd.mkdir(parents=True, exist_ok=True)
        disk_dirs.append(dd)

    fouts = []
    for dd in disk_dirs:
        fouts.append(open(dd / "base_vectors.f32", "wb", buffering=0))

    # disk_offsets[cid] = (start,end) in *vectors* within that disk file
    disk_offsets = np.full((k, 2), -1, dtype=np.int64)
    disk_write_pos = np.zeros(4, dtype=np.int64)

    for cid in range(k):
        s, e = int(offsets[cid]), int(offsets[cid + 1])
        disk = int(c2d[cid])
        start_in_disk = int(disk_write_pos[disk])

        if e > s:
            idxs = perm[s:e]
            for p in range(0, idxs.shape[0], args.write_chunk):
                sub = idxs[p:p + args.write_chunk]
                buf = X[sub]  # float32
                fouts[disk].write(buf.tobytes(order="C"))
            disk_write_pos[disk] += (e - s)

        end_in_disk = int(disk_write_pos[disk])
        disk_offsets[cid] = (start_in_disk, end_in_disk)

    for f in fouts:
        f.close()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    np.save(out_root / "disk_offsets.npy", disk_offsets)
    np.save(out_root / "cluster_to_disk.npy", c2d.astype(np.int16, copy=False))
    np.save(out_root / "centroids.npy", centroids.astype(np.float32, copy=False))

    meta = {
        "N": int(N),
        "d": int(d),
        "k": int(k),
        "disk_mounts": args.disk_mounts,
        "disk_relpath": "sift1m_ivf/base_vectors.f32",
        "files": {
            "disk_offsets": "disk_offsets.npy",
            "cluster_to_disk": "cluster_to_disk.npy",
            "centroids": "centroids.npy",
        },
        "layout": "cluster-contiguous within each disk file; clusters assigned by cluster_to_disk",
    }
    with open(out_root / "layout_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[OK] wrote disk files:")
    for dd in disk_dirs:
        print(" ", dd / "base_vectors.f32")
    print("[OK] wrote meta:", out_root)

if __name__ == "__main__":
    main()