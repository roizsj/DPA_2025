#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def read_fvecs(path: str, max_vecs: int | None = None) -> np.ndarray:
    """
    Read FAISS-style .fvecs:
      each vector: int32 dim, followed by dim float32.
    Returns float32 array of shape (n, dim).
    """
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size < 1:
        raise ValueError(f"Empty file: {path}")

    dim = int(raw[0])
    if dim <= 0 or dim > 100000:
        raise ValueError(f"Unreasonable dim={dim} in {path}")

    rec_len = 1 + dim
    if raw.size % rec_len != 0:
        raise ValueError(f"File size mismatch: total int32={raw.size}, rec_len={rec_len}")

    n = raw.size // rec_len
    if max_vecs is not None:
        n = min(n, max_vecs)

    raw = raw[: n * rec_len].reshape(n, rec_len)
    X = raw[:, 1:].view(np.float32)
    return X.copy()


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, eps)


def knn_kth_distance_sklearn(X: np.ndarray, k: int) -> np.ndarray:
    """
    d_k(i): distance from each point to its k-th nearest neighbor (excluding itself).
    Uses sklearn brute-force.
    """
    n, _ = X.shape
    if k >= n:
        raise ValueError(f"k must be < n (k={k}, n={n})")

    kk = k + 1  # include self (distance 0)
    nn = NearestNeighbors(n_neighbors=kk, algorithm="brute", metric="euclidean")
    nn.fit(X)
    dist, _ = nn.kneighbors(X, return_distance=True)
    return dist[:, k].astype(np.float64)


def density_proxy_log_rho(dk: np.ndarray, k: int, dim: int, eps: float = 1e-12) -> np.ndarray:
    """
    rho_i ∝ k / (dk^D), ignore constant V_D.
    log rho = log k - D log(dk) + const (const ignored)
    """
    dk_safe = np.maximum(dk, eps)
    return (math.log(k) - dim * np.log(dk_safe)).astype(np.float64)


def summarize(name: str, dk: np.ndarray, log_rho: np.ndarray) -> dict:
    mean = float(np.mean(dk))
    std = float(np.std(dk))
    cv = std / mean if mean > 0 else float("nan")
    return {
        "name": name,
        "dk_mean": mean,
        "dk_std": std,
        "dk_cv": cv,
        "logrho_p5": float(np.percentile(log_rho, 5)),
        "logrho_p50": float(np.percentile(log_rho, 50)),
        "logrho_p95": float(np.percentile(log_rho, 95)),
        "logrho_iqr": float(np.percentile(log_rho, 75) - np.percentile(log_rho, 25)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fvecs", type=str, required=True, help="Path to sift_base.fvecs (or any .fvecs)")
    ap.add_argument("--sample", type=int, default=20000, help="Random sample size (default 20k for sklearn brute)")
    ap.add_argument("--k", type=int, default=10, help="k for kNN distance (default 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", action="store_true", help="L2-normalize vectors before analysis")
    ap.add_argument("--plots", action="store_true", help="Show plots")
    ap.add_argument("--savefig", type=str, default="", help="If set, save plots with this prefix, e.g. out/sift")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    X = read_fvecs(args.fvecs)
    n, d = X.shape
    print(f"[info] Loaded: n={n}, d={d} from {args.fvecs}")

    m = min(args.sample, n)
    idx = rng.choice(n, size=m, replace=False)
    Xs = X[idx].astype(np.float32, copy=False)

    if args.normalize:
        Xs = l2_normalize(Xs).astype(np.float32, copy=False)
        print("[info] Applied L2 normalization")

    # Gaussian baseline (same m, d)
    G = rng.standard_normal(size=(m, d)).astype(np.float32)
    if args.normalize:
        G = l2_normalize(G).astype(np.float32, copy=False)

    print(f"[info] Computing d_k with sklearn brute: m={m}, k={args.k} ...")
    dk_sift = knn_kth_distance_sklearn(Xs, k=args.k)
    dk_gaus = knn_kth_distance_sklearn(G,  k=args.k)

    logrho_sift = density_proxy_log_rho(dk_sift, k=args.k, dim=d)
    logrho_gaus = density_proxy_log_rho(dk_gaus, k=args.k, dim=d)

    s1 = summarize("SIFT", dk_sift, logrho_sift)
    s2 = summarize("Gaussian", dk_gaus, logrho_gaus)

    print("\n===== Summary =====")
    for s in (s1, s2):
        print(
            f"{s['name']:8s} | "
            f"dk_mean={s['dk_mean']:.6g}  dk_std={s['dk_std']:.6g}  dk_CV={s['dk_cv']:.6g} | "
            f"logrho p5/p50/p95 = {s['logrho_p5']:.3f} / {s['logrho_p50']:.3f} / {s['logrho_p95']:.3f} | "
            f"logrho IQR={s['logrho_iqr']:.3f}"
        )

    # Plots
    if args.plots or args.savefig:
        # d_k histogram
        plt.figure()
        plt.hist(dk_sift, bins=80, alpha=0.6, label="SIFT d_k")
        plt.hist(dk_gaus, bins=80, alpha=0.6, label="Gaussian d_k")
        plt.xlabel(f"d_k (k={args.k})")
        plt.ylabel("count")
        plt.title("kNN distance distribution (Method 1)")
        plt.legend()
        if args.savefig:
            outdir = os.path.dirname(args.savefig)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            plt.savefig(args.savefig + f"_dk_k{args.k}.png", dpi=200, bbox_inches="tight")
        if args.plots:
            plt.show()
        else:
            plt.close()

        # log rho histogram
        plt.figure()
        plt.hist(logrho_sift, bins=100, alpha=0.6, label="SIFT log(rho)")
        plt.hist(logrho_gaus, bins=100, alpha=0.6, label="Gaussian log(rho)")
        plt.xlabel("log(rho) (up to constant)")
        plt.ylabel("count")
        plt.title("Local density proxy distribution (Method 2)")
        plt.legend()
        if args.savefig:
            plt.savefig(args.savefig + f"_logrho_k{args.k}.png", dpi=200, bbox_inches="tight")
        if args.plots:
            plt.show()
        else:
            plt.close()

    print("\n[done] 判据：SIFT 的 dk_CV 明显更大、log(rho) 分布更宽/长尾 => 点分布更不均匀（更聚簇）。")


if __name__ == "__main__":
    main()
