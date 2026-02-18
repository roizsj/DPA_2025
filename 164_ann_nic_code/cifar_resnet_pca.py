import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt


class ResNet50Pool(nn.Module):
    """ResNet-50 up to global average pooling => 2048-d features."""
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # keep everything except final fc
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # (B,2048,1,1)

    def forward(self, x):
        y = self.backbone(x).flatten(1)  # (B,2048)
        return y


def make_fixed_projection(in_dim=2048, out_dim=1024, seed=7):
    """
    Fixed random projection matrix W (in_dim x out_dim).
    We'll use Gaussian and column-normalize for stability.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=50000, help="use at most N images from CIFAR-10 train set")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--proj_dim", type=int, default=1024)
    ap.add_argument("--ipca_batch", type=int, default=4096)
    ap.add_argument("--out_png", type=str, default="resnet_proj1024_pca_cumvar.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # CPU only
    device = torch.device("cpu")
    torch.set_num_threads(max(1, os_cpu_threads()))

    # CIFAR-10 -> resize to 224 for ResNet
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)

    # Optional: limit size for speed
    if args.limit and args.limit < len(ds):
        indices = np.random.default_rng(args.seed).choice(len(ds), size=args.limit, replace=False)
        ds = torch.utils.data.Subset(ds, indices.tolist())

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=False)

    # Model
    model = ResNet50Pool().to(device)
    model.eval()

    # Fixed projection 2048 -> proj_dim
    W = make_fixed_projection(2048, args.proj_dim, seed=args.seed)

    # Incremental PCA on projected embeddings
    ipca = IncrementalPCA(n_components=args.proj_dim, batch_size=args.ipca_batch)

    # 1st pass: partial_fit
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            feat2048 = model(xb).cpu().numpy().astype(np.float32)  # (B,2048)
            emb = feat2048 @ W  # (B,1024)
            ipca.partial_fit(emb)

    # explained variance ratio cumulative
    evr = ipca.explained_variance_ratio_
    cum = np.cumsum(evr)
    x = np.arange(1, args.proj_dim + 1)

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, cum, linewidth=2)
    plt.xlabel("PCA components (x)")
    plt.ylabel("Cumulative explained variance ratio (1..x)")
    plt.ylim(0, 1.01)
    plt.xlim(1, args.proj_dim)
    plt.grid(True, linewidth=0.5)
    plt.title("CIFAR-10-RESNET-1024D")
    for t in [0.80, 0.90, 0.95, 0.99]:
        k = int(np.searchsorted(cum, t) + 1)
        plt.text(min(k + 1, 1024), min(t + 0.01, 1.0), f"{int(t*100)}% @ {k}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"[OK] saved: {args.out_png}")
    print(f"[OK] used images: {len(ds)}, embedding dim: {args.proj_dim}")


def os_cpu_threads():
    # safe default if os.cpu_count() is None
    import os
    c = os.cpu_count() or 4
    # don't over-thread too hard on shared machines
    return min(c, 16)


if __name__ == "__main__":
    main()
