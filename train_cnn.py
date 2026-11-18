"""Train the digit classifier.

  python train_cnn.py --data dataset --synthetic 600 --epochs 30 --out models/digit_cnn.pt
  python train_cnn.py --data dataset --backend sklearn --out models/digit_mlp.pkl   # no PyTorch needed

`dataset/` = folders 0..9 of 28x28 PNG crops (made by `python -m sevenseg.crops export ...`
then corrected by hand).  --synthetic N adds crops from N generated meter images.
"""
import argparse
import os
import pickle

import numpy as np

from sevenseg.classifier import augment, build_torch_model
from sevenseg.crops import load_folder, synthetic


def split_by_group(groups, rng, validation_fraction=0.15):
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two independent source images are needed for validation.")
    order = rng.permutation(unique)
    count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    validation_groups = set(order[:count])
    validation = np.array([group in validation_groups for group in groups])
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--synthetic", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--backend", choices=["torch", "sklearn"], default="torch")
    ap.add_argument("--out", default="models/digit_cnn.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verified-only", action="store_true",
                    help="train only on crop files whose names include _verified_")
    ap.add_argument("--pretrained", help="initialize PyTorch training from existing model weights")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    Xs, ys, groups = [], [], []
    if a.data:
        if not os.path.isdir(a.data):
            raise FileNotFoundError(f"Digit-crop dataset folder not found: {a.data}")
        X, y, source_groups = load_folder(a.data, verified_only=a.verified_only)
        if not len(X):
            raise ValueError(f"No readable digit crops found under {a.data}.")
        print("real crops:", len(X))
        Xs.append(X); ys.append(y); groups.append(np.char.add("real-", source_groups))
    if a.synthetic:
        X, y, source_groups = synthetic(a.synthetic, a.seed)
        print("synthetic crops:", len(X))
        if len(X):
            Xs.append(X); ys.append(y); groups.append(source_groups)
    if not Xs:
        raise ValueError("Provide a labeled crop folder or a positive --synthetic count.")
    X, y, groups = np.concatenate(Xs), np.concatenate(ys), np.concatenate(groups)
    tr, va = split_by_group(groups, rng)
    print(f"validation crops: {len(va)} from {len(np.unique(groups[va]))} held-out source images")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def aug_set(ix, k):
        return np.stack([augment(X[i], rng) for i in np.repeat(ix, k)]), np.repeat(y[ix], k)

    if a.backend == "sklearn":
        if a.pretrained:
            raise ValueError("--pretrained is supported only with the torch backend.")
        from sklearn.neural_network import MLPClassifier
        Xa, ya = aug_set(tr, 4)
        m = MLPClassifier((128, 64), max_iter=200, random_state=a.seed).fit(Xa.reshape(len(Xa), -1), ya)
        print("val acc:", m.score(X[va].reshape(len(va), -1), y[va]))
        pickle.dump(m, open(a.out, "wb")); return

    import torch, torch.nn as nn
    model = build_torch_model()
    if a.pretrained:
        if not os.path.isfile(a.pretrained):
            raise FileNotFoundError(f"Pretrained model not found: {a.pretrained}")
        model.load_state_dict(torch.load(a.pretrained, map_location="cpu", weights_only=True))
    learning_rate = 3e-4 if a.pretrained else 2e-3
    model_weight_decay = 1e-3
    opt = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=model_weight_decay)
    max_lr = 8e-4 if a.pretrained else 5e-3
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, total_steps=a.epochs * (len(tr) * 3 // 64 + 1))
    Xv = torch.from_numpy(X[va][:, None]); yv = torch.from_numpy(y[va])
    best = 0.0
    for ep in range(a.epochs):
        model.train(); Xa, ya = aug_set(tr, 3); p = rng.permutation(len(Xa))
        for i in range(0, len(p), 64):
            b = p[i:i + 64]
            loss = nn.functional.cross_entropy(model(torch.from_numpy(Xa[b][:, None])), torch.from_numpy(ya[b]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        print(f"epoch {ep + 1:2d}  loss {loss.item():.3f}  val acc {acc:.3f}")
        if acc >= best:
            best = acc; torch.save(model.state_dict(), a.out)
    print("best val acc", best, "->", a.out)


if __name__ == "__main__":
    main()
