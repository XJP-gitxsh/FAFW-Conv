"""Reusable classification and reconstruction-anomaly experiments for FAFW-Conv.

Classification input: UEA .ts train/test files.
Anomaly input: train.npy/test.npy/test_label.npy or equivalent CSV files.
Thresholds use only a held-out normal validation split from the training file.
"""
import os
import glob
import csv
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from models.ModernTCN import Model as ForecastModel

_UEA_CACHE = {}


def _read_uea_ts(path):
    """Memory-efficient reader for equal or variable length UEA .ts files."""
    has_labels = True
    timestamps = False
    declared_channels = declared_length = None
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            low = line.lower()
            if low.startswith("@classlabel"):
                parts = line.split()
                has_labels = len(parts) > 1 and parts[1].lower() == "true"
            elif low.startswith("@timestamps"):
                parts = line.split()
                timestamps = len(parts) > 1 and parts[1].lower() == "true"
            elif low.startswith("@dimensions"):
                declared_channels = int(line.split()[1])
            elif low.startswith("@serieslength"):
                declared_length = int(line.split()[1])
            elif low == "@data":
                break
        else:
            raise ValueError("UEA .ts file has no @data marker: " + path)
    if timestamps:
        raise ValueError("Timestamped UEA .ts input is not supported by this reader: " + path)

    # First pass determines the allocation size without retaining Python float
    # objects for every value (important for FaceDetection).
    sample_count = max_channels = max_length = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        in_data = False
        for raw in f:
            line = raw.strip()
            if not in_data:
                if line.lower() == "@data": in_data = True
                continue
            if not line or line.startswith("#"): continue
            fields = line.split(":")
            if has_labels: fields.pop()
            sample_count += 1
            max_channels = max(max_channels, len(fields))
            for field in fields: max_length = max(max_length, field.count(",") + (1 if field.strip() else 0))
    if sample_count == 0:
        raise ValueError("No UEA samples found in: " + path)
    channels = declared_channels or max_channels
    length = declared_length or max_length
    arr = np.zeros((sample_count, channels, length), dtype=np.float32)
    labels = []
    with open(path, "r", encoding="utf-8-sig") as f:
        in_data = False; i = 0
        for raw in f:
            line = raw.strip()
            if not in_data:
                if line.lower() == "@data": in_data = True
                continue
            if not line or line.startswith("#"): continue
            fields = line.split(":")
            label = fields.pop() if has_labels else "0"
            labels.append(label)
            for j, field in enumerate(fields[:channels]):
                if "?" in field:
                    vals = np.asarray([np.nan if v.strip() == "?" else float(v)
                                       for v in field.split(",") if v.strip()], dtype=np.float32)
                else:
                    vals = np.fromstring(field, dtype=np.float32, sep=",")
                n = min(length, len(vals))
                if n:
                    if np.isnan(vals[:n]).any():
                        good = np.flatnonzero(np.isfinite(vals[:n]))
                        vals[:n] = np.interp(np.arange(n), good, vals[good]) if len(good) else 0.0
                    arr[i, j, :n] = vals[:n]
                    if n < length: arr[i, j, n:] = vals[n-1]
            i += 1
    return arr, np.asarray(labels)


class UEAClassificationData(Dataset):
    def __init__(self, root, dataset, split="train", val_fraction=0.1, seed=2021):
        folder = os.path.join(root, dataset)
        train_path = os.path.join(folder, dataset + "_TRAIN.ts")
        test_path = os.path.join(folder, dataset + "_TEST.ts")
        if not os.path.isfile(train_path):
            raise FileNotFoundError("Expected UEA files: " + train_path + " and " + test_path)
        cache_key=(os.path.abspath(train_path),os.path.abspath(test_path))
        if cache_key not in _UEA_CACHE:
            _UEA_CACHE[cache_key]=(_read_uea_ts(train_path),_read_uea_ts(test_path))
        (train_x, train_y), (test_x, test_y) = _UEA_CACHE[cache_key]
        # Some valid UEA datasets use different fixed lengths for train/test
        # splits (e.g. JapaneseVowels). Preserve all points by padding both
        # splits to their joint shape instead of truncating the test sequences.
        max_len = max(train_x.shape[-1], test_x.shape[-1])
        max_ch = max(train_x.shape[1], test_x.shape[1])
        test_x = _fit_shape(test_x, max_ch, max_len)
        train_x = _fit_shape(train_x, max_ch, max_len)
        classes = sorted(set(train_y.tolist()))
        self.class_to_id = {v: i for i, v in enumerate(classes)}
        train_y = np.asarray([self.class_to_id[v] for v in train_y], dtype=np.int64)
        test_y = np.asarray([self.class_to_id[v] for v in test_y], dtype=np.int64)
        idx = np.arange(len(train_y))
        tr, va = train_test_split(idx, test_size=val_fraction, random_state=seed,
                                  stratify=train_y if len(classes) > 1 else None)
        if split in ("train", "val"):
            ids = tr if split == "train" else va
            x, y = train_x[ids], train_y[ids]
        else:
            x, y = test_x, test_y
        # Per-channel statistics are fit on the training partition only.
        fit_x = train_x[tr]
        mu = fit_x.mean(axis=(0, 2), keepdims=True)
        sd = fit_x.std(axis=(0, 2), keepdims=True) + 1e-6
        self.x = torch.from_numpy(((x - mu) / sd).astype(np.float32))
        self.y = torch.from_numpy(y)
        self.num_classes = len(classes)
        self.num_channels = max_ch
        self.seq_len = max_len

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i].transpose(0, 1), self.y[i]


def _fit_shape(x, channels, length):
    if x.shape[1] == channels and x.shape[2] == length:
        return x
    out = np.zeros((x.shape[0], channels, length), dtype=np.float32)
    c, t = min(channels, x.shape[1]), min(length, x.shape[2])
    out[:, :c, :t] = x[:, :c, :t]
    if t < length and t > 0:
        out[:, :c, t:] = out[:, :c, t-1:t]
    return out


def _numeric_csv(path):
    df = pd.read_csv(path)
    label_col = next((c for c in df.columns if str(c).strip().lower() in
                      ("label", "labels", "attack", "normal/attack", "is_anomaly")), None)
    labels = None
    if label_col is not None:
        raw = df.pop(label_col)
        if raw.dtype == object:
            vals = raw.astype(str).str.lower().str.strip()
            labels = (~vals.isin(["normal", "0", "false", "none"])).astype(np.int64).to_numpy()
        else:
            labels = (raw.to_numpy() != 0).astype(np.int64)
    # Drop timestamp columns and coerce features.
    for c in list(df.columns):
        if not pd.api.types.is_numeric_dtype(df[c]):
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().sum() == 0:
                df = df.drop(columns=[c])
            else:
                df[c] = converted
    arr = df.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").fillna(0).to_numpy(np.float32)
    return arr, labels


def _load_array(root, stem, dataset=None):
    names = ([dataset + "_" + stem, stem + "_" + dataset, stem] if dataset else [stem])
    candidates = []
    for name in names:
        candidates += glob.glob(os.path.join(root, "**", name + ".npy"), recursive=True)
    if candidates:
        return np.load(candidates[0], allow_pickle=False)
    candidates = []
    for name in names:
        candidates += glob.glob(os.path.join(root, "**", name + ".csv"), recursive=True)
    if candidates:
        return _numeric_csv(candidates[0])
    return None


def _load_smd_series(root, series_name):
    """Load one SMD machine from the conventional train/test/test_label folders."""
    paths = {}
    for split in ("train", "test", "test_label"):
        folder = os.path.join(root, split)
        files = sorted(glob.glob(os.path.join(folder, "*.txt")))
        if not files:
            raise FileNotFoundError("Expected SMD machine files under " + folder)
        if series_name:
            matches = [p for p in files if os.path.splitext(os.path.basename(p))[0] == series_name]
            if not matches:
                raise FileNotFoundError("SMD series '{}' not found under {}".format(series_name, folder))
            paths[split] = matches[0]
        else:
            paths[split] = files
    if not series_name:
        stems = {os.path.splitext(os.path.basename(p))[0] for p in paths["train"]}
        if len(stems) != 1:
            raise ValueError("SMD has multiple machines; pass --series_name (for example machine-1-1)")
        series_name = next(iter(stems))
        for split in paths:
            paths[split] = next(p for p in paths[split]
                                if os.path.splitext(os.path.basename(p))[0] == series_name)
    train = np.loadtxt(paths["train"], delimiter=",")
    test = np.loadtxt(paths["test"], delimiter=",")
    labels = np.loadtxt(paths["test_label"], delimiter=",")
    return train, test, labels, series_name


def load_anomaly_arrays(root, dataset=None, series_name=None):
    if str(dataset).upper() == "SMD" and os.path.isdir(os.path.join(root, "train")):
        return _load_smd_series(root, series_name)
    train = _load_array(root, "train", dataset)
    test = _load_array(root, "test", dataset)
    label = _load_array(root, "test_label", dataset)
    if train is None or test is None:
        raise FileNotFoundError("Need train.npy/csv and test.npy/csv under --data_root")
    train = train[0] if isinstance(train, tuple) else train
    embedded_labels = None
    if isinstance(test, tuple):
        test, embedded_labels = test
        if label is None: label = embedded_labels
    if isinstance(label, tuple):
        label_array, label_column = label
        label = label_column if label_column is not None else np.asarray(label_array).squeeze()
    if label is None:
        raise FileNotFoundError("Need test_label.npy/csv or a label column in test.csv")
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    label = np.asarray(label).reshape(-1)
    if train.ndim == 3: train = train.reshape(-1, train.shape[-1])
    if test.ndim == 3: test = test.reshape(-1, test.shape[-1])
    if test.shape[0] != label.shape[0]:
        raise ValueError("test and test_label lengths differ")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train/test feature dimensions differ")
    return train, test, (label != 0).astype(np.int64), series_name


class AnomalyWindows(Dataset):
    def __init__(self, values, labels=None, window=100, stride=None):
        self.values = torch.from_numpy(np.asarray(values, dtype=np.float32))
        self.labels = None if labels is None else torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be a positive integer")
        self.stride = int(stride or max(1, window // 2))
        if self.stride < 1 or self.stride > self.window:
            raise ValueError("stride must be positive and no larger than window to cover every point")
        self.starts = list(range(0, max(1, len(values) - window + 1), self.stride))
        if not self.starts or self.starts[-1] + window < len(values):
            self.starts.append(max(0, len(values) - window))
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = self.starts[i]
        x = self.values[s:s+self.window]
        y = torch.zeros(len(x), dtype=torch.long) if self.labels is None else self.labels[s:s+self.window]
        if len(x) < self.window:
            pad = self.window - len(x)
            x = torch.cat([x, x[-1:].repeat(pad, 1)], 0)
            y = torch.cat([y, y[-1:].repeat(pad)], 0)
        return x, y


def make_backbone_config(args, channels, seq_len, pred_len):
    return SimpleNamespace(
        stem_ratio=6, downsample_ratio=2, ffn_ratio=args.ffn_ratio,
        num_blocks=[1], large_size=[args.large_size], small_size=[5],
        dims=[args.dims] * 4, dw_dims=[args.dims] * 4, enc_in=channels,
        small_kernel_merged=False, dropout=args.dropout, head_dropout=0.0,
        use_multi_scale=False, revin=True, affine=False, subtract_last=False,
        freq="h", seq_len=seq_len, c_in=channels, individual=False,
        pred_len=pred_len, kernel_size=25, patch_size=args.patch_size,
        patch_stride=args.patch_stride, decomposition=False,
        fractional_mode=args.fractional_mode, use_hhconv=args.use_hhconv,
        use_wavelet=args.use_wavelet,
    )


class Classifier(nn.Module):
    def __init__(self, args, channels, seq_len, n_classes):
        super().__init__()
        cfg = make_backbone_config(args, channels, seq_len, 1)
        self.backbone = ForecastModel(cfg)
        self.classifier = nn.Linear(channels * args.dims, n_classes)
    def forward(self, x):
        feat = self.backbone.model.encode_features(x)
        pooled = feat.mean(dim=-1).flatten(1)
        return self.classifier(pooled)


def train_classification(args, device):
    tr = UEAClassificationData(args.data_root, args.dataset, "train", args.val_fraction, args.seed)
    va = UEAClassificationData(args.data_root, args.dataset, "val", args.val_fraction, args.seed)
    te = UEAClassificationData(args.data_root, args.dataset, "test", args.val_fraction, args.seed)
    model = Classifier(args, tr.num_channels, tr.seq_len, tr.num_classes).to(device)
    loaders = [DataLoader(d, batch_size=args.batch_size, shuffle=(i==0), num_workers=0)
               for i, d in enumerate((tr, va, te))]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, best_epoch = float("inf"), None, 0
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in loaders[0]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            opt.step()
        model.eval(); losses=[]
        with torch.no_grad():
            for x, y in loaders[1]: losses.append(loss_fn(model(x.to(device)), y.to(device)).item())
        val = float(np.mean(losses))
        if val < best:
            best, best_epoch = val, epoch + 1
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        print("epoch", epoch+1, "val_loss", round(val, 6))
        if stale_epochs >= args.patience:
            print("early_stop", "best_epoch", best_epoch)
            break
    if best_state is None:
        raise RuntimeError("Classification training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "classification_{}_seed{}.pt".format(args.dataset, args.seed)))
    model.eval(); yp=[]; yt=[]
    with torch.no_grad():
        for x,y in loaders[2]: yp.extend(model(x.to(device)).argmax(-1).cpu().tolist()); yt.extend(y.tolist())
    score = accuracy_score(yt, yp)
    print("test_accuracy", score)
    return {"accuracy": float(score), "n_test": len(yt), "best_val_loss": best,
            "best_epoch": best_epoch, "epochs_run": epoch + 1}


def _aggregate_reconstruction_scores(model, loader, dataset, device):
    """Average overlapping window reconstruction errors back to point scores."""
    total = np.zeros(len(dataset.values), dtype=np.float64)
    count = np.zeros_like(total)
    offset = 0
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x_device = x.to(device)
            errors = (model(x_device) - x_device).pow(2).mean(-1).cpu().numpy()
            for row, window_scores in enumerate(errors):
                start = dataset.starts[offset + row]
                end = min(len(total), start + dataset.window)
                valid = end - start
                total[start:end] += window_scores[:valid]
                count[start:end] += 1
            offset += len(errors)
    if offset != len(dataset):
        raise RuntimeError("Reconstruction scoring did not cover every window")
    return total / np.maximum(count, 1)


def train_anomaly(args, device):
    train, test, labels, series_name = load_anomaly_arrays(
        args.data_root, args.dataset, getattr(args, "series_name", None))
    split = max(args.window, int(len(train) * (1-args.val_fraction)))
    if len(train) - split < args.window:
        split = len(train) - args.window
    if split < args.window or len(train) - split < args.window:
        raise ValueError("Training series must contain at least two windows so fit and validation are disjoint")
    # Fit normalization on the gradient-training segment only. The held-out
    # training tail is used for early stopping and threshold calibration.
    fit_raw = train[:split]
    mu = fit_raw.mean(0, keepdims=True); sd = fit_raw.std(0, keepdims=True) + 1e-6
    train, test = (train-mu)/sd, (test-mu)/sd
    fit_values, val_values = train[:split], train[split:]
    fit_ds = AnomalyWindows(fit_values, window=args.window, stride=args.stride)
    val_ds = AnomalyWindows(val_values, window=args.window, stride=args.stride)
    test_ds = AnomalyWindows(test, labels, window=args.window, stride=args.stride)
    fit_dl = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    channels = train.shape[1]
    cfg = make_backbone_config(args, channels, args.window, args.window)
    model = ForecastModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, best_vals, best_epoch = float("inf"), None, None, 0
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        for x, _ in fit_dl:
            x=x.to(device); opt.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(model(x), x)
            loss.backward(); opt.step()
        model.eval()
        val_scores = _aggregate_reconstruction_scores(model, val_dl, val_ds, device)
        val=float(np.mean(val_scores))
        if val<best:
            best,best_epoch=val,epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            # Threshold calibration uses the same point-level score aggregation
            # as test evaluation, including the averaging of overlapping windows.
            best_vals=val_scores
            stale_epochs=0
        else:
            stale_epochs+=1
        print("epoch",epoch+1,"val_reconstruction_mse",round(val,6))
        if stale_epochs>=args.patience:
            print("early_stop","best_epoch",best_epoch)
            break
    if best_state is None or len(best_vals) == 0:
        raise RuntimeError("Anomaly training did not produce a valid checkpoint")
    model.load_state_dict(best_state); model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "_" + series_name if series_name else ""
    torch.save(model.state_dict(), os.path.join(args.output_dir, "anomaly_{}{}_seed{}.pt".format(args.dataset, suffix, args.seed)))
    # A one-class threshold is selected from held-out normal validation scores only.
    threshold=float(np.quantile(best_vals, 1.0-args.contamination))
    scores = _aggregate_reconstruction_scores(model, test_dl, test_ds, device)
    pred=(scores>threshold).astype(np.int64)
    p,r,f1,_=precision_recall_fscore_support(labels[:len(pred)],pred,average="binary",zero_division=0)
    print("threshold",threshold,"precision",p,"recall",r,"f1",f1)
    return {"precision":float(p),"recall":float(r),"f1":float(f1),"threshold":threshold,
            "n_test":len(pred),"series_name":series_name,"best_val_mse":best,
            "best_epoch":best_epoch,"epochs_run":epoch+1}
