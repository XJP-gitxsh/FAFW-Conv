"""Matched ETTh2 and ETTm2 forecast grid using the ETTh1/ETTm1 protocol."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "forecast_grid_ett2_20261028"
LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
CHECKPOINTS = Path("forecast_grid_ett2_20261028") / "checkpoints"
PYTHON = sys.executable

for dataset, freq in (("ETTh2", "h"), ("ETTm2", "t")):
    for horizon in (96, 192, 336, 720):
        for mode in ("difference", "none"):
            for seed in (2021, 2022, 2023):
                name = f"{dataset}_H{horizon}_{mode}_seed{seed}"
                log = LOGS / f"{name}.log"
                if log.exists() and re.search(r"mse:[0-9.eE+-]+,\s*mae:[0-9.eE+-]+",
                                              log.read_text(encoding="utf-8", errors="replace")):
                    print(f"SKIP completed {name}", flush=True)
                    continue
                cmd = [
                    PYTHON, "-u", "scripts/run.py", "--is_training", "1", "--model_id", "g4",
                    "--model", "ModernTCN", "--root_path", "./all_six_datasets/dataset/",
                    "--data_path", f"{dataset}.csv", "--data", dataset, "--features", "M",
                    "--freq", freq, "--seq_len", "336", "--label_len", "48", "--pred_len", str(horizon),
                    "--random_seed", str(seed), "--expand", "2", "--d_state", "16", "--d_conv", "4",
                    "--ffn_ratio", "2", "--patch_size", "8", "--patch_stride", "4",
                    "--num_blocks", "1", "1", "1", "1", "--large_size", "51", "51", "51", "51",
                    "--small_size", "5", "5", "5", "5", "--dims", "64", "64", "64", "64",
                    "--dw_dims", "64", "64", "64", "64", "--small_kernel_merged", "True",
                    "--use_multi_scale", "False", "--fractional_mode", mode, "--use_hhconv", "True",
                    "--use_wavelet", "True", "--decomposition", "1", "--enc_in", "7", "--dec_in", "7",
                    "--c_out", "7", "--dropout", "0.05", "--fc_dropout", "0.05", "--head_dropout", "0.0",
                    "--revin", "1", "--affine", "0", "--train_epochs", "100", "--batch_size", "256",
                    "--patience", "15", "--learning_rate", "0.0001", "--lradj", "type3", "--loss", "mse",
                    "--checkpoints", str(CHECKPOINTS), "--des", f"{dataset[-1]}h{horizon}{mode[0]}s{seed}",
                    "--num_workers", "0", "--itr", "1", "--use_gpu", "True", "--gpu", "0",
                ]
                print(f"START {name}", flush=True)
                with log.open("w", encoding="utf-8") as f:
                    proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
                if proc.returncode:
                    raise SystemExit(f"Run failed ({proc.returncode}): {name}; see {log}")
                print(f"DONE {name}", flush=True)
