import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Classification and reconstruction-anomaly entry point.

Examples:
  python scripts/run_multitask.py --task classification --dataset UWaveGestureLibrary --data_root ./UEA
  python scripts/run_multitask.py --task anomaly --dataset SMD --data_root ./anomaly_data/SMD --window 100
"""
import argparse
import json
import os
import random
import platform
import numpy as np
import torch
from utils.str2bool import str2bool

from exp.exp_multitask import train_classification, train_anomaly


def main():
    p=argparse.ArgumentParser("FAFW-Conv classification/anomaly runner")
    p.add_argument("--task",choices=["classification","anomaly"],required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--data_root",required=True)
    p.add_argument("--series_name",default=None,help="SMD machine name, e.g. machine-1-1")
    p.add_argument("--output_dir",default="./multitask_results")
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--patience",type=int,default=15,
                   help="Early-stopping patience on validation loss")
    p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--val_fraction",type=float,default=0.1)
    p.add_argument("--window",type=int,default=100)
    p.add_argument("--stride",type=int,default=None)
    p.add_argument("--contamination",type=float,default=0.01)
    p.add_argument("--dims",type=int,default=64)
    p.add_argument("--ffn_ratio",type=int,default=2)
    p.add_argument("--large_size",type=int,default=31)
    p.add_argument("--patch_size",type=int,default=8)
    p.add_argument("--patch_stride",type=int,default=4)
    p.add_argument("--dropout",type=float,default=0.1)
    p.add_argument("--fractional_mode",choices=["difference","sum","none"],default="difference",
                   help="difference=(1-L)^alpha; sum=(1-L)^(-alpha); none=ordinary convolution control")
    p.add_argument("--use_hhconv",type=str2bool,default=True,
                   help="process wavelet detail subbands with HHConv2D; false leaves them unchanged")
    p.add_argument("--use_wavelet",type=str2bool,default=True,
                   help="enable DWT/IDWT blocks; false uses the corresponding non-wavelet ModernTCN path")
    args=p.parse_args()
    if args.epochs < 1 or args.patience < 1:
        p.error("--epochs and --patience must be positive")
    if not 0.0 < args.val_fraction < 0.5:
        p.error("--val_fraction must be between 0 and 0.5")
    if not 0.0 < args.contamination < 0.5:
        p.error("--contamination must be between 0 and 0.5")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device,"task",args.task,"dataset",args.dataset)
    result=(train_classification(args,device) if args.task=="classification"
            else train_anomaly(args,device))
    result.update({"task":args.task,"dataset":args.dataset,"seed":args.seed})
    result["config"] = vars(args).copy()
    result["environment"] = {"python":platform.python_version(),
                              "torch":torch.__version__,
                              "numpy":np.__version__,
                              "device":str(device)}
    os.makedirs(args.output_dir,exist_ok=True)
    suffix=("_"+args.series_name) if args.series_name else ""
    path=os.path.join(args.output_dir,"{}_{}{}_seed{}.json".format(args.task,args.dataset,suffix,args.seed))
    with open(path,"w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print("saved",path)


if __name__=="__main__": main()
