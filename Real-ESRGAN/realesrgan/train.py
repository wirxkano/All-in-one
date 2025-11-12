# flake8: noqa\
import sys
from pathlib import Path
import os.path as osp

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from BasicSR.basicsr.train import train_pipeline

if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
