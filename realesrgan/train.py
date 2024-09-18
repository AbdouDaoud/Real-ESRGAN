# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':

    logger = TensorBoardLogger("tb_logs", name = "my_realesrgan")
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
