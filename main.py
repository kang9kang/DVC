import torch
import argparse

from dataset import VideoDataset
from model import VideoCompressor
from loss import CompressionLoss
from trainer import Trainer

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    parser = argparse.ArgumentParser(description="DVC")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--lambda_", type=float, default=1024, help="lambda")
    parser.add_argument(
        "--base_lr", type=float, default=1e-4, help="base learning rate"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.1, help="learning rate decay"
    )
    parser.add_argument("--lr_decay_epoch", type=int, default=10, help="lr decay epoch")
    args = parser.parse_args()

    trainer = Trainer(
        VideoDataset(), VideoCompressor(), CompressionLoss(0.1, args.lambda_), args
    )
    trainer.train(epochs=args.epochs)
