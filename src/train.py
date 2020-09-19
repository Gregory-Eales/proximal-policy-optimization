"""
This file defines the core research contribution   
"""
import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pytorch_lightning as pl

# trains the specific algorithm
from algorithm.trainer import AlgorithmTrainer

# lightning wrapper for orchestrating the algorithm
from algorithm.algorithm import Algorithm

pl.seed_everything(123)

def train():
    pass


if __name__ == '__main__':
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--learning_rate', default=0.02, type=float)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # parse params
    args = parser.parse_args()

    # init module
    model = CoolSystem(hparams=args)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_data)


