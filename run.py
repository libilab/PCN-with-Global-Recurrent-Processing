#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train import train_prednet
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--cls', default=6, type=int, help='number of circles')
parser.add_argument('--model', default='PredNetTied', help= 'models to train')
parser.add_argument('--gpunum', default=2, type=int, help='number of gpu used to train the model')
parser.add_argument('--lr', default=0.01, type=float, help='number of gpu used to train the model')
args = parser.parse_args()
 
train_prednet(model=args.model, cls=5, gpunum=args.gpunum, lr=args.lr)
train_prednet(model=args.model, cls=6, gpunum=args.gpunum, lr=args.lr)
train_prednet(model=args.model, cls=1, gpunum=args.gpunum, lr=args.lr)
train_prednet(model=args.model, cls=2, gpunum=args.gpunum, lr=args.lr)

