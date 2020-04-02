# PCN with Global Recurrent Processing
This repository contains the code for PCN with global recurrent processing introduced in the following paper:

[Deep Predictive Coding Network for Object Recognition](https://arxiv.org/abs/1802.04762) (ICML2018)

Haiguang Wen, Kuan Han, Junxing Shi, Yizhen Zhang, Eugenio Culurciello, Zhongming Liu

The code is built on Pytorch

## Introduction

Deep predictive coding network (PCN) with global recurrent processing is a bi-directional and recurrent neural net, based on the predictive coding theory in neuroscience. It has feedforward, feedback, and recurrent connections. Feedback connections from a higher layer carry the prediction of its lower-layer representation; feedforward connections carry the prediction errors to its higher-layer. Given image input, PCN runs recursive cycles of bottom-up and top-down computation to update its internal representations and reduce the difference between bottom-up input and top-down prediction at every layer. PCN was found to always outperform its feedforward-only counterpart: a model without any mechanism for recurrent dynamics. Its performance tended to improve given more cycles of computation over time. 

![Image of pcav1](https://github.com/libilab/PCN_v1/blob/master/figures/Figure1.png)
(a) An example PCN with 9 layers and its CNN counterpart (or the plain model).

(b) Two-layer substructure of PCN. Feedback (blue), feedforward (green), and recurrent (black) connections convey the top-down prediction, the bottom-up prediction error, and the past information, respectively.

(c) The dynamic process in the PCN iteratively updates and refines the representation of visual input over time. PCN outputs the probability over candidate categories for object recognition. 

## Usage
Install Torch and required dependencies like cuDNN. See the instructions [here](https://github.com/pytorch/pytorch) for a step-by-step guide.

Clone this repo: https://github.com/libilab/PCN_v1.git

As an example, the following command trains a PCN with circles = 6 on CIFAR-100 using 4 GPU:

```bash
python run.py --circles 6 --model 'PredNet' --gpunum 4
```

## Results on CIFAR

![Image of pcav1](https://github.com/libilab/PCN_v1/blob/master/figures/fig_3.png)

Testing accuracies of PCNs with different time steps.

## Updates
10/17/2018:

   (1) readme file.

02/12/2020:

   (1) removed group normalization to match the implementation in the paper.
