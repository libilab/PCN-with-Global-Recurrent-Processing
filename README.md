# PCN_v1
This repository contains the code for PCN v1 introduced in the following paper:

[Deep Predictive Coding Network for Object Recognition](https://arxiv.org/abs/1802.04762) (ICML2018)

Haiguang Wen, Kuan Han, Junxing Shi, Yizhen Zhang, Eugenio Culurciello, Zhongming Liu

The code is built on Pytorch

## Introduction

Deep predictive coding network (PCN) v1 is a bi-directional and recurrent neural net, based on the predictive coding theory in neuroscience. It has feedforward, feedback, and recurrent connections. Feedback connections from a higher layer carry the prediction of its lower-layer representation; feedforward connections carry the prediction errors to its higher-layer. Given image input, PCN runs recursive cycles of bottom-up and top-down computation to update its internal representations and reduce the difference between bottom-up input and top-down prediction at every layer. After multiple cycles of recursive updating, the representation is used for image classification. With benchmark data (CIFAR-10/100, SVHN, and MNIST), PCN was found to always outperform its feedforward-only counterpart: a model without any mechanism for recurrent dynamics. Its performance tended to improve given more cycles of computation over time. In short, PCN reuses a single architecture to recursively run bottom-up and top-down processes. As a dynamical system, PCN can be unfolded to a feedforward model that becomes deeper and deeper over time, while refining it representation towards more accurate and definitive object recognition.

![Image of pcav1](https://github.com/libilab/PCN_v1/blob/master/figures/Figure1.png)
(a) An example PCN with 9 layers and its CNN counterpart (or the plain model).

(b) Two-layer substructure of PCN. Feedback (blue), feedforward (green), and recurrent (black) connections convey the top-down prediction, the bottom-up prediction error, and the past information, respectively.

(c) The dynamic process in the PCN iteratively updates and refines the representation of visual input over time. PCN outputs the probability over candidate categories for object recognition. 
