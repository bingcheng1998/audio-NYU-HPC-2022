#!/bin/env python

import sys
print(sys.path)
print('111111')

import torch

print(torch.__file__)
print(torch.__version__)

# How many GPUs are there?
print(torch.cuda.device_count())

# Is PyTorch using a GPU?
print(torch.cuda.is_available())

# Get the name of the current GPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))