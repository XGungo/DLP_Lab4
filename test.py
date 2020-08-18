# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import combinations
# m = nn.LogSoftmax(dim = 1)
# a = nn.NLLLoss()
# b = torch.Tensor([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
# c = torch.Tensor([1,2,3])
# print(b.shape)
# print(m(b))
# print(c.shape)
# loss = a(m(b),c)
# print(loss)

m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
# input is of size N x C = 3 x 5
input = torch.Tensor([[1,6,8,9,3],[2,1,3,4,5],[3,2,5,2,1]])
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 4])
output = loss(m(input), target)
print(output)
output.backward()
