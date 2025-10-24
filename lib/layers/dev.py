# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:24:46 2022

@author: naouf
"""

import torch
import numpy


vec = torch.tensor([[1, 2], [3, 4]])
n_point = 20
eps = 1e-6
new_shape = list(vec.shape)
new_shape.append(n_point)

res = torch.zeros(new_shape)
vec_aug = torch.cat((torch.zeros(vec.shape[0],1), vec), 1)

for i in range(res.shape[0]):
    res[i,:] = torch.linspace(vec_aug[i],vec_aug[i+1]-eps, n_point)
    

def test(mat):
    
    n_point = 20
    eps = 1e-6
    init_shape = mat.shape
    
    flat_mat = torch.flatten(mat)
    
    new_shape = list(flat_mat.shape)
    new_shape.append(n_point)
    res = torch.zeros(new_shape)
    
    s0 = set(range(0, res.shape[0], init_shape[1]))
    sT = set(range(res.shape[0]))
    diff = sT - s0
    

    for i in s0:
        res[i,:] = torch.linspace(0, flat_mat[i], n_point)
    
    for i in diff:
        res[i,:] = torch.linspace(flat_mat[i-1]+eps, flat_mat[i], n_point)
    
    final_shape = list(init_shape)
    final_shape.append(n_point)
    
    res = torch.reshape(res, final_shape)
    return res


vec = torch.tensor([1, 2, 3, 4])
mat = torch.tensor([[1, 2], [3, 4], [5,6]])

test(mat)
