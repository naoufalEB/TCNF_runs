# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:43:49 2022

@author: naouf
"""


import torch
import torch.nn as nn

from .odefunc import NONLINEARITIES

#f_activ = NONLINEARITIES['softplus']
f_activ = nn.Softplus()


class ExplicitNFModel(nn.Module):
    def __init__(self, dim_input = 1):
        super(ExplicitNFModel, self).__init__()
        
        self.dim_input = dim_input
        self.theta = nn.Parameter(torch.randn(dim_input, dim_input))
        self.x0 = nn.Parameter(torch.randn(dim_input, 1))
        

    def forward(self, x, logpx = None, reverse = False):

        t = x[:,-1:]
        x_vec = x[:,0:-1]

        x0_arr = self.x0.view(1, self.x0.shape[0], 1).expand(x_vec.shape[0], -1, -1)
        exp_neg_theta_t = torch.matrix_exp(-t.unsqueeze(-1)*self.theta)
        exp_theta_t = torch.matrix_exp(t.unsqueeze(-1)*self.theta)
        
        logdet = torch.log(torch.abs(torch.prod(torch.diagonal(exp_neg_theta_t, dim1=-2, dim2=-1), dim = 1)))

        if not reverse:
            x_transform = torch.matmul(exp_theta_t, (x_vec.unsqueeze(-1) - torch.matmul(exp_neg_theta_t, x0_arr)))
        else:
            x_transform = torch.matmul(exp_neg_theta_t, x0_arr + x_vec.unsqueeze(-1))
            
        x_transform = x_transform.squeeze(-1)
        x = torch.cat((x_transform, t), dim = -1)
        
        if logpx is None:
            return x
        else:
            if not reverse:
                logpx = logpx - logdet.unsqueeze(-1)
            else:
                logpx = logpx + logdet.unsqueeze(-1)

            return x, logpx

    
    