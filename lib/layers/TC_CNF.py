# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 22:06:03 2022

@author: naouf
"""

import torch.nn as nn
import torch

class TimeChangedCNF(nn.Module):
    

    def __init__(self, sequential_flow, time_change_func):
        
        super(TimeChangedCNF, self).__init__()
        
        self.sequential_flow = sequential_flow
        self.time_change_func = time_change_func

    def forward(self, x, logpx=None, reverse=False, inds=None):#, dim_x = 1):
        
        """
        reverse == True : x = aug_model(base_value, reverse = true)
        reverse == False : base_value = aug_model(x, reverse = false)
        w_t , flow_logdet = aug_model(X_t, 0, reverse = False)
        logdet_final = normal(w_t) - flow_logdet
        X_t , flow_logdet = aug_model(w_t, , reverse = True)
        """
                
        #x_hat = x[:,:dim_x]
        #t_hat = x[:,dim_x:]
        
        #phi_t = self.time_change_func(t_hat)
        
        #x = torch.cat([x_hat, phi_t], dim = 1 )
        
        x, logpx = self.sequential_flow(x, logpx, reverse, inds)
        
        #regul_loss = torch.sum(torch.square(phi_t - x[:,dim_x:]))
        regul_loss = torch.tensor(0.).to(x)
            
        return x, logpx, regul_loss

class TimeChangedExplicitNF(nn.Module):
    

    def __init__(self, explicit_NF, time_change_func):
        
        super(TimeChangedExplicitNF, self).__init__()
        
        self.explicit_NF = explicit_NF
        self.time_change_func = time_change_func

    def forward(self, x, logpx=None, reverse=False, inds=None):
        
        """
        reverse == True : x = aug_model(base_value, reverse = true)
        reverse == False : base_value = aug_model(x, reverse = false)
        w_t , flow_logdet = aug_model(X_t, 0, reverse = False)
        logdet_final = normal(w_t) - flow_logdet
        X_t , flow_logdet = aug_model(w_t, , reverse = True)
        """
                
        x, logpx = self.explicit_NF(x, logpx, reverse)
        
        regul_loss = torch.tensor(0.).to(x)
            
        return x, logpx, regul_loss