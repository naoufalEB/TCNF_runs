# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:43:49 2022

@author: naouf
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .odefunc import NONLINEARITIES

#f_activ = NONLINEARITIES['softplus']
f_activ = nn.Softplus()

class time_change_func_id(nn.Module):
    def __init__(self):
        super(time_change_func_id, self).__init__()
        
        self.param_inutil =   nn.Parameter(torch.tensor(0.0))
        
    def forward(self, t):
        
        return t


    
class time_change_func_positif_old(nn.Module):
    def __init__(self, time_change_dims):
        
        super(time_change_func_positif, self).__init__()
        
        hidden_dims = list(tuple(map(int, time_change_dims.split(","))))
        hidden_dims.insert(0, 1)
        hidden_dims.append(1)
        
        layers = []
        
        for i in range(len(hidden_dims)-1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            with torch.no_grad():
                w = torch.ones(hidden_dims[i+1], hidden_dims[i])/10
                b = torch.ones(hidden_dims[i+1])/10
                layer.weight.copy_(w)
                layer.bias.copy_(b)
                
            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
        self.activation_fn1 = nn.Tanh()
        self.activation_fn2 = nn.Softplus()
        
    
  
    def forward(self, t):
        
        t1 = t[:,None]
        for l, layer in enumerate(self.layers):
            t1 = layer(t1)
            
            if l < len(self.layers) - 1 :
                t1 = self.activation_fn1(t1)
                #m = torch.mean(t1)
                #s = torch.std(t1)
                #t1 = (t1 - m)/s
                
            else:
                t1 = self.activation_fn2(t1)
            
        
        
        t1 = torch.reshape(t1, t.shape)
        
        return t1
    
    
class time_change_func_exp(nn.Module):
    def __init__(self):
        super(time_change_func_exp, self).__init__()
        
        self.theta = nn.Parameter(torch.tensor([1.3]))
                
    def forward(self, t, reverse = False):
                
        t1 = 2*torch.abs(self.theta)*t
        phi_t = torch.exp(t1) - 1
                
        return phi_t
    
    def show_plot_(self,t, epoch):
        
        test = torch.linspace(0,1,100).to(t)
        y = self.forward(test)
        y_np = y.detach().cpu().numpy()
        
        plt.plot(test.detach().cpu().numpy(), y_np)
        plt.title("Time change func : " + str(epoch))
        plt.show()
        
        
class time_change_func_integral(nn.Module):
    def __init__(self, input_dim = 1):
        super(time_change_func_integral, self).__init__()
        
        
        self.integrand = integrand(input_dim)
                    
    def forward(self, t):
                
        #t = torch.squeeze(t, dim = 2)
        
        g_t = torch.zeros_like(t)
        dg_t = self.integrand(t)
        #dg_t = torch.reshape(self.integrand(t[:,:,None]), t.shape)
        
        for j in range(g_t.shape[1]):
            g_t[:, j, 0] = torch.trapezoid(dg_t[:,0:(j+1),0], t[:,0:(j+1),0], dim = 1)
        
        
        
        return g_t
    
    def func_plot(self, start = 0, end = 10):
        
        x = torch.linspace(0, start, end).cuda()
        y = self.forward(x[None,:,None])
        y_np = y.detach().cpu().numpy()
          
        plt.figure(figsize=(7, 7))
        plt.plot(y_np[0,:])
        plt.title("Time change function")
        plt.show()
        
        
    def integrand_plot(self, start = 0, end =10):
        
        x1 = torch.linspace(0, start, end).cuda()
        y1 = self.integrand(x1[:,None])
        y1_np = y1.detach().cpu().numpy()
        
        plt.figure(figsize=(7, 7))
        plt.plot(y1_np[:,0])
        plt.title("Integrand - time change")
        plt.show()
        
class integrand(nn.Module):
    def __init__(self, input_dim = 1):
        super(integrand, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.activation_fn = nn.Softplus()
        
    def forward(self, t):
                
        t1 = self.fc1(t)
        t1 = self.activation_fn(t1)
        
        t1 = self.fc2(t1)
        t1 = self.activation_fn(t1)
        
        t1 = self.fc3(t1)
        t1 = self.activation_fn(t1)
        
        t1 = self.fc4(t1)
        t1 = self.activation_fn(t1)
        
        t1 = self.fc5(t1)
        t1 = self.activation_fn(t1)+1
        
        return t1

class time_change_func_convex(nn.Module):
    def __init__(self, input_dim = 1):
        super(time_change_func_convex, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.activation_fn = nn.Softplus()
        
    def conv(self, t):
                
        t1 = self.fc1(t)
        t1 = self.activation_fn(t1)
        
        t1 = self.fc2(t1)
        t1 = self.activation_fn(t1)
        
        t1 = t1**2
        
        return t1
    
    def forward(self, t):
        
        t2 = t.clone().detach().requires_grad_(True).to(t)
        df = self.conv(t2)
        
        df.sum().backward()
        
        df_t = t2.grad
        
        if torch.min(df_t)<0:
            # on shift la courbe si negative, et on ajoute le premier saut pour éviter le 0
            temp_diff = torch.diff(df_t, dim = 1)
            
            df_t = df_t - torch.min(df_t) + torch.min(temp_diff[temp_diff>0])
            
            
        return df_t
    
    
    
def which_is(mytensor, tolook):
    res = ((mytensor == tolook).nonzero(as_tuple = True)[0])
    
    return res



class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.ones((in_dim, out_dim))/10)
        self.bias = nn.Parameter(torch.zeros((out_dim,)))
        
    def forward(self, x):
        return torch.matmul(x, torch.abs(self.weight)) + self.bias
    

class time_change_func_positif(nn.Module):
    def __init__(self, time_change_dims):
        
        super(time_change_func_positif, self).__init__()
        
        hidden_dims = list(tuple(map(int, time_change_dims.split(","))))
        hidden_dims.insert(0, 1)
        hidden_dims.append(1)
        
        layers = []
        
        for i in range(len(hidden_dims)-1):
            layer = PosLinear(hidden_dims[i], hidden_dims[i+1])
                            
            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
        self.activation_fn1 = nn.Tanh()
        self.activation_fn2 = nn.Softplus()
            
  
    def forward(self, t):
        
        t1 = t[:,None]
        for l, layer in enumerate(self.layers):
            t1 = layer(t1)
            
            if l < len(self.layers) - 1 :
                t1 = self.activation_fn1(t1)
                
            else:
                t1 = self.activation_fn1(t1)
            
        
        
        t1 = torch.reshape(t1, t.shape)
        t1 = torch.exp(t1)
        
        return t1
    
    def show_plot_(self,t, epoch):
        
        test = torch.linspace(0,1,100).to(t)
        y = self.forward(test)
        y_np = y.detach().cpu().numpy()
        
        plt.plot(test.detach().cpu().numpy(), y_np)
        plt.title("Time change func : " + str(epoch))
        plt.show()
        

def primitive_Relu(t):
    res = 0.5*(t**2)*(t>=0) + 0*(t<0)
    return res

def positif_phi_t_old(t, init_shape, eps = 0.001):
    
    if (init_shape is None) or (len(init_shape) == 1):
        t1 = t[:,None]
        min_t1,_ = torch.min(t1, 0)
    else:
        t1 = torch.reshape(t, init_shape)
        min_t1,_ = torch.min(t1, 1)
    
    correction = ( eps - min_t1 )*( min_t1 <= 0 )
    
    res = t1 + correction[:,None]
    
    res = torch.reshape(res, t.shape)
    
    return res

def positif_phi_t(t, eps = 0.001):
    
    min_t,_ = torch.min(t, 0)
    
    correction = ( eps - min_t )*( min_t <= 0 )
    
    res = t + correction
        
    return res

class time_change_func_M_MGN(nn.Module):
    def __init__(self, K, dim_input = 1):
        super(time_change_func_M_MGN, self).__init__()
        
        dim_input
        vec_b = []
        vec_W = []
              
        
        for i in range(K):
            b_temp = nn.Parameter(torch.rand(dim_input, 1))
            vec_b.append(b_temp)
            
            w_temp = nn.Parameter(torch.rand(dim_input, dim_input))
            vec_W.append(w_temp)
            
        self.b_array = nn.ParameterList(vec_b)
        self.W_array = nn.ParameterList(vec_W)
        
        self.V = nn.Parameter(torch.rand(dim_input, dim_input))
        
        self.a = nn.Parameter(torch.rand(dim_input,1))
        
        self.activation_fn1 = nn.ReLU() 

    def forward(self, t):
        
        t1 = t[None,:]
        K = len(self.b_array)
        
        NN_module_sum = 0
        for i in range(K):
            temp_z_i = torch.matmul(self.W_array[i], t1) + self.b_array[i]
            
            NN_module_i = primitive_Relu(temp_z_i)*(torch.matmul(torch.t(self.W_array[i]), self.activation_fn1(temp_z_i)))
            
            NN_module_sum = NN_module_sum + NN_module_i
        
        cross_prod = torch.matmul( torch.matmul(torch.t(self.V), self.V), t1)
        
        MGN = torch.reshape(self.a + cross_prod + NN_module_sum, t.shape)
        
        MGN = positif_phi_t(MGN)
        
        return MGN
    
    def show_plot_(self,t, epoch):
        
        test = torch.linspace(0,1,100).to(t)
        #test2 = test.expand(5, 100)
        #test3 = torch.flatten(test2)
        
        # y = self.forward(test3, test2.shape)
        # y = torch.reshape(y, test2.shape)
        # y_np = y[0,:].detach().cpu().numpy()
        y = self.forward(test)
        y_np = y.detach().cpu().numpy()
        
        plt.plot(test.detach().cpu().numpy(), y_np)
        plt.title("Time change func at epoch : " + str(epoch))
        plt.show()
        
class time_change_func_sqrt(nn.Module):
    def __init__(self, input_dim = 1):
        
        super(time_change_func_sqrt, self).__init__()
        
        self.theta = nn.Parameter(torch.tensor([1]))
                
    def forward(self, t, reverse = False):
                
        t1 = 2*torch.abs(self.theta)*t
        phi_t = torch.exp(t1) - 1
                
        return phi_t
    
    def show_plot_(self,t, epoch):
        
        test = torch.linspace(0,1,100).to(t)
        y = self.forward(test)
        y_np = y.detach().cpu().numpy()
        
        plt.plot(test.detach().cpu().numpy(), y_np)
        plt.title("Time change func : " + str(epoch))
        plt.show()