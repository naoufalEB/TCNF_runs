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
        
                
    def forward(self, t):
                
        phi_t = t
                
        return phi_t
    
    def show_plot_(self,t, epoch, args):
        
        if args.effective_shape == 1:
            test = torch.linspace(0,1,100).to(t)
        else:
            test = torch.linspace(0,1,100).unsqueeze(1)
            test = test.expand(-1, args.effective_shape).to(t)
            
        y = self.forward(test)
        y_np = y.detach().cpu().numpy()
        
        func_type = args.time_change
        title = "phi_t | func : " + str(func_type) + " | epoch : " + str(epoch)
        plt.plot(test.detach().cpu().numpy(), y_np)
        plt.title(title)
        plt.show()
        
    
class time_change_func_exp(nn.Module):
    def __init__(self, dim_input = 1):
        super(time_change_func_exp, self).__init__()
        
        self.theta = nn.Parameter(torch.randn(dim_input))
               
    def forward(self, t):
                
        #phi_t = torch.exp(2*(self.theta**2)*t)-1        
        
        # theoretical time change
        phi_t = torch.exp(2*torch.abs(self.theta)*t)*(t - 0.5/torch.abs(self.theta)) + 0.5/torch.abs(self.theta)
        phi_t = phi_t/torch.abs(self.theta)

        return phi_t
    
    def show_plot_(self,t, epoch, args):
        
        if args.effective_shape == 1:
            test = torch.linspace(0,1,100).to(t)
        else:
            test = torch.linspace(0,1,100).unsqueeze(1)
            test = test.expand(-1, args.effective_shape).to(t)
            
        y = self.forward(test)
        y_np = y.detach().cpu().numpy()
        
        func_type = args.time_change
        title = "phi_t | func : " + str(func_type) + " | epoch : " + str(epoch)
        plt.plot(test.detach().cpu().numpy(), y_np)
        plt.title(title)
        plt.show()
        
    
def which_is(mytensor, tolook):
    res = ((mytensor == tolook).nonzero(as_tuple = True)[0])
    
    return res


def primitive_Relu(t):
    s_k = 0.5*(t**2)*(t>=0) + 0*(t<0)
    res = torch.sum(s_k, dim = 0)
    return res

def positif_phi_t(t, eps = 0.02):
    
    
    min_t,_ = torch.min(t.min(), 0)
    
    
    correction = ( eps - min_t )*( min_t <= 0 )
    
    res = t + correction
        
    return res

class time_change_func_M_MGN(nn.Module):
    def __init__(self, K, N, dim_input = 1):
        super(time_change_func_M_MGN, self).__init__()
        
        
        vec_b = []
        vec_W = []
              
        
        for i in range(K):
            b_temp = nn.Parameter(torch.rand(N, 1))
            vec_b.append(b_temp)
            
            w_temp = nn.Parameter(torch.rand(N, dim_input))
            vec_W.append(w_temp)
            
        self.b_array = nn.ParameterList(vec_b)
        self.W_array = nn.ParameterList(vec_W)
        
        #self.V = nn.Parameter(torch.rand(N, dim_input))
        self.V = nn.Parameter(torch.eye(N, dim_input))
        
        #self.a = nn.Parameter(torch.rand(dim_input,1))
        self.a = nn.Parameter(torch.zeros(dim_input,1))
        
        self.activation_fn1 = nn.ReLU()
        

    def forward(self, t):
        
        t1 = t.t()
        K = len(self.b_array)
        
        NN_module_sum = 0
        for i in range(K):
            
            # try:
            #     temp_z_i = torch.matmul(self.W_array[i], t1) + self.b_array[i]
            # except ValueError:
            #     test = 1
            temp_z_i = torch.matmul(self.W_array[i], t1) + self.b_array[i]
            
            NN_module_i = primitive_Relu(temp_z_i)*(torch.matmul(torch.t(self.W_array[i]), self.activation_fn1(temp_z_i)))
            
            NN_module_sum = NN_module_sum + NN_module_i
        
        cross_prod = torch.matmul( torch.matmul(torch.t(self.V), self.V), t1)
        
        #MGN = torch.reshape(self.a + cross_prod + NN_module_sum, t.shape)
        MGN = (self.a + cross_prod + NN_module_sum).t()
        
        MGN = positif_phi_t(MGN)
        
                
        return MGN
    
    def show_plot_(self,t, epoch, args):
        
        if args.effective_shape == 1:
            test = torch.linspace(0,1,100).to(t)
        else:
            test = torch.linspace(0,1,100).unsqueeze(1)
            test = test.expand(-1, args.effective_shape).to(t)
            
        y = self.forward(test)
        y_np = y.detach().cpu().numpy()
        
        func_type = args.time_change
        title = "phi_t | func : " + str(func_type) + " | epoch : " + str(epoch)
        plt.plot(test.detach().cpu().numpy(), y_np)
        plt.title(title)
        plt.show()



class time_change_func_M_MGN_chain(nn.Module):
    def __init__(self, K_chain, N_chain):
        super(time_change_func_M_MGN_chain, self).__init__()
        
        K_dims = tuple(map(int, K_chain.split(",")))
        N_dims = tuple(map(int, N_chain.split(",")))
        
        if len(K_dims) != len(N_dims):
            raise ValueError("different dimensions size for M_MGN chain")
        
        self.dim_chain = len(K_dims)
        self.MGN_chain = nn.ModuleList([
            time_change_func_M_MGN(K_dims[i], N_dims[i]) for i in range(self.dim_chain)
            ])
        
    def forward(self, t):
        ## used for multidimensional case where t is expanded to match X_input shape. 
        ## only t[:,0] is needed for the forward pass
        
        results = torch.stack([mgn(t) for mgn in self.MGN_chain], dim=1).squeeze(-1)
            
        return results
        
        
    
# class time_change_correl_matrix_rho(nn.Module):
#     def __init__(self):
#         super(time_change_correl_matrix_rho, self).__init__()

#         self.a = nn.Parameter(torch.tensor([.1]))
#         self.b = nn.Parameter(torch.tensor([.1]))
        
#         self.correl_mat = torch.tensor([[1., self.a],[self.b, 1.]])
    
#     def forward(self,t):
        
#         correlated_time_change = torch.matmul(torch.matmul(self.correl_mat, t), torch.t(self.correl_mat))
#         return correlated_time_change
        
class time_change_func_exp_poly(nn.Module):
    def __init__(self):
        super(time_change_func_exp_poly, self).__init__()
        
        self.theta = nn.Parameter(torch.tensor([1.3]))
        #self.theta = torch.tensor([1.3]).cuda()
        
    def forward(self, t, reverse = False):
                
        t1 = 2*torch.abs(self.theta)*t
        phi_t = torch.exp(t1)*(t1 -1) + 1
                
        return phi_t
    
    def show_plot_(self,t, epoch, func_type):
        
        test = torch.linspace(0,1,100).to(t)
        y = self.forward(test)
        y_np = y.detach().cpu().numpy()
        
        plt.plot(test.detach().cpu().numpy(), y_np)
        plt.title("phi_t | func : " + str(func_type) + " | epoch : " + str(epoch))
        plt.show()

class time_change_func_exp_poly_simple(nn.Module):
    def __init__(self):
        super(time_change_func_exp_poly_simple, self).__init__()
        
        self.theta = nn.Parameter(torch.randn(1))
                
    def forward(self, t, reverse = False):
                
        t1 = 2*torch.abs(self.theta)*t
        phi_t = t1 -1 + torch.exp(-t1)
                
        return phi_t
    
    def show_plot_(self,t, epoch, func_type):
        
        test = torch.linspace(0,1,100).to(t)
        y = self.forward(test)
        y_np = y.detach().cpu().numpy()
        
        plt.plot(test.detach().cpu().numpy(), y_np)
        plt.title("phi_t | func : " + str(func_type) + " | epoch : " + str(epoch))
        plt.show()
        

class time_change_func_general_exp(nn.Module):
    def __init__(self, degre):
        super(time_change_func_general_exp, self).__init__()
        
        self.theta = nn.Parameter(torch.tensor([1.]))
        
        self.vec_coeff = nn.Parameter(torch.rand(degre + 1))
        
        #self.powers = torch.linspace(0, degre, degre + 1) # les degres du polynomes
        self.degre = degre
        
    def forward(self, t):
        
        les_puissances = torch.linspace(0, self.degre, self.degre + 1).to(t)
        t_powers = t[:,None]**les_puissances
        
        abs_last_coeff = torch.abs(self.vec_coeff[-1])
        vec_coeff_monotonic = torch.cat((self.vec_coeff[:-1], abs_last_coeff.reshape(1))).to(t)
        
        
        exp_part = torch.exp(torch.abs(self.theta)*t)
        poly_part = t_powers*vec_coeff_monotonic
        
        phi_t = exp_part*torch.sum(poly_part, dim = 1)
                
        return phi_t
    
    def show_plot_(self,t, epoch, func_type):
        
        test = torch.linspace(0,1,100).to(t)
        y = self.forward(test)
        y_np = y.detach().cpu().numpy()
        
        plt.plot(test.detach().cpu().numpy(), y_np)
        plt.title("phi_t | func : " + str(func_type) + " | epoch : " + str(epoch))
        plt.show()    
        
        
# class time_change_func_M_MGN_old(nn.Module):
#     def __init__(self, K, dim_input = 1):
#         super(time_change_func_M_MGN, self).__init__()
        
#         dim_input
#         vec_b = []
#         vec_W = []
              
        
#         for i in range(K):
#             b_temp = nn.Parameter(torch.rand(dim_input, 1))
#             vec_b.append(b_temp)
            
#             w_temp = nn.Parameter(torch.rand(dim_input, dim_input))
#             vec_W.append(w_temp)
            
#         self.b_array = nn.ParameterList(vec_b)
#         self.W_array = nn.ParameterList(vec_W)
        
#         self.V = nn.Parameter(torch.rand(dim_input, dim_input))
        
#         self.a = nn.Parameter(torch.rand(dim_input,1))
        
#         self.activation_fn1 = nn.ReLU() 

#     def forward(self, t):
        
#         t1 = t[None,:]
#         K = len(self.b_array)
        
#         NN_module_sum = 0
#         for i in range(K):
#             temp_z_i = torch.matmul(self.W_array[i], t1) + self.b_array[i]
            
#             NN_module_i = primitive_Relu(temp_z_i)*(torch.matmul(torch.t(self.W_array[i]), self.activation_fn1(temp_z_i)))
            
#             NN_module_sum = NN_module_sum + NN_module_i
        
#         cross_prod = torch.matmul( torch.matmul(torch.t(self.V), self.V), t1)
        
#         MGN = torch.reshape(self.a + cross_prod + NN_module_sum, t.shape)
        
#         MGN = positif_phi_t(MGN)
        
#         return MGN
    
#     def show_plot_(self,t, epoch):
        
#         test = torch.linspace(0,1,100).to(t)
        
#         y = self.forward(test)
#         y_np = y.detach().cpu().numpy()
        
#         plt.plot(test.detach().cpu().numpy(), y_np)
#         plt.title("Time change func at epoch : " + str(epoch))
#         plt.show()