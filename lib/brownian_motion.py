# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:02:38 2022

@author: naouf
"""

import numpy as np


def get_dW_old(T_vec) -> np.ndarray:
    
    """
    Sample from a normal distribution with 0 mean,
    and variance given by diff(T_vec) to simulate
    Brownian motion increments  dW
    
    """
    T_vec_tilde = np.insert(T_vec, 0, 0)
    var_vec = np.diff(T_vec_tilde)
    
    return np.random.normal(0, np.sqrt(var_vec))

def get_dW(T_vec, dim_input = 1) -> np.ndarray:
    
    """
    Sample from a normal distribution with 0 mean,
    and variance given by diff(T_vec) to simulate
    Brownian motion increments  dW
    
    """
    dim_T = len(T_vec.shape)
    if (dim_T != dim_input):
        print("Attention : la taille du vecteur temps du Brownien est différente des param !!")
    
    if (dim_input > 1):
        zero_to_insert = np.zeros((1, dim_input))
    elif(dim_input == 1):
        zero_to_insert = np.zeros((1))
        
    
    T_vec_tilde = np.concatenate((zero_to_insert, T_vec) , axis = 0)

    var_vec = np.diff(T_vec_tilde, axis = 0)
    
    return np.random.normal(0, np.sqrt(var_vec))


def get_W(T_vec, dim_input = 1) -> np.ndarray:
    
    """
    create a Brownian motion trajectory by sampling from get_dW
    and transforming the increments into values
    
    """
    
    dW = get_dW(T_vec, dim_input)
    
    return dW.cumsum(axis = 0)



    
    """
    create a time vector of random increment (Poisson process)
        
    """