# Copyright (c) 2019-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path as osp

import lib.layers as layers
import numpy as np
import torch
import torch.nn.functional as F
from lib.utils import sample_standard_gaussian

from train_misc import set_cnf_options

import matplotlib.pyplot as plt

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams"]


def parse_arguments():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser("Continuous Time Flow Process")
    parser.add_argument("--data_path", type=str, default="data/gbm_2.pkl")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dims", type=str, default="8,32,32,8")
    parser.add_argument(
        "--aug_hidden_dims",
        type=str,
        default=None,
        help="The hiddden dimension of the odenet taking care of augmented dimensions",
    )
    parser.add_argument(
        "--aug_dim",
        type=int,
        default=0,
        help="The dimension along which input is augmented. 0 for 1-d input",
    )
    parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
    parser.add_argument(
        "--num_blocks", type=int, default=1, help="Number of stacked CNFs."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="ode_rnn",
        choices=["ode_rnn", "rnn", "np", "attentive_np"],
    )

    parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
    parser.add_argument(
        "--layer_type",
        type=str,
        default="ignore",
        choices=[
            "ignore",
            "concat",
            "concat_v2",
            "squash",
            "concatsquash",
            "concatcoord",
            "hyper",
            "blend",
        ],
    )
    parser.add_argument(
        "--divergence_fn",
        type=str,
        default="approximate",
        choices=["brute_force", "approximate"],
    )
    parser.add_argument(
        "--nonlinearity",
        type=str,
        default="softplus",
        choices=["tanh", "relu", "softplus", "elu", "swish"],
    )
    parser.add_argument("--solver", type=str, default="dopri5", choices=SOLVERS)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--step_size", type=float, default=None, help="Optional fixed step size."
    )

    parser.add_argument(
        "--test_solver", type=str, default=None, choices=SOLVERS + [None]
    )
    parser.add_argument("--test_atol", type=float, default=None)
    parser.add_argument("--test_rtol", type=float, default=None)

    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--aug_size", type=int, default=1, help="size of time")
    parser.add_argument(
        "--latent_size", type=int, default=10, help="size of latent dimension"
    )
    parser.add_argument(
        "--rec_size", type=int, default=20, help="size of the recognition network"
    )
    parser.add_argument(
        "--rec_layers",
        type=int,
        default=1,
        help="number of layers in recognition network(ODE)",
    )
    parser.add_argument(
        "-u",
        "--units",
        type=int,
        default=100,
        help="Number of units per layer in encoder ODE func",
    )
    parser.add_argument(
        "-g",
        "--gru-units",
        type=int,
        default=100,
        help="Number of units per layer in each of GRU update networks in encoder",
    )
    parser.add_argument(
        "-n",
        "--num_iwae_samples",
        type=int,
        default=3,
        help="Number of samples to train IWAE encoder",
    )
    parser.add_argument(
        "--niwae_test", type=int, default=25, help="Numver of IWAE samples during test"
    )
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--time_length", type=float, default=1.0)
    parser.add_argument("--train_T", type=eval, default=True)
    parser.add_argument("--aug_mapping", action="store_true")
    parser.add_argument(
        "--activation", type=str, default="exp", choices=["exp", "softplus", "identity"]
    )

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--test_batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument(
        "--amsgrad", action="store_true", help="use amsgrad for adam optimizer"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum value for sgd optimizer"
    )

    parser.add_argument("--decoder_frequency", type=int, default=3)
    parser.add_argument("--aggressive", action="store_true")

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
    parser.add_argument("--residual", type=eval, default=False, choices=[True, False])
    parser.add_argument("--autoencode", type=eval, default=False, choices=[True, False])
    parser.add_argument("--rademacher", type=eval, default=True, choices=[True, False])
    parser.add_argument("--multiscale", type=eval, default=False, choices=[True, False])
    parser.add_argument("--parallel", type=eval, default=False, choices=[True, False])

    # Regularizations
    parser.add_argument("--l1int", type=float, default=None, help="int_t ||f||_1")
    parser.add_argument("--l2int", type=float, default=None, help="int_t ||f||_2")
    parser.add_argument(
        "--dl2int", type=float, default=None, help="int_t ||f^T df/dt||_2"
    )
    parser.add_argument(
        "--JFrobint", type=float, default=None, help="int_t ||df/dx||_F"
    )
    parser.add_argument(
        "--JdiagFrobint", type=float, default=None, help="int_t ||df_i/dx_i||_F"
    )
    parser.add_argument(
        "--JoffdiagFrobint",
        type=float,
        default=None,
        help="int_t ||df/dx - df_i/dx_i||_F",
    )

    parser.add_argument(
        "--time_penalty", type=float, default=0, help="Regularization on the end_time."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1e10,
        help="Max norm of graidents (default is just stupidly high to avoid any clipping)",
    )

    parser.add_argument("--begin_epoch", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save", type=str, default="time_change_NF")
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument(
        "--no_tb_log", action="store_true", help="Do not use tensorboard logging"
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        choices=["train", "test", "val"],
        help="The split of dataset to evaluate the model on",
    )
    
    parser.add_argument("--time_change", type=str)
    parser.add_argument("--time_change_dims", type=str)
    parser.add_argument("--lambda_reg_loss_time_change", type=float, default = 0.1)
    
    parser.add_argument("--time_change_MGN_K", type=int, default = 1)
    parser.add_argument("--time_change_MGN_N", type=int, default = 10)
    
    parser.add_argument("--time_change_MGN_K_chain", type=str, default = '1')
    parser.add_argument("--time_change_MGN_N_chain", type=str, default = '3')
    
    parser.add_argument("--l1_reg_loss", type=eval, default=False, choices=[True, False], help="mean reg loss")
    parser.add_argument("--l2_reg_loss", type=eval, default=False, choices=[True, False], help="Variance reg loss")

    parser.add_argument("--alpha_l1_reg_loss", type=float, default=0.)
    parser.add_argument("--alpha_l2_reg_loss", type=float, default=0.)
    
    parser.add_argument("--bool_scheduler", type=eval, default=False, choices=[True, False], help="optimization scheduler")
    
    parser.add_argument("--time_change_polynom_degre", type=int, default = 3)

    args = parser.parse_args()
    args.save = osp.join("experiments", args.save)

    args.effective_shape = args.input_size
    return args


def build_augmented_model_tabular(args, dims, regularization_fns=None, time_change = None, time_change_dims = None):
    """
    The function used for creating conditional Continuous Normlizing Flow
    with augmented neural ODE

    Parameters:
        args: arguments used to create conditional CNF. Check args parser for details.
        dims: dimension of the input. Currently only allow 1-d input.
        regularization_fns: regularizations applied to the ODE function

    Returns:
        a ctfp model based on augmened neural ode
    """
    hidden_dims = tuple(map(int, args.dims.split(",")))
    if args.aug_hidden_dims is not None:
        aug_hidden_dims = tuple(map(int, args.aug_hidden_dims.split(",")))
    else:
        aug_hidden_dims = None

    def build_cnf():
        diffeq = layers.AugODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            effective_shape=args.effective_shape,
            strides=None,
            conv=False,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
            aug_dim=args.aug_dim,
            aug_mapping=args.aug_mapping,
            aug_hidden_dims=args.aug_hidden_dims,
        )
        odefunc = layers.AugODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
            effective_shape=args.effective_shape,
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
            rtol=args.rtol,
            atol=args.atol,
        )
        return cnf
    
    
    chain = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [
            layers.MovingBatchNorm1d(
                dims, bn_lag=args.bn_lag, effective_shape=args.effective_shape
            )
            for _ in range(args.num_blocks)
        ]
        bn_chain = [
            layers.MovingBatchNorm1d(
                dims, bn_lag=args.bn_lag, effective_shape=args.effective_shape
            )
        ]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)
    set_cnf_options(args, model)
    
    if time_change == "exp":
        time_change_func = layers.time_change_func_exp(dim_input = args.effective_shape)
    elif time_change == "id":
        time_change_func = layers.time_change_func_id()
    elif time_change == "M_MGN":
        depth = args.time_change_MGN_K
        latent_dim = args.time_change_MGN_N
        time_change_func = layers.time_change_func_M_MGN(depth, latent_dim, dim_input = args.aug_size)
    elif time_change == "M_MGN_chain":
        depth_chain = args.time_change_MGN_K_chain
        latent_dim_chain = args.time_change_MGN_N_chain
        time_change_func = layers.time_change_func_M_MGN_chain(depth_chain, latent_dim_chain)
    else:
        raise NotImplementedError
        
    time_change_model = layers.TimeChangedCNF(model, time_change_func)
    
    return time_change_model, time_change_func


def build_augmented_model_tabular_explicit(args, dims, time_change = None):
    
    hidden_dims = tuple(map(int, args.dims.split(",")))
    if args.aug_hidden_dims is not None:
        aug_hidden_dims = tuple(map(int, args.aug_hidden_dims.split(",")))
    else:
        aug_hidden_dims = None

    
    model = layers.explicit_NF.ExplicitNFModel(args.effective_shape)

        
    if time_change == "exp":
        time_change_func = layers.time_change_func_exp(dim_input = args.effective_shape)
    elif time_change == "id":
        time_change_func = layers.time_change_func_id()
    elif time_change == "M_MGN":
        depth = args.time_change_MGN_K
        latent_dim = args.time_change_MGN_N
        time_change_func = layers.time_change_func_M_MGN(depth, latent_dim, dim_input = args.aug_size)
    elif time_change == "M_MGN_chain":
        depth_chain = args.time_change_MGN_K_chain
        latent_dim_chain = args.time_change_MGN_N_chain
        time_change_func = layers.time_change_func_M_MGN_chain(depth_chain, latent_dim_chain)
    else:
        raise NotImplementedError
        
    time_change_model = layers.TimeChangedExplicitNF(model, time_change_func)
    
    return time_change_model, time_change_func

def log_jaco(values, reverse=False):
    """
    compute log transformation and log determinant of jacobian

    Parameters:
        values: tensor to be transformed
        reverse (bool): If reverse is False, given z_1 return z_0 = log(z_1) and
                        log det of d z_1/d z_0. If reverse is True, given z_0
                        return z_1 = exp(z_0) and log det of d z_1/d z_0

    Returns:
        transformed tesnors and log determinant of the transformation
    """
    if not reverse:
        log_values = torch.log(values)
        return log_values, torch.sum(log_values, dim=2)
    else:
        return torch.exp(values), torch.sum(values, dim=2)


def inversoft_jaco(values, reverse=False):
    """
    compute softplus  transformation and log determinant of jacobian

    Parameters:
        values: tensor to be transformed
        reverse (bool): If reverse is False, given z_1 return
                        z_0 = inverse_softplus(z_1) and log det of d z_1/d z_0.
                        If reverse is True, given z_0 return z_1 = softplus(z_0)
                        and log det of d z_1/d z_0

    Returns:
        transformed tesnors and log determinant of the transformation
    """
    if not reverse:
        inverse_values = torch.log(1 - torch.exp(-values)) + values
        log_det = torch.sum(
            inverse_values - torch.nn.functional.softplus(inverse_values), dim=2
        )
        return inverse_values, log_det
    else:
        log_det = torch.sum(values - torch.nn.functional.softplus(values), dim=2)
        return torch.nn.functional.softplus(values)


def compute_loss(log_det, base_variables, vars, masks, transformed_times, reg_loss):
    """
    This function computes the loss of observations with respect to base wiener
    process.

    Parameters:
        log_det: log determinant of transformation 1-D vectors of size
                 batch_size*length
        base_variables: Tensor after mapping observations back to the space of
                        base Wiener process. 2-D tensor of size batch_size*length
                        x input_shape
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size*length x input_shape
        masks: Binary tensor showing whether a place is actual observation or
               padded dummy variable. 1-D binary vectors of size
               batch_size*length

    Returns:
        the step-wise mean of observations' negative log likelihood
    """
    mean_martingale = base_variables.clone()
    mean_martingale[:, 1:] = base_variables.clone()[:, :-1]
    mean_martingale[:, 0:1] = 0
    mean_martingale = mean_martingale.view(-1, mean_martingale.shape[2])
    base_variables = base_variables.view(-1, base_variables.shape[2])
    non_zero_idx = masks.nonzero()[:, 0]
    mean_martingale_masked = mean_martingale[non_zero_idx]
    #vars_masked = vars[non_zero_idx]
    log_det_masked = log_det[non_zero_idx]
    base_variables_masked = base_variables[non_zero_idx]
    num_samples = non_zero_idx.shape[0]
    
    
    
    ## corriger les dimensions à cause du non_zero_idx qui est faut
    delta_transformed_time = transformed_times[:,1:] - transformed_times[:,0:-1]
    delta_transformed_time = torch.cat(( transformed_times[:,0:1], delta_transformed_time) , dim = 1)
    delta_transformed_time = delta_transformed_time.view(-1, delta_transformed_time.shape[2])
    delta_transformed_time_masked = delta_transformed_time[non_zero_idx]
    
    
    
    if base_variables.shape[1] > 1:
                
        multi_delta_transformed_time = torch.diag_embed(delta_transformed_time_masked)
        
        normal_distri = torch.distributions.MultivariateNormal(
            mean_martingale_masked, multi_delta_transformed_time
            )
          
        # try:
        #     normal_distri = torch.distributions.MultivariateNormal(
        #         mean_martingale_masked, multi_delta_transformed_time
        #         )
        # except ValueError:
        #     print("ca casse")
            
    elif base_variables.shape[1] == 1:
        normal_distri = torch.distributions.Normal(
            mean_martingale_masked, torch.sqrt(delta_transformed_time_masked)
        )
        
    LL = torch.squeeze(normal_distri.log_prob(base_variables_masked)) - torch.squeeze(log_det_masked)
    return (-torch.mean(LL) + reg_loss)


def batch_cov(points):
    B, N, D = points.size() # B : batch size, N : length of path, D : dimensions
    mean = points.mean(dim = 0)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim = 0) / (B - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def compute_time_change_reg_loss_multi(aug_model, values, base_values, times, time_change_fun, args):
    
    covariance_values = batch_cov(values)
    mean_values = torch.mean(values, dim = 0)
    
    
    flat_times = times.view(-1, args.aug_size)
    flat_base_values = base_values.view(-1, args.input_size)
    model_inputs = torch.cat((flat_base_values, flat_times), dim = 1)
    model_outputs, _ , _= aug_model(model_inputs, torch.zeros(model_inputs.shape[0], 1), reverse = True)#, dim_x = args.input_size)
    model_outputs = torch.reshape(model_outputs[:,0:args.input_size], values.shape)
    
    covariance_model = batch_cov(model_outputs)
    mean_model = torch.mean(model_outputs, dim = 0)
    
    reg_loss_l2 = (torch.sum(torch.square(covariance_model - covariance_values))/covariance_model.shape[0])
    reg_loss_l1 = (torch.sum(torch.square(mean_model - mean_values))/mean_model.shape[0])
    
    scale_fact_l2 = 10**torch.floor(torch.log10(torch.abs(reg_loss_l2).max()))
    scale_fact_l1 = 10**torch.floor(torch.log10(torch.abs(reg_loss_l1).max()))
    
    scaled_reg_l2 = (reg_loss_l2/scale_fact_l2)/10
    scaled_reg_l1 = (reg_loss_l1/scale_fact_l1)/10
    
    return scaled_reg_l2, scaled_reg_l1


def compute_time_change_reg_loss(aug_model, values, base_values, times, time_change_fun, args):
    
    std_values = torch.std(values, dim = 0)
    mean_values = torch.mean(values, dim = 0)
    
    if args.activation == "exp":
        transform_base_values, _ = log_jaco(torch.unsqueeze(base_values, dim = 2), reverse = True)
    elif args.activation == "softplus":
        transform_base_values, _ = inversoft_jaco(torch.unsqueeze(base_values, dim = 2), reverse = True)
    elif args.activation == "identity":
        transform_base_values = values
        
    else:
        raise NotImplementedError
    
    flat_times = torch.flatten(times)
    flat_transform_base_values = torch.flatten(transform_base_values)
    model_inputs = torch.cat((flat_transform_base_values[:,None], flat_times[:,None]), dim = 1)
    model_outputs, _ , _ = aug_model(model_inputs, torch.zeros(model_inputs.shape[0], 1), reverse = True)
    
    model_outputs = torch.reshape(model_outputs[:,0], values.shape)
    
    std_model = torch.std(model_outputs, dim = 0)
    mean_model = torch.mean(model_outputs, dim = 0)
    
    reg_loss_l2 = args.alpha_l2_reg_loss*torch.sum(torch.abs(std_model - std_values))/len(std_model)
    reg_loss_l1 = args.alpha_l1_reg_loss*torch.sum(torch.abs(mean_model - mean_values))/len(mean_model)
        
    return reg_loss_l2, reg_loss_l1


def compute_ll(log_det, base_variables, vars, masks):
    """
    This function computes the log likelihood of observations with respect to base wiener
    process used for latent_CTFP.

    Parameters:
        log_det: log determinant of transformation 2-D vectors of size
                 batch_size x length
        base_variables: Tensor after mapping observations back to the space of
                        base Wiener process. 3-D tensor of size batch_size x
                        length x input_shape
        vars: Difference between consequtive observation time stampes.
              3-D tensor of size batch_size x length x 1
        masks: Binary tensor showing whether a place is actual observation or
               padded dummy variable. 2-D binary vectors of size
               batch_size x length

    Returns:
        the sum of log likelihood of all observations
    """
    mean_martingale = base_variables.clone()
    mean_martingale[:, 1:] = base_variables.clone()[:, :-1]
    mean_martingale[:, 0:1] = 0
    normal_distri = torch.distributions.Normal(mean_martingale, torch.sqrt(vars))
    LL = normal_distri.log_prob(base_variables)
    LL = (torch.sum(LL, -1) - log_det) * masks
    return torch.sum(LL, -1)


def run_ctfp_model(args, aug_model, values, times, vars, masks, time_change_func):
    """
    Functions for running the ctfp model

    Parameters:
        args: arguments returned from parse_arguments
        aug_model: ctfp model as decoder
        values: observations, a 3-D tensor of shape batchsize x max_length x input_size
        times: observation time stampes, a 3-D tensor of shape batchsize x max_length x 1
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size x length
        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
               position is observation or padded dummy variables

    Returns:
    """
    aux = torch.cat([torch.zeros_like(values), times], dim = 2) #time_change_func(times).unsqueeze(2)
    aux = aux.view(-1, aux.shape[2])
    #aux, _ , _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
    #aux = aux[:, args.effective_shape:]
    ## run flow backward
    if args.activation == "exp":
        transform_values, transform_logdet = log_jaco(values)
    elif args.activation == "softplus":
        transform_values, transform_logdet = inversoft_jaco(values)
    elif args.activation == "identity":
        transform_values = values
        transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
    else:
        raise NotImplementedError
    
    
    #if args.effective_shape == 1:
    #    transformed_times = time_change_func(times.view(-1, times.shape[2]))
    #else:
    #    expanded_times = times.expand(-1,-1, args.effective_shape)
    #    transformed_times = time_change_func(expanded_times.view(-1, expanded_times.shape[2]))
    
    #aug_values = torch.cat(
    #    [transform_values.view(-1, transform_values.shape[2]), transformed_times], dim=1
    #)
    
    aug_values = torch.cat(
        [transform_values.view(-1, transform_values.shape[2]), times.view(-1, times.shape[2])], dim=1
    )
    
    
    base_values, flow_logdet, reg_loss = aug_model(
        aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
    )
    
    transformed_times = time_change_func(times.view(-1, times.shape[2]))    
    #transformed_times = base_values[:, args.input_size : ]
    transformed_times = transformed_times.view(times.shape[0], -1, args.input_size)
    
    
    base_values = base_values[:, : args.input_size]
    base_values = base_values.view(values.shape[0], -1, args.input_size)
    
    
    
    # flow_logdet and transform_logdet are both of size length*batch_size
    loss = compute_loss(
        flow_logdet.view(-1, 1)
        + transform_logdet.view(-1, 1),
        base_values,
        vars.view(-1, 1),
        masks.view(-1),
        transformed_times,
        reg_loss
    )
    
    
    
    time_change_reg_loss = torch.zeros(1), torch.zeros(1)
    if (args.l2_reg_loss or args.l1_reg_loss) :
        
        
        time_change_reg_loss = compute_time_change_reg_loss_multi(
            aug_model,
            values,
            base_values,
            times,
            time_change_func,
            args)
        
        
    return loss, time_change_reg_loss


def run_ctfp_model_old(args, aug_model, values, times, vars, masks, time_change_func):
    
    
    aux = torch.cat([torch.zeros_like(values), times], dim=2)
    aux = aux.view(-1, aux.shape[2])
    aux, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
    aux = aux[:, args.effective_shape:]
    ## run flow backward
    if args.activation == "exp":
        transform_values, transform_logdet = log_jaco(values)
    elif args.activation == "softplus":
        transform_values, transform_logdet = inversoft_jaco(values)
    elif args.activation == "identity":
        transform_values = values
        transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
    else:
        raise NotImplementedError

    aug_values = torch.cat(
        [transform_values.view(-1, transform_values.shape[2]), aux], dim=1
    )
    base_values, flow_logdet = aug_model(
        aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
    )

    base_values = base_values[:, : args.effective_shape]
    base_values = base_values.view(values.shape[0], -1, args.effective_shape)

    ## flow_logdet and transform_logdet are both of size length*batch_size
    loss = compute_loss(
        flow_logdet.view(-1, args.effective_shape)
        + transform_logdet.view(-1, args.effective_shape),
        base_values,
        vars.view(-1, args.effective_shape),
        masks.view(-1),
        times,
        time_change_func
    )
    return loss

def create_separate_batches(data, times, masks):
    """
    Separate a batch of data with unequal length into smaller batch of size 1
    the length of each smaller batch is different and contains no padded dummy
    variables

    Parameters:
       data: observations, a 3-D tensor of shape batchsize x max_length x input_size
       times: observation time stamps, a 2-D tensor of shape batchsize x max_length
       masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
              position is observation or padded dummy variables

    Returns:
        a list of tuples containing the data, time, masks
    """
    batch_size = data.shape[0]
    data_size = data.shape[-1]
    ## only repeat the last dimension to concatenate with data
    repeat_times = tuple([1] * (len(data.shape) - 1) + [data_size])
    separate_batches = []
    for i in range(batch_size):
        length = int(torch.sum(masks[i]))
        data_item = data[i: i + 1, :length]
        time_item = times[i, :length].squeeze(-1)
        mask_item = masks[i: i + 1, :length].unsqueeze(-1).repeat(*repeat_times)
        separate_batches.append((torch.cat([data_item, mask_item], -1), time_item))
    return separate_batches


def run_latent_ctfp_model(
        args, encoder, aug_model, values, times, vars, masks, evaluation=False
):
    """
    Functions for running the latent ctfp model

    Parameters:
        args: arguments returned from parse_arguments
        encoder: ode_rnn model as encoder
        aug_model: ctfp model as decoder
        values: observations, a 3-D tensor of shape batchsize x max_length x input_size
        times: observation time stampes, a 3-D tensor of shape batchsize x max_length x 1
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size x length
        masks: a 2-D binary tensor of shape batchsize x max_length showing whehter the
               position is observation or padded dummy variables
        evluation (bool): whether to run the latent ctfp model in the evaluation
                          mode. Return IWAE if set to true. Return both IWAE and
                          training loss if set to false

    Returns:
        Return IWAE if evaluation set to true.
        Return both IWAE and training loss if evaluation set to false.
    """
    if evaluation:
        num_iwae_samples = args.niwae_test
        batch_size = args.test_batch_size
    else:
        num_iwae_samples = args.num_iwae_samples
        batch_size = args.batch_size
    data_batches = create_separate_batches(values, times, masks)
    mean_list, stdv_list = [], []
    for item in data_batches:
        z_mean, z_stdv = encoder(item[0], item[1])
        mean_list.append(z_mean)
        stdv_list.append(z_stdv)
    means = torch.cat(mean_list, dim=1)
    stdvs = torch.cat(stdv_list, dim=1)
    # Sample latent variables
    repeat_times = [1] * len(means.shape)
    repeat_times[0] = num_iwae_samples
    means = means.repeat(*repeat_times)
    stdvs = stdvs.repeat(*repeat_times)
    latent = sample_standard_gaussian(means, stdvs)

    ## Decode latent

    latent_sequence = latent.view(-1, args.latent_size).unsqueeze(1)
    max_length = times.shape[1]
    latent_sequence = latent_sequence.repeat(1, max_length, 1)
    time_to_cat = times.repeat(num_iwae_samples, 1, 1)
    times = torch.cat([latent_sequence, time_to_cat], -1)

    ## run flow forward to get augmented dimensions
    values = values.repeat(num_iwae_samples, 1, 1)
    aux = torch.cat([torch.zeros_like(values), times], dim=2)
    aux = aux.view(-1, aux.shape[2])
    aux, _ = aug_model(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
    aux = aux[:, args.effective_shape:]
    ## run flow backward
    if args.activation == "exp":
        transform_values, transform_logdet = log_jaco(values)
    elif args.activation == "softplus":
        transform_values, transform_logdet = inversoft_jaco(values)
    elif args.activation == "identity":
        transform_values = values
        transform_logdet = torch.sum(torch.zeros_like(values), dim=2)
    else:
        raise NotImplementedError

    aug_values = torch.cat(
        [transform_values.view(-1, transform_values.shape[2]), aux], dim=1
    )

    base_values, flow_logdet = aug_model(
        aug_values, torch.zeros(aug_values.shape[0], 1).to(aug_values)
    )

    base_values = base_values[:, : args.effective_shape]
    base_values = base_values.view(values.shape[0], -1, args.effective_shape)

    ## flow_logdet and transform_logdet are both of size length*batch_size x length
    flow_logdet = flow_logdet.sum(-1).view(num_iwae_samples * batch_size, -1)
    transform_logdet = transform_logdet.view(num_iwae_samples * batch_size, -1)
    if len(vars.shape) == 2:
        vars_unsqueed = vars.unsqueeze(-1)
    else:
        vars_unsqueed = vars
    ll = compute_ll(
        flow_logdet + transform_logdet,
        base_values,
        vars_unsqueed.repeat(num_iwae_samples, 1, 1),
        masks.repeat(num_iwae_samples, 1),
    )
    ll = ll.view(num_iwae_samples, batch_size)
    ## Reconstruction log likelihood
    ## Compute KL divergence and compute IWAE
    posterior = torch.distributions.Normal(means[:1], stdvs[:1])
    prior = torch.distributions.Normal(
        torch.zeros_like(means[:1]), torch.ones_like(stdvs[:1])
    )
    # kl_latent = kl_divergence(posterior, prior).sum(-1)

    prior_z = prior.log_prob(latent).sum(-1)
    posterior_z = posterior.log_prob(latent).sum(-1)

    weights = ll + prior_z - posterior_z
    loss = -torch.logsumexp(weights, 0) + np.log(num_iwae_samples)
    if evaluation:
        return torch.sum(loss) / torch.sum(masks)
    loss = torch.sum(loss) / (batch_size * max_length)
    loss_training = -torch.sum(F.softmax(weights, 0).detach() * weights) / (
            batch_size * max_length
    )
    return loss, loss_training


def compute_loss_old(log_det, base_variables, vars, masks, times, time_change_func, reg_loss):
    """
    This function computes the loss of observations with respect to base wiener
    process.

    Parameters:
        log_det: log determinant of transformation 1-D vectors of size
                 batch_size*length
        base_variables: Tensor after mapping observations back to the space of
                        base Wiener process. 2-D tensor of size batch_size*length
                        x input_shape
        vars: Difference between consequtive observation time stampes.
              2-D tensor of size batch_size*length x input_shape
        masks: Binary tensor showing whether a place is actual observation or
               padded dummy variable. 1-D binary vectors of size
               batch_size*length

    Returns:
        the step-wise mean of observations' negative log likelihood
    """
    mean_martingale = base_variables.clone()
    mean_martingale[:, 1:] = base_variables.clone()[:, :-1]
    mean_martingale[:, 0:1] = 0
    mean_martingale = mean_martingale.view(-1, mean_martingale.shape[2])
    base_variables = base_variables.view(-1, base_variables.shape[2])
    non_zero_idx = masks.nonzero()[:, 0]
    mean_martingale_masked = mean_martingale[non_zero_idx]
    vars_masked = vars[non_zero_idx]
    log_det_masked = log_det[non_zero_idx]
    base_variables_masked = base_variables[non_zero_idx]
    num_samples = non_zero_idx.shape[0]
    
    
    times_double = times.clone().double().to(times)
    transformed_time = time_change_func(times_double)
    #pos_time = torch.reshape(torch.arange(times.shape[0]*times.shape[1]),times.shape)
    delta_transformed_time = transformed_time[:,1:] - transformed_time[:,0:-1]
    delta_transformed_time = torch.cat(( transformed_time[:,0,None], delta_transformed_time) , dim = 1)
    #pos_delta_time = torch.reshape(torch.arange(delta_transformed_time.shape[0]*delta_transformed_time.shape[1]), delta_transformed_time.shape)
    delta_transformed_time = delta_transformed_time.view(-1, delta_transformed_time.shape[2])
    #pos_delta_time_view = pos_delta_time.view(-1)
    delta_transformed_time_masked = delta_transformed_time[non_zero_idx]
    #pos_delta_time_masked = pos_delta_time_view[non_zero_idx]  
    
    ## temporary fix for small precision
    #delta_transformed_time_masked[layers.which_is(delta_transformed_time_masked,0)] = 1e-6   

    normal_distri = torch.distributions.Normal(
        mean_martingale_masked, torch.sqrt(delta_transformed_time_masked)
    )
    LL = normal_distri.log_prob(base_variables_masked) - log_det_masked
    return (-torch.mean(LL) + reg_loss)

def plot_model(aug_model, times, epoch):
    
    x = torch.linspace(-3, 3, 100).to(times)
    t = torch.linspace(0.01, 1, 100).to(times)
    model_input = torch.cat((x[:,None], t[:,None]), dim = 1)
    
    
    model_output, _, _ = aug_model(model_input, torch.zeros(model_input.shape[0], 1).to(times), reverse=True)
    
    y_np = model_output[:,0].detach().cpu().numpy()
    x_np = np.linspace(-3, 3, 100)
    
    plt.plot(x_np, y_np)
    plt.title("TCNF at epoch : " + str(epoch))
    plt.show()




#########################################################
####    historique dev    ###############################

#### compute loss 18/11/2023 : sauvegarder pour recoder correctement pour le multivarié
#### après ecriture correcte de la solution du OU en 2d, donc sans necessité de matrix de correl

# def compute_loss(log_det, base_variables, vars, masks, transformed_times, reg_loss, correl_transformed_times):
#     """
#     This function computes the loss of observations with respect to base wiener
#     process.

#     Parameters:
#         log_det: log determinant of transformation 1-D vectors of size
#                  batch_size*length
#         base_variables: Tensor after mapping observations back to the space of
#                         base Wiener process. 2-D tensor of size batch_size*length
#                         x input_shape
#         vars: Difference between consequtive observation time stampes.
#               2-D tensor of size batch_size*length x input_shape
#         masks: Binary tensor showing whether a place is actual observation or
#                padded dummy variable. 1-D binary vectors of size
#                batch_size*length

#     Returns:
#         the step-wise mean of observations' negative log likelihood
#     """
#     mean_martingale = base_variables.clone()
#     mean_martingale[:, 1:] = base_variables.clone()[:, :-1]
#     mean_martingale[:, 0:1] = 0
#     mean_martingale = mean_martingale.view(-1, mean_martingale.shape[2])
#     base_variables = base_variables.view(-1, base_variables.shape[2])
#     non_zero_idx = masks.nonzero()[:, 0]
#     mean_martingale_masked = mean_martingale[non_zero_idx]
#     #vars_masked = vars[non_zero_idx]
#     log_det_masked = log_det[non_zero_idx]
#     base_variables_masked = base_variables[non_zero_idx]
#     num_samples = non_zero_idx.shape[0]
    
    
    
#     ## corriger les dimensions à cause du non_zero_idx qui est faut
#     delta_transformed_time = transformed_times[:,1:] - transformed_times[:,0:-1]
#     delta_transformed_time = torch.cat(( transformed_times[:,0:1], delta_transformed_time) , dim = 1)
#     delta_transformed_time = delta_transformed_time.view(-1, delta_transformed_time.shape[2])
#     delta_transformed_time_masked = delta_transformed_time[non_zero_idx]
    
    
    
#     if base_variables.shape[1] > 1:
        
#         #print("test prod : %.f" %torch.prod(delta_transformed_time_masked>0))
#         #if torch.prod(delta_transformed_time_masked>0) == 0:
#         #    #stop ici
#         #    test = 0
        
#         #diag_delta_transformed_time = torch.diag_embed(delta_transformed_time_masked)
#         #correlated_delta_transformed_time_masked = torch.matmul(correl_mat_phi_t, torch.matmul(diag_delta_transformed_time, torch.t(correl_mat_phi_t)))
        
#         # correlated_delta_transformed_time_masked = torch.zeros(delta_transformed_time_masked.shape[0], 
#         #                                                         delta_transformed_time_masked.shape[1], 
#         #                                                         delta_transformed_time_masked.shape[1]
#         #                                                         ).to(transformed_times)
        
#         # # compute the matrix product : correl_mat * delta_phi * correl_mat.t()
#         # correlated_delta_transformed_time_masked[:,0,0] = delta_transformed_time_masked[:,0]+(correl_mat_phi_t[0,1]**2)*delta_transformed_time_masked[:,1]
#         # correlated_delta_transformed_time_masked[:,1,1] = delta_transformed_time_masked[:,1]+(correl_mat_phi_t[1,0]**2)*delta_transformed_time_masked[:,0]
#         # correlated_delta_transformed_time_masked[:,0,1] = delta_transformed_time_masked[:,0]*correl_mat_phi_t[1,0] + delta_transformed_time_masked[:,1]*correl_mat_phi_t[0,1]
#         # correlated_delta_transformed_time_masked[:,1,0] = delta_transformed_time_masked[:,0]*correl_mat_phi_t[1,0] + delta_transformed_time_masked[:,1]*correl_mat_phi_t[0,1]
        
#         #multi_delta_transformed_time = torch.diag_embed(delta_transformed_time_masked)
        
#         delta_correl_transformed_times = correl_transformed_times[:,1:,:] - correl_transformed_times[:,0:-1,:]
#         delta_correl_transformed_times = torch.cat((correl_transformed_times[:,0:1,:], delta_correl_transformed_times), dim = 1)
#         delta_correl_transformed_times = delta_correl_transformed_times.view(-1, delta_correl_transformed_times.shape[2], delta_correl_transformed_times.shape[2])
        
#         normal_distri = torch.distributions.MultivariateNormal(
#             mean_martingale_masked, delta_correl_transformed_times
#             )
          
#         # try:
#         #     normal_distri = torch.distributions.MultivariateNormal(
#         #         mean_martingale_masked, multi_delta_transformed_time
#         #         )
#         # except ValueError:
#         #     print("ca casse")
            
#     elif base_variables.shape[1] == 1:
#         normal_distri = torch.distributions.Normal(
#             mean_martingale_masked, torch.sqrt(delta_transformed_time_masked)
#         )
        
#     LL = torch.squeeze(normal_distri.log_prob(base_variables_masked)) - torch.squeeze(log_det_masked)
#     return (-torch.mean(LL) + reg_loss)

##### Fin comment 18/11/2023





# def compute_loss_new(log_det, base_variables, vars, masks, times, time_change_func):
#     """
#     This function computes the loss of observations with respect to base wiener
#     process.

#     Parameters:
#         log_det: log determinant of transformation 1-D vectors of size
#                  batch_size*length
#         base_variables: Tensor after mapping observations back to the space of
#                         base Wiener process. 2-D tensor of size batch_size*length
#                         x input_shape
#         vars: Difference between consequtive observation time stampes.
#               2-D tensor of size batch_size*length x input_shape
#         masks: Binary tensor showing whether a place is actual observation or
#                padded dummy variable. 1-D binary vectors of size
#                batch_size*length

#     Returns:
#         the step-wise mean of observations' negative log likelihood
#     """
#     mean_martingale = base_variables.clone()
#     mean_martingale[:, 1:] = base_variables.clone()[:, :-1]
#     mean_martingale[:, 0:1] = 0
#     mean_martingale = mean_martingale.view(-1, mean_martingale.shape[2])
#     base_variables = base_variables.view(-1, base_variables.shape[2])
#     non_zero_idx = masks.nonzero()[:, 0]
#     mean_martingale_masked = mean_martingale[non_zero_idx]
#     vars_masked = vars[non_zero_idx]
#     log_det_masked = log_det[non_zero_idx]
#     base_variables_masked = base_variables[non_zero_idx]
#     num_samples = non_zero_idx.shape[0]
    
    
#     transformed_time = time_change_func(times)
    
#     ## Brownian values
#     base_variables_2 = torch.reshape(base_variables, times.shape)
#     transformed_base_values = torch.sqrt(transformed_time/times)*base_variables_2
#     transformed_base_values_masked = transformed_base_values[non_zero_idx]
    
#     transformed_mean_martingale = transformed_base_values.clone()
#     transformed_mean_martingale[:, 1:, 0] = transformed_base_values[:, :-1, 0]
#     transformed_mean_martingale[:, 0:1, 0] = 0
#     transformed_mean_martingale = transformed_mean_martingale.view(-1, transformed_mean_martingale.shape[2])
#     transformed_mean_martingale_masked = transformed_mean_martingale[non_zero_idx]
    
#     delta_transformed_time = transformed_time[:,1:] - transformed_time[:,0:-1]
#     delta_transformed_time = torch.cat(( transformed_time[:,0,None], delta_transformed_time) , dim = 1)
#     delta_transformed_time = delta_transformed_time.view(-1)
#     delta_transformed_time_masked = delta_transformed_time[non_zero_idx]
    
#     ## temporary fix for small precision
#     delta_transformed_time_masked[layers.which_is(delta_transformed_time_masked,0)] = 1e-6
    
#     transformed_normal_distri = torch.distributions.Normal(mean_martingale_masked, torch.sqrt(delta_transformed_time_masked)
#     ## Brownian values
    
    
       

#     normal_distri = torch.distributions.Normal(
#         mean_martingale_masked, torch.sqrt(delta_transformed_time_masked)
#     )
#     LL = normal_distri.log_prob(base_variables_masked) - log_det_masked
    
#     return -torch.mean(LL)