# Copyright (c) 2019-present Royal Bank of Canada
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


#
import os.path as osp
import time
import os

import sys

#file_path = 'C:/Users/naouf/Documents/0-These/2-Scripts/10-Sto_norm_flow/5-TCNF'
#os.chdir(file_path)


import lib.utils as utils
import numpy as np
import torch
from lib.utils import optimizer_factory
import matplotlib.pyplot as plt

from bm_sequential import get_dataset
from ctfp_tools import build_augmented_model_tabular
from ctfp_tools import run_ctfp_model as run_model, parse_arguments, plot_model
from train_misc import (
    create_regularization_fns,
    get_regularization,
    append_regularization_to_log,
)
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time

from torch.optim import lr_scheduler

#from lib.UMNN.models.UMNN import MonotonicNN


RUNNINGAVE_PARAM = 0.7
torch.backends.cudnn.benchmark = True


def save_model(args, aug_model, time_change_func, optimizer, epoch, itr, save_path):
    """
    save CTFP model's checkpoint during training

    Parameters:
        args: the arguments from parse_arguments in ctfp_tools
        aug_model: the CTFP Model
        optimizer: optimizer of CTFP model
        epoch: training epoch
        itr: training iteration
        save_path: path to save the model
    """
    torch.save(
        {
            "args": args,
            "state_dict": aug_model.module.state_dict()
            if torch.cuda.is_available() and not args.use_cpu
            else aug_model.state_dict(),
            "time_func_state_dict": time_change_func.state_dict()
            if torch.cuda.is_available() and not args.use_cpu
            else time_change_func.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "last_epoch": epoch,
            "iter": itr,
        },
        save_path,
    )


if __name__ == "__main__":
    args = parse_arguments()
    
    #### args ######
    
    # args.batch_size = 100
    # args.test_batch_size = 100
    # args.num_blocks = 1
    # args.log_freq = 1
    # args.save = "experiments\\ou_multi_K3_N6"
    # args.num_workers = 2
    # args.layer_type = "concat"
    # args.dims = '32,64,128,64,32'
    # args.nonlinearity = "tanh"
    # args.lr = 5e-4
    # args.num_epochs = 50
    # args.data_path = os.path.join(file_path, "data/ou_2_multi_sym_neb.pkl")
    # args.activation = "identity"
    # #args.resume = os.path.join(file_path, args.save , "checkpt_last.pth")
    # args.time_change = "M_MGN"
    # args.time_change_MGN_K = 3
    # args.time_change_MGN_N = 3
    
    # args.l1_reg_loss = False
    # args.alpha_l1_reg_loss = 1.
    
    # args.l2_reg_loss = False
    # args.alpha_l2_reg_loss = 1.
    
    # args.effective_shape = 2
    # args.input_size = 2
    # args.aug_size = 2
    ##### args  ######
    
    
    plot_phi_freq = 10
    #bool_scheduler = False
    
    
    # logger
    utils.makedirs(args.save)
    #path_log = os.path.join(file_path, 'train_TCNF.py')
    logger = utils.get_logger(
        logpath=os.path.join(args.save, "logs"), filepath=os.path.abspath(__file__)
    )

    if args.layer_type == "blend":
        logger.info(
            "!! Setting time_length from None to 1.0 due to use of Blend layers."
        )
        args.time_length = 1.0
    logger.info(args)
    if not args.no_tb_log:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(osp.join(args.save, "tb_logs"))
        writer.add_text("args", str(args))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get deivce
    if args.use_cpu:
        device = torch.device("cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_loader, val_loader = get_dataset(args)

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)

    aug_model, time_change_func = build_augmented_model_tabular(
        args,
        args.aug_size + args.effective_shape,
        regularization_fns=regularization_fns,
        time_change = args.time_change,
        time_change_dims = args.time_change_dims
    )
    
    set_cnf_options(args, aug_model)
    logger.info(aug_model)

    logger.info(
        "Number of trainable parameters: {}".format(count_parameters(aug_model))
    )

    # optimizer
    parameter_list = list(aug_model.parameters())+ list(time_change_func.parameters())
    optimizer, num_params = optimizer_factory(args, parameter_list)
    if args.bool_scheduler:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=5)
    
    print("Num of Parameters: %d" % num_params)

    # restore parameters
        
    itr = 0
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        aug_model.load_state_dict(checkpt["state_dict"])
        time_change_func.load_state_dict(checkpt["time_func_state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)
        if "iter" in checkpt.keys():
            itr = checkpt["iter"]
        if "last_epoch" in checkpt.keys():
            args.begin_epoch = checkpt["last_epoch"] + 1

    if torch.cuda.is_available() and not args.use_cpu:
        aug_model = torch.nn.DataParallel(aug_model).cuda()
        time_change_func = time_change_func.cuda()

    # For visualization.

    time_meter = utils.RunningAverageMeter(RUNNINGAVE_PARAM)
    loss_meter = utils.RunningAverageMeter(RUNNINGAVE_PARAM)
    steps_meter = utils.RunningAverageMeter(RUNNINGAVE_PARAM)
    grad_meter = utils.RunningAverageMeter(RUNNINGAVE_PARAM)
    tt_meter = utils.RunningAverageMeter(RUNNINGAVE_PARAM)

    best_loss = float("inf")
    l2_val_reg_loss_per_epoch = []
    l1_val_reg_loss_per_epoch = []
    l2_train_reg_loss_per_epoch = []
    l1_train_reg_loss_per_epoch = []
    likli_val_loss_per_epoch = []
    num_best_epoch = 0
    start_time = time.time()
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        aug_model.train()
        
        l2_train_reg_loss = []
        l1_train_reg_loss = []
        
        for temp_idx, x in enumerate(train_loader):
            ## x is a tuple of (values, times, stdv, masks)
            start = time.time()
            optimizer.zero_grad()

            # cast data and move to device
            x = map(cvt, x)
            values, times, vars, masks = x
            # compute loss
            loss, reg_loss_time_change = run_model(args, aug_model, values, times, vars, masks, time_change_func)

            total_time = count_total_time(aug_model)
            ## Assume the base distribution be Brownian motion

            if regularization_coeffs:
                reg_states = get_regularization(aug_model, regularization_coeffs)
                reg_loss = sum(
                    reg_state * coeff
                    for reg_state, coeff in zip(reg_states, regularization_coeffs)
                    if coeff != 0
                )
                loss = loss + reg_loss
            
            if args.l2_reg_loss:
                loss = loss + reg_loss_time_change[0]
            if args.l1_reg_loss:
                loss = loss + reg_loss_time_change[1]
                
            l2_train_reg_loss.append(reg_loss_time_change[0].data.cpu().numpy())
            l1_train_reg_loss.append(reg_loss_time_change[1].data.cpu().numpy())
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                aug_model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            
        
            time_meter.update(time.time() - start)
            loss_meter.update(loss.item())
            steps_meter.update(count_nfe(aug_model))
            grad_meter.update(grad_norm)
            tt_meter.update(total_time)

            if not args.no_tb_log:
                writer.add_scalar("train/NLL", loss.cpu().data.item(), itr)

            if itr % args.log_freq == 0:
                log_message = (
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Bit/dim {:.4f}({:.4f}) | "
                    "Steps {:.0f}({:.2f}) | Grad Norm {:.4f}({:.4f}) | Total Time "
                    "{:.2f}({:.2f})".format(
                        itr,
                        time_meter.val,
                        time_meter.avg,
                        loss_meter.val,
                        loss_meter.avg,
                        steps_meter.val,
                        steps_meter.avg,
                        grad_meter.val,
                        grad_meter.avg,
                        tt_meter.val,
                        tt_meter.avg,
                    )
                )
                
                reg_loss_message = "Reg_loss l2: %.4f | Reg_loss l1: %.4f"
                print(reg_loss_message %(reg_loss_time_change[0].item(), reg_loss_time_change[1].item()))
                
                
                if regularization_coeffs:
                    log_message = append_regularization_to_log(
                        log_message, regularization_fns, reg_states
                    )
                logger.info(log_message)
                                
                
            itr += 1
        
        # mean l2 regularisation loss
        l2_train_reg_loss_per_epoch.append(np.mean(l2_train_reg_loss))
        l1_train_reg_loss_per_epoch.append(np.mean(l1_train_reg_loss))
        
        if epoch % plot_phi_freq == 0:
            
            time_change_func.show_plot_(times, epoch, args)
            #plot_model(aug_model, times, epoch)
            
        if epoch % args.val_freq == 0:
            
            
            
            with torch.no_grad():
                start = time.time()
                logger.info("validating...")
                losses = []
                l2_val_reg_losses = []
                l1_val_reg_losses = []
                num_observes = []
                aug_model.eval()
                for temp_idx, x in enumerate(val_loader):
                    ## x is a tuple of (values, times, stdv, masks)
                    start = time.time()

                    # cast data and move to device
                    x = map(cvt, x)
                    values, times, vars, masks = x
                    loss, reg_loss_time_change = run_model(args, aug_model, values, times, vars, masks, time_change_func)
                    
                    if args.l2_reg_loss:
                        loss = loss + reg_loss_time_change[0]
                    if args.l1_reg_loss:
                        loss = loss + reg_loss_time_change[1]
                        
                    # compute loss
                    losses.append(loss.data.cpu().numpy())
                    l2_val_reg_losses.append(reg_loss_time_change[0].data.cpu().numpy())
                    l1_val_reg_losses.append(reg_loss_time_change[1].data.cpu().numpy())
                    
                    num_observes.append(torch.sum(masks).data.cpu().numpy())

                loss = np.sum(np.array(losses) * np.array(num_observes)) / np.sum(
                    num_observes
                )
                
                likli_val_loss_per_epoch.append(loss)
                l2_val_reg_loss_per_epoch.append(np.mean(l2_val_reg_losses))
                l1_val_reg_loss_per_epoch.append(np.mean(l1_val_reg_losses))
                
                if not args.no_tb_log:
                    writer.add_scalar("val/NLL", loss, epoch)
                logger.info(
                    "Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}".format(
                        epoch, time.time() - start, loss
                    )
                )
                        
                save_model(
                    args,
                    aug_model,
                    time_change_func,
                    optimizer,
                    epoch,
                    itr,
                    os.path.join(args.save, "checkpt_last.pth"),
                )
                save_model(
                    args,
                    aug_model,
                    time_change_func,
                    optimizer,
                    epoch,
                    itr,
                    os.path.join(args.save, "checkpt_%d.pth") % (epoch),
                )

                if loss < best_loss:
                    
                    num_best_epoch = epoch
                    best_loss = loss
                    save_model(
                        args,
                        aug_model,
                        time_change_func,
                        optimizer,
                        epoch,
                        itr,
                        os.path.join(args.save, "checkpt_best.pth"),
                    )
            
            if args.bool_scheduler:
                scheduler.step(loss)
            
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    fig, ax = plt.subplots()
    ax.plot(l2_train_reg_loss_per_epoch, label = "train_reg_loss")
    ax.plot(l2_val_reg_loss_per_epoch, label = "val_reg_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("L2 reg loss for func : " + args.time_change)
    leg = ax.legend()
    
    fig, ax = plt.subplots()
    ax.plot(l1_train_reg_loss_per_epoch, label = "train_reg_loss")
    ax.plot(l1_val_reg_loss_per_epoch, label = "val_reg_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("L1 reg loss for func : " + args.time_change)
    leg = ax.legend()
    
    fig, ax = plt.subplots()
    ax.plot(likli_val_loss_per_epoch)
    ax.set_title("Validation loss : " + args.time_change)