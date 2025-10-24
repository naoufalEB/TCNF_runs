# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 22:47:03 2022

@author: naouf
"""

import matplotlib.pyplot as plt

from lib.brownian_motion import get_W
import numpy as np
import torch
from scipy.stats import norm, multivariate_normal, gaussian_kde
from scipy.linalg import expm
import random

import pickle
import glob
from PIL import Image

from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import io

from scipy.stats import entropy


##########################
######## test dataset

test_data_list = test_loader.dataset.data
test_data_all = np.stack(test_data_list, axis=0)
test_data = test_data_all[:,:,1:]

## Param processus
dim_x = 2
x0 = np.array([1.5, 1.])
A = np.array([[1.3, 0.4],[0.4, 0.7]])
sigma = 1.3
Sigma = sigma*np.eye(dim_x)
mu = np.ones(dim_x)*0
T = 1
dt = 0.01
n = 800 # num of trajectoires
N = 50 # longueur trajectoire
n_bins = 200 # points to compute the densities
quantiles = [0.25, 0.75]

numero_to_plot = [10, 20, 30, 40, 49]

## plot param
t = np.linspace(dt, T, N)
xstart = -3
x = np.linspace(xstart, -xstart, n_bins)

# spectral decompo
w, Q = np.linalg.eigh(A)
D = np.diag(w)
A_reconstructed = Q @ D @ Q.T

x0_arr = np.repeat(x0[None,:,None], N, axis = 0)
exp_neg_A_t = expm(-t[:,None,None]*A[None,:,:])
exp_neg_2A_t = expm(-2*t[:,None,None]*A[None,:,:])
inv_A = np.repeat(np.linalg.inv(A)[None,:,:], N, axis = 0)
I2 = np.repeat(np.eye(dim_x)[None,:,:], N, axis = 0)

time_change_int = (np.exp(2*D*t[:,None,None])-1)/np.diag(D)
time_change_mat = Q @ ((np.exp(2*D*t[:,None,None])-1)/np.diag(D)) @ Q.T

# ######### Ground truth OU : dX_t = -A*X_t*dt + sigma*dW_t

GT_mean = np.matmul(exp_neg_A_t, x0_arr)
#GT_var = 0.5*(sigma**2)*np.matmul(inv_A, I2 - np.matmul(exp_neg_A_t, exp_neg_A_t))
GT_cov_mat = 0.5*(sigma**2) * (exp_neg_2A_t @ time_change_mat)
#GT_var_array = GT_var.diagonal(0,axis1=1, axis2 = 2) # les variances uniquement, pas les covariance


pdf_array = np.zeros((n_bins, n_bins, len(numero_to_plot)))
GT_marginal = np.zeros((n_bins, len(numero_to_plot), dim_x))

quantile_array = np.zeros((len(quantiles), N, dim_x))

XX, YY = np.meshgrid(x, x)


for i in range(len(numero_to_plot)):
        
    dist = multivariate_normal(GT_mean[numero_to_plot[i], :, 0], GT_cov_mat[numero_to_plot[i], ...])
    pdf_array[:, :, i] = dist.pdf(np.dstack((XX, YY)))
        
        
    #for dim_j in range(dim_x):
    #    temp_marg = norm(GT_mean[numero_to_plot[i], dim_j, 0], np.sqrt(GT_var[numero_to_plot[i], dim_j, dim_j]))
    #    GT_marginal[:, i, dim_j] = temp_marg.pdf(x)
    #    quantile_array[:,i, dim_j] = temp_marg.ppf(quantiles)
        
        
        
GT_mean = np.squeeze(GT_mean, axis = 2)




####### TC : X_t = F(W_phi_t, phi_t)
nn_phi_tt = time_change_func(torch.as_tensor(t[:,None], dtype = torch.float).cuda()).detach().cpu().numpy()

plt.plot(t, nn_phi_tt[:,0], label = "$t1$")
plt.plot(t, nn_phi_tt[:,1], label = "$t2$")
plt.title("time change function")
plt.legend()

#TC_pdf_array = np.zeros((n, n, len(numero_to_plot)))
#TC_x_marginal = np.zeros((n, len(numero_to_plot)))
#TC_y_marginal = np.zeros((n, len(numero_to_plot)))
TC_pdf_array = np.zeros((n_bins, n_bins, len(numero_to_plot)))
TC_x_marginal = np.zeros((n_bins, len(numero_to_plot)))
TC_y_marginal = np.zeros((n_bins, len(numero_to_plot)))
TC_marginal = np.zeros((n_bins, len(numero_to_plot), dim_x))


TC_paths = np.zeros((n, N, dim_x))
TC_quantile_array = np.zeros((len(quantiles), N, dim_x))

XX, YY = np.meshgrid(x, x)

torch.cuda.empty_cache()

with torch.no_grad():

    for i in range(len(numero_to_plot)):     
        
        print(str(i + 1) + " / " + str(len(numero_to_plot)))
        
        TT = t[numero_to_plot[i]]*np.ones_like(XX)
                        
        temp_input = np.dstack((XX, YY, TT))
        temp_input_torch = torch.as_tensor(temp_input, dtype=torch.float, device=device).view(-1, temp_input.shape[2])
        
        zz_temp, flow_logdet, _ = aug_model_TC(temp_input_torch, torch.zeros(temp_input_torch.shape[0], 1), reverse=False)#, dim_x = dim_x)
        
        zz_temp_np = zz_temp[:, 0:dim_x ].reshape(n_bins, n_bins, dim_x).detach().cpu().numpy()
        flow_logdet_np = flow_logdet.reshape(XX.shape).detach().cpu().numpy()
    
        dist_W_phi_tt = multivariate_normal(np.zeros(dim_x), np.diag(nn_phi_tt[numero_to_plot[i],:]))
        TC_pdf_array[:,:,i] = np.exp(dist_W_phi_tt.logpdf(zz_temp_np) - flow_logdet_np)
        
        TC_x_marginal[:,i] = np.trapz(TC_pdf_array[:,:,i], x, axis = 0)
        TC_y_marginal[:,i] = np.trapz(TC_pdf_array[:,:,i], x, axis = 1)

for dim_j in range(dim_x) : 
    TC_marginal[:, :, dim_j] = np.trapz(TC_pdf_array[:,:,:], x, axis = dim_j)
    
    
with torch.no_grad():
    
        
    W_phi_tt_array = np.array([get_W(nn_phi_tt, dim_input = dim_x) for j in range(n)])
        
    TT, XX = np.meshgrid(t, np.linspace(xstart, -xstart, n))
    
        
    temp_input = np.concatenate((W_phi_tt_array, TT[:,:,None]) , axis = -1)
    temp_input_torch = torch.as_tensor(temp_input, dtype = torch.float, device = device).view(-1, temp_input.shape[2])
    
    flow_process,_,_ = aug_model_TC(temp_input_torch, torch.zeros(temp_input_torch.shape[0], 1), reverse = True, dim_x = dim_x)
    TC_paths = flow_process[:,0:dim_x].reshape(W_phi_tt_array.shape).detach().cpu().numpy()

TC_mean = np.mean(TC_paths, axis = 0)
TC_sd = np.std(TC_paths, axis = 0)

TC_quantile_array = np.quantile(TC_paths, quantiles,axis = 0)


TC_cov_matrix = np.zeros((N, dim_x, dim_x))

for timestep in range(len(t)):
    TC_cov_matrix[timestep,:,:] = np.cov(TC_paths[:,timestep,:], rowvar = False)

#### Errors

error_pdf = np.abs(TC_pdf_array - pdf_array)
mean_error_pdf = np.mean(error_pdf)
print(f"{mean_error_pdf:.4f}")

plt.scatter(t[numero_to_plot], np.mean(error_pdf, axis =(0,1)))
plt.xlabel('$t$')
plt.title("density mean error")

for i in numero_to_plot:
    fig = plt.figure(figsize=(7, 4), dpi=200)
    plt.imshow(TC_pdf_array[:,:,i], extent = [x[0],x[-1],x[0],x[-1]], cmap = 'viridis', origin = 'lower', aspect='auto')
    plt.title("TC Density | t :" + "{:.2f}".format(t[i]))
    #plt.ylim(top = 3, bottom = -1)
    #plt.xlim(right = 3, left = -1)
    plt.colorbar()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    #plt.savefig(os.path.join(path_plot, "density_j_" + str(j) + ".jpg"))
    


######## Graphique commun
path_plot = 'C:/Users/naouf/Documents/0-These/2-Scripts/10-Sto_norm_flow/5-TCNF/plots'

frames = []
for i in range(len(numero_to_plot)):
    fig, ax = plt.subplots(1, 2, figsize=(9, 5), dpi=200)
    
    # Plot the first image on the left axis
    im1 = ax[0].imshow(TC_pdf_array[:,:,i], extent=[x[0], x[-1], x[0], x[-1]], cmap='viridis', origin='lower', aspect='auto')
    ax[0].set_xlabel("$x1$")
    ax[0].set_ylabel("$x2$")
    ax[0].set_title("TCNF density | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    # Plot the second image on the right axis
    im2 = ax[1].imshow(pdf_array[:,:,i], extent=[x[0], x[-1], x[0], x[-1]], cmap='viridis', origin='lower', aspect='auto')
    ax[1].set_xlabel("$x1$")
    ax[0].set_ylabel("$x2$")
    ax[1].set_title("GT density | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    # Create a common colorbar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    # Convert the plot to an image
    #buf = io.BytesIO()
    #plt.savefig(buf, format='png')
    #buf.seek(0)
    #frames.append(Image.open(buf))
    
    #plt.savefig(os.path.join(path_plot, "density_" + str(i) + "_exp.png"))
    #plt.show()

# frames = []
# for i in range(len(numero_to_plot)):
#     name = "density_" + str(i) + "_exp.png"
#     image = Image.open(os.path.join(path_plot, name))
#     frames.append(image)

frames[0].save(os.path.join(path_plot, "density_all_exp.gif"), save_all = True, append_images = frames[1:], duration = 2000, loop = 0 )

########## Graphique contour
for i in range(len(numero_to_plot)):
    
    data_min = np.min([TC_pdf_array[:,:,i], pdf_array[:,:,i]])
    data_max = np.max([TC_pdf_array[:,:,i], pdf_array[:,:,i]])

    fig, ax = plt.subplots(1, 2, figsize=(9, 5), dpi=200)
    ax[0].contourf(x, x, TC_pdf_array[:,:,i], cmap = "Blues", vmin=data_min, vmax=data_max)
    ax[0].set_xlabel("$x1$")
    ax[0].set_ylabel("$x2$")
    ax[0].set_title("TCNF density | t " + f"{t[numero_to_plot[i]]:.2f}")
        
    ax[1].contourf(x, x, pdf_array[:,:,i], cmap = "Blues", vmin=data_min, vmax=data_max)
    ax[1].set_xlabel("$x1$")
    ax[1].set_title("GT density | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    


########## Graphique moyenne
for j in range(dim_x):
    fig = plt.figure()
    plt.plot(t, GT_mean[:,j], 'C1', label = 'GT-mean')
    plt.plot(t, TC_mean[:,j], 'C2', label = 'TC-mean')
    plt.xlabel('$t$')
    plt.title("Mean value | $X" + str(j+1) + "$")
    plt.legend()


    
########## Graphique sd
for j in range(dim_x):
    fig = plt.figure()
    plt.plot(t, np.sqrt(GT_var_array[:,j]), 'C1', label = 'GT-sd')
    plt.plot(t, TC_sd[:,j], 'C2', label = 'TC-sd')
    plt.xlabel('$t$')
    plt.title("Sd value | $X" + str(j+1) + "$")
    plt.legend()
    
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharey='row')  # Adjust the figsize as needed

# Plot mean values
for j, ax in enumerate(axes[0, :]):
    ax.plot(t, GT_mean[:, j], 'C1', label='GT-mean')
    ax.plot(t, TC_mean[:, j], 'C2', label='TC-mean')
    ax.set_xlabel('$t$')
    ax.set_title("Mean value | $X" + str(j + 1) + "$")
    ax.legend()

# Plot standard deviation values
for j, ax in enumerate(axes[1, :]):
    ax.plot(t, np.sqrt(GT_var_array[:, j]), 'C1', label='GT-sd')
    ax.plot(t, TC_sd[:, j], 'C2', label='TC-sd')
    ax.set_xlabel('$t$')
    ax.set_title("Sd value | $X" + str(j + 1) + "$")
    ax.legend()

# Add space between subplots
plt.tight_layout()

# Show the combined plots
plt.show()


########## Graphique covariance
plt.plot(t, GT_var[:,0,1],'C1', label = "GT-cov")
plt.plot(t, TC_cov_matrix[:,0,1],'C2', label = "TC-cov")
plt.xlabel('$t$')
plt.title("Covariance $X1$ | $X2$")
plt.legend()

########### plot trajectoires
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), sharey='row')  # Adjust the figsize as needed

for i in range(10):
    axes[0].plot(t, test_data[i, 0:N , 0], 'C1')
    axes[1].plot(t, TC_paths[i,: , 0], '--C1')
    axes[0].plot(t, test_data[i, 0:N , 1], 'C2')
    axes[1].plot(t, TC_paths[i,: , 1], '--C2')
axes[0].set_title("GT sample paths")
axes[1].set_title("TCNF sample paths")
plt.tight_layout()



########## Graphique marginal
palette = sns.color_palette("Spectral", 100).as_hex()

# Marginal x
fig, ax = plt.subplots(1, 2, figsize=(9, 5), dpi=200)
im1 = ax[0].imshow(TC_x_marginal, extent = [t[0],t[-1], x[0], x[-1]], cmap='Reds', origin='lower', aspect='auto')
ax[0].plot(t, TC_mean[:,0], color = palette[75], label = 'TC_x')
ax[0].plot(t, TC_quantile_array[0,:,0], '--', color = palette[75])
ax[0].plot(t, TC_quantile_array[1,:,0], '--', color = palette[75])
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$x_1$")
ax[0].set_ylim(-1, 3)
ax[0].set_title("TCNF Marginal $x$1")

im2 = ax[1].imshow(GT_x_marginal, extent=[t[0],t[-1], x[0], x[-1]], cmap='Reds', origin='lower', aspect='auto')
ax[1].plot(t,GT_mean[:,0], color = palette[90], label = 'GT_x')
ax[1].plot(t, quantile_array[0,:,0], '-.', color = palette[90])
ax[1].plot(t, quantile_array[1,:,0], '-.', color = palette[90])
ax[1].set_xlabel("$t$")
ax[1].set_ylim(-1, 3)
ax[1].set_title("GT Marginal $x$1")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)



# Marginal y
fig, ax = plt.subplots(1, 2, figsize=(9, 5), dpi=200)
im1 = ax[0].imshow(TC_y_marginal, extent = [t[0],t[-1], x[0], x[-1]], cmap='Reds', origin='lower', aspect='auto')
ax[0].plot(t, TC_mean[:,1], color = palette[75], label = 'TC_y')
ax[0].plot(t, TC_quantile_array[0,:,1], '--', color = palette[75])
ax[0].plot(t, TC_quantile_array[1,:,1], '--', color = palette[75])
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$x_2$")
ax[0].set_ylim(-1, 3)
ax[0].set_title("TCNF Marginal $x2$")

im2 = ax[1].imshow(GT_y_marginal, extent=[t[0],t[-1], x[0], x[-1]], cmap='Reds', origin='lower', aspect='auto')
ax[1].plot(t,GT_mean[:,1], color = palette[90], label = 'GT_y')
ax[1].plot(t, quantile_array[0,:,1], '-.', color = palette[90])
ax[1].plot(t, quantile_array[1,:,1], '-.', color = palette[90])
ax[1].set_xlabel("$t$")
ax[1].set_ylim(-1, 3)
ax[1].set_title("GT Marginal $x2$")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)

#path_plot = 'C:/Users/naouf/Documents/0-These/2-Scripts/10-Sto_norm_flow/5-TCNF/plots/ou_multi_64_128_64_K3_N3/'


#### plot de marginales
frames = []
for i in range(len(numero_to_plot)):
    fig = plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(x, GT_x_marginal[:,i], 'C1', label = "GT_$p_X1$")
    plt.plot(x, TC_x_marginal[:,i], 'C2', label = 'TC_$p_X1$')
    plt.xlabel('$x1$')
    plt.title("Marginal $X1$ | t " + f"{t[numero_to_plot[i]]:.2f}")
    plt.legend()
    # Convert the plot to an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(Image.open(buf))
#frames[0].save(os.path.join(path_plot, "marginal_x1_exp.gif"), save_all = True, append_images = frames[1:], duration = 2000, loop = 0 )

frames = []
for i in range(len(numero_to_plot)):
    fig = plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(x, GT_y_marginal[:,i], 'C1', label = "GT_$p_X2$")
    plt.plot(x, TC_y_marginal[:,i], 'C2', label = 'TC_$p_X2$')
    plt.xlabel('$x2$')
    plt.title("Marginal $X2$ | t " + f"{t[numero_to_plot[i]]:.2f}")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(Image.open(buf))
#frames[0].save(os.path.join(path_plot, "marginal_x2_exp.gif"), save_all = True, append_images = frames[1:], duration = 2000, loop = 0 )




frames = []
for j in range(N_print_pdf):
    name = "density_j_" + str(j) + ".jpg"
    #print(name)
    image = Image.open(os.path.join(path_plot, name))
    frames.append(image)

frames[0].save(os.path.join(path_plot, "density_all.gif"), save_all = True, append_images = frames[1:], duration = 500, loop = 0 )

        
        
        
for j in range(dim_x):
    plt.imshow(TC_pdf_emp_array[:,:,j], extent = [t[0],t[-1],x[0],x[-1]], cmap = 'viridis', origin = 'lower', aspect='auto')
    plt.plot(t, TC_mean[:,j], 'C1')
    plt.plot(t, TC_quantile_array[0,:,j], '--C1')
    plt.plot(t, TC_quantile_array[1,:,j], '--C1')
    plt.title("Model marginal :" + str(j))
    plt.ylim(top = 2, bottom = -0.5)
    plt.colorbar()
    plt.xlabel("$t$")
    plt.show()

for j in range(dim_x):
    cs = plt.contourf(TT, XX, TC_pdf_emp_array[:,:,j], levels = 200, cmap = "Blues", vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel("$t$")
    plt.plot(t, TC_mean[:,j],'C1')
    plt.plot(t, TC_quantile_array[0,:,j], '--C1')
    plt.plot(t, TC_quantile_array[1,:,j], '--C1')
    plt.ylim(top = 2.5, bottom = -0.5)
    plt.title(r'$F_{\A}(W_{\phi(t)}; \phi(t))$ ' + "density")
    plt.show()
    

# graphes commun
###########################


### Lecture data CTFP
def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

#### Load CTFP
path_global = 'C:/Users/naouf/Documents/0-These/2-Scripts/10-Sto_norm_flow/5-TCNF/data/1-CTFP_metriques/OU_Multi_Sym/'
path_ctfp_pdf = path_global + 'ctfp_pdf_array.pkl'
path_ctfp_mean = path_global + 'ctfp_mean.pkl'
path_ctfp_sd = path_global + 'ctfp_sd.pkl'
path_ctfp_marg = path_global + 'ctfp_marg.pkl'
path_ctfp_cov = path_global + 'ctfp_cov.pkl'

ctfp_pdf = load_object(path_ctfp_pdf)
ctfp_mean = load_object(path_ctfp_mean)
ctfp_sd = load_object(path_ctfp_sd)
ctfp_marg = load_object(path_ctfp_marg)
ctfp_cov = load_object(path_ctfp_cov)

#### load TCNF
path_global = 'C:/Users/naouf/Documents/0-These/2-Scripts/10-Sto_norm_flow/5-TCNF/data/0-TCNF_metriques/OU_Multi_Sym/'
path_tc_pdf = path_global + 'tcnf_pdf_array.pkl'
path_tc_mean = path_global + 'tcnf_mean.pkl'
path_tc_sd = path_global + 'tcnf_sd.pkl'
path_tc_marg = path_global + 'tcnf_marg.pkl'
path_tc_cov = path_global + 'tcnf_cov.pkl'

TC_pdf_array = load_object(path_tc_pdf)
tc_mean = load_object(path_tc_mean)
tc_sd = load_object(path_tc_sd)
TC_marginal = load_object(path_tc_marg)
tc_cov = load_object(path_tc_cov)

# density
for i in range(len(numero_to_plot)):
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), dpi=200)
    
    # TCNF density
    im1 = ax[0].imshow(TC_pdf_array[:,:,i], extent=[x[0], x[-1], x[0], x[-1]], cmap='viridis', origin='lower', aspect='auto')
    ax[0].set_xlabel("$x1$")
    ax[0].set_ylabel("$x2$")
    ax[0].set_title("TCNF density | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    # GT density
    im2 = ax[1].imshow(pdf_array[:,:,i], extent=[x[0], x[-1], x[0], x[-1]], cmap='viridis', origin='lower', aspect='auto')
    ax[1].set_xlabel("$x1$")
    #ax[0].set_ylabel("$x2$")
    ax[1].set_title("GT density | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    # CTFP
    im3 = ax[2].imshow(ctfp_pdf[:,:,i], extent=[x[0], x[-1], x[0], x[-1]], cmap='viridis', origin='lower', aspect='auto')
    ax[2].set_xlabel("$x1$")
    ax[0].set_ylabel("$x2$")
    ax[2].set_title("CTFP density | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    # Create a common colorbar
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax)
    
    
    # ### errors
    # fig, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
    # # TCNF density
    # im1 = ax[0].imshow(np.abs(TC_pdf_array[:,:,i]- pdf_array[:,:,i]), extent=[x[0], x[-1], x[0], x[-1]], cmap='viridis', origin='lower', aspect='auto')
    # ax[0].set_xlabel("$x1$")
    # ax[0].set_ylabel("$x2$")
    # ax[0].set_title("Error TCNF | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    # # CTFP
    # im2 = ax[1].imshow(np.abs(ctfp_pdf[:,:,i] -pdf_array[:,:,i]), extent=[x[0], x[-1], x[0], x[-1]], cmap='viridis', origin='lower', aspect='auto')
    # ax[1].set_xlabel("$x1$")
    # ax[1].set_title("Error CTFP | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    # divider = make_axes_locatable(ax[1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im2, cax=cax)
    
##### contour
for i in range(len(numero_to_plot)):
    # fig, ax = plt.subplots(1, 3, figsize=(9, 3), dpi=200)
    
    # data_min = np.min([TC_pdf_array[:,:,i], pdf_array[:,:,i], ctfp_pdf[:,:,i]])
    # data_max = np.max([TC_pdf_array[:,:,i], pdf_array[:,:,i], ctfp_pdf[:,:,i]])
    
    # # TCNF density
    # ax[0].contourf(x, x, TC_pdf_array[:,:,i], cmap = "Blues", vmin=data_min, vmax=data_max)
    # ax[0].set_xlabel("$x1$")
    # ax[0].set_ylabel("$x2$")
    # ax[0].set_title("TCNF density | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    # # GT density
    # ax[1].contourf(x, x, pdf_array[:,:,i], cmap = "Blues", vmin=data_min, vmax=data_max)
    # ax[1].set_xlabel("$x1$")
    # ax[1].set_title("GT density | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    # # CTFP
    # ax[2].contourf(x, x, ctfp_pdf[:,:,i], cmap = "Blues", vmin=data_min, vmax=data_max)
    # ax[2].set_xlabel("$x1$")
    # ax[2].set_title("CTFP density | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    ### errors
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
    
    data_min = np.min([np.abs(TC_pdf_array[:,:,i] - pdf_array[:,:,i]), np.abs(pdf_array[:,:,i] - ctfp_pdf[:,:,i])])
    data_max = np.max([np.abs(TC_pdf_array[:,:,i] - pdf_array[:,:,i]), np.abs(pdf_array[:,:,i] - ctfp_pdf[:,:,i])])
    
    # TCNF density
    ax[0].contourf(x, x, np.abs(TC_pdf_array[:,:,i]- pdf_array[:,:,i]), cmap = "Blues", vmin=data_min, vmax=data_max)
    ax[0].set_xlabel("$x1$")
    ax[0].set_ylabel("$x2$")
    ax[0].set_title("Error TCNF | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    # CTFP
    ax[1].contourf(x,x, np.abs(ctfp_pdf[:,:,i] -pdf_array[:,:,i]), cmap = "Blues", vmin=data_min, vmax=data_max)
    ax[1].set_xlabel("$x1$")
    ax[1].set_title("Error CTFP | t " + f"{t[numero_to_plot[i]]:.2f}")
    
    
    
for i in range(len(numero_to_plot)):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=200)
    
    # x marginal
    ax[0].plot(x,GT_marginal[:,i, 0], 'C1', label = "GT_$p_{X1}$")
    ax[0].plot(x,TC_marginal[:,i, 0], ':C2', label = "TC_$p_{X1}$")
    ax[0].plot(x,ctfp_marg[:,i, 0], '--C3', label = "CTFP_$p_{X1}$")
    ax[0].set_xlabel("$x1$")
    ax[0].set_title("Marginal $X1$ | t " + f"{t[numero_to_plot[i]]:.2f}")
    ax[0].legend()
    
    # Y marginal
    ax[1].plot(x,GT_marginal[:,i, 1], 'C1', label = "GT_$p_{X2}$")
    ax[1].plot(x,TC_marginal[:,i, 1], ':C2', label = "TC_$p_{X2}$")
    ax[1].plot(x,ctfp_marg[:,i, 1], '--C3', label = "CTFP_$p_{X2}$")
    ax[1].set_xlabel("$x2$")
    ax[1].set_title("Marginal $X2$ | t " + f"{t[numero_to_plot[i]]:.2f}")
    
## Moyenne & Sd
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharey='row')  # Adjust the figsize as needed

    # Plot mean values
    for j, ax in enumerate(axes[0, :]):
        ax.plot(t, GT_mean[:, j], 'C1', label='GT')
        ax.plot(t, TC_mean[:, j], ':C2', label='TC')
        ax.plot(t, ctfp_mean[:, j], '--C3', label='ctfp')
        ax.set_xlabel('$t$')
        ax.set_title("Mean value | $X" + str(j + 1) + "$")
        ax.legend()

    # Plot standard deviation values
    for j, ax in enumerate(axes[1, :]):
        ax.plot(t, np.sqrt(GT_var_array[:, j]), 'C1', label='GT')
        ax.plot(t, TC_sd[:, j], ':C2', label='TC')
        ax.plot(t, ctfp_sd[:, j], '--C3', label='ctfp')
        ax.set_xlabel('$t$')
        ax.set_title("Sd value | $X" + str(j + 1) + "$")
        ax.legend()

    # Add space between subplots
    plt.tight_layout()

    # Show the combined plots
    plt.show()
    
#### Covariance

plt.plot(t, GT_var[:,0,1],'C1', label = "GT")
plt.plot(t, TC_cov_matrix[:,0,1],':C2', label = "TCNF")
plt.plot(t, ctfp_cov[:,0,1],'--C3', label = "CTFP")
plt.xlabel('$t$')
plt.title("Covariance $X1$ | $X2$")
plt.legend()

##### KL divergence

eps = 1e-10
KL_TCNF = entropy((pdf_array + eps), (TC_pdf_array + eps) , axis = (0,1))
KL_CTFP = entropy((pdf_array + eps), (ctfp_pdf + eps) , axis = (0,1))

plt.scatter(t[numero_to_plot], KL_TCNF, label = 'TC')
plt.scatter(t[numero_to_plot], KL_CTFP,label = 'CTFP')
plt.title('KL divergence')
plt.legend()

##### Mean error
TC_error_l1 = np.mean(np.abs(TC_pdf_array - pdf_array), axis = (0, 1))
CTFP_error_l1 = np.mean(np.abs(ctfp_pdf - pdf_array), axis = (0, 1))

TC_error_l2 = np.mean(np.square(TC_pdf_array - pdf_array), axis = (0, 1))
CTFP_error_l2 = np.mean(np.square(ctfp_pdf - pdf_array), axis = (0, 1))

TC_mre_l2 = np.sum(np.square(TC_pdf_array - pdf_array), axis = (0,1))/np.sum(np.square(pdf_array), axis = (0,1))
CTFP_mre_l2 = np.sum(np.square(ctfp_pdf - pdf_array), axis = (0,1))/np.sum(np.square(pdf_array), axis = (0,1))

plt.scatter(t[numero_to_plot], TC_error_l1, label = 'TCNF')
plt.scatter(t[numero_to_plot], CTFP_error_l1, label = 'CTFP')
plt.xlabel('$t$')
plt.ylabel("Mean Abs Error")
plt.title("Mean Abs Error TCNF vs CTFP")
plt.legend()

plt.scatter(t[numero_to_plot], TC_error_l2, label = 'TCNF')
plt.scatter(t[numero_to_plot], CTFP_error_l2, label = 'CTFP')
plt.xlabel('$t$')
plt.ylabel("MSE")
plt.title("Mean Square Error TCNF vs CTFP")
plt.legend()


plt.scatter(t[numero_to_plot], TC_mre_l2, label = 'TCNF')
plt.scatter(t[numero_to_plot], CTFP_mre_l2, label = 'CTFP')
plt.xlabel('$t$')
plt.ylabel("Mean Relative Error")
plt.title("Mean Relative Error TCNF vs CTFP")
plt.legend()