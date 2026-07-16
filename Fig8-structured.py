import gen_test_set_structured
import torch
from model import RNN
import numpy as np
import imageio


from torch.nn import MSELoss
device = 'cpu'
import matplotlib.pyplot as plt
from torch.nn import MSELoss
mse = MSELoss()

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 
import scipy.stats as sp

cw_color = '#2278B5'
ccw_color = '#E5A024'
c_color = '#009E73'
#%%
def speed_pos_ratemaps(pos, x):
    N = x.shape[1]
    nbins = 25
    space = np.zeros((N,batch_size//n_trials, nbins))
    jitter = np.random.normal(0, 2*np.pi/nbins, size = pos.shape)
    for n in range(N):
        for b in range(batch_size//n_trials):
            hist_1 = np.zeros(nbins)
            hist_2 = np.zeros(nbins)
            for j in range(n_trials):
                hist1_r, bins = np.histogram((pos + jitter)[b*n_trials + j,:].numpy(), weights = x[b*n_trials + j,n,:], range = [0, 2*np.pi], bins = nbins)
                hist2_r, bins = np.histogram((pos + jitter)[b*n_trials + j,:].numpy(), range = [0, 2*np.pi], bins = nbins)                
                hist_1 += hist1_r
                hist_2 += hist2_r

            space[n,b] = hist_1/hist_2
    return space

#%%
nombre = 'model_spacetime'
model = torch.load('Models/'+nombre+'.pt', map_location=torch.device(device))
sim_parameters = model.sim

hidden_size = model.hidden_size
act = np.arange(hidden_size)
batch_size = 2048
t0 = 0

sim_parameters['batch_size'] = batch_size


input_size = model.input_size
v1 = 0
v2 = 3
dt = sim_parameters['dt']
model.func = torch.nn.Softmax(dim=1)
#%%

tsteps = 20
t0 = 5

sim_parameters['tsteps'] = tsteps
model.tsteps = tsteps

model.batch_size = batch_size

t_vis = torch.from_numpy(np.zeros((batch_size,1), dtype = 'int'))
n_trials = 16

p_0 = torch.rand(batch_size,1)*2*np.pi

ins_CW, theta_CW, target_CW = gen_test_set_structured.inputs_and_targets_st(sim_parameters, t_vis,0, n_trials, v1, v2,p_0,t0)
ins_CW_st, theta_CW_st, target_CW_st = gen_test_set_structured.inputs_and_targets_st(sim_parameters, t_vis,0, n_trials, v1, v2,0,t0)
ins_CCW, theta_CCW, target_CCW =gen_test_set_structured.inputs_and_targets_st(sim_parameters, t_vis,1, n_trials, v1, v2,p_0,t0)

output_CW,x_CW = model(ins_CW.to(device)[:,:,:input_size])
output_CW_st,x_CW_st = model(ins_CW_st.to(device)[:,:,:input_size])
output_CCW_st,x_CCW = model(ins_CCW.to(device)[:,:,:input_size])
#%% FIG C
x0_max = x_CW[:,:,t0:].max(axis = (0,2))
x1_max = x_CCW[:,:,t0:].max(axis = (0,2))

x0_max = x0_max
x1_max = x1_max
Iv = model.I_r.weight.detach()[:,0]

thres = 0.01

cells_der_in = (x1_max<thres)*(x0_max>thres)
cells_izq_in = (x0_max<thres)*(x1_max>thres)
cells_c_in = (~cells_der_in)*(~cells_izq_in)


plt.scatter(x1_max[cells_der_in],x0_max[cells_der_in], zorder = 5, color = cw_color)
plt.scatter(x1_max[cells_izq_in],x0_max[cells_izq_in], zorder = 5, color = ccw_color)
plt.scatter(x1_max[cells_c_in],x0_max[cells_c_in], zorder = 5, color = c_color)

plt.ylabel('Max firing clockwise')
plt.xlabel('Max firing counterclockwise')
#%%
cells = [3,14,19,32]

ratemaps = speed_pos_ratemaps(theta_CW_st[:,t0:],x_CW_st[:,cells,t0:])


ratemaps_t = np.zeros((4,batch_size//n_trials, tsteps - t0))
for n in range(4):
    #ax = axes.flat[n]
    for b in range(batch_size//n_trials):
        hist1 = np.mean(x_CW_st[b*n_trials:(b+1)*n_trials,cells[n],t0:], axis = 0)
        ratemaps_t[n,b] = hist1
                
#%% FIG 8 a
r_max = np.max(np.nan_to_num(ratemaps))
t_max = np.max(ratemaps_t)

fig, axes = plt.subplots(2,len(cells))

for i in range(4):
    ax1 = axes[0,i]
    im1 = ax1.imshow(ratemaps[i], aspect = 0.55)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('N = ' + str(cells[i]))
    fig.colorbar(im1, ax=ax1, orientation = 'horizontal', shrink = 0.9)    
    
    ax2 = axes[1,i]
    im2 = ax2.imshow(ratemaps_t[i], aspect = 0.35)
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.colorbar(im2, ax=ax2, orientation = 'horizontal', shrink = 0.9)    
    
fig.supylabel('Speed')
