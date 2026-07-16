import gen_test_set_structured
import torch
from model import RNN
import numpy as np

from Functions import pop_means,periodic_center_of_mass, vel_att
from torch.nn import MSELoss
device = 'cpu'
import matplotlib.pyplot as plt
from torch.nn import MSELoss
mse = MSELoss()
import scipy.stats as sp
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#%%
def speed_pos_ratemaps(pos, x):
    N = x.shape[1]
    x_min = 0
    x_max = 2*np.pi
    nbins = 25
    
    space = np.zeros((N,batch_size//n_trials, nbins))
    dither = np.random.normal(0, 1*np.pi/nbins, size = pos.shape)
    for n in range(N):
        for b in range(batch_size//n_trials):
            hist_1 = np.zeros(nbins)
            hist_2 = np.zeros(nbins)
            
            for j in range(n_trials):
                hist1_r, _ = np.histogram((pos + dither)[b*n_trials + j,:].numpy(), weights = x[b*n_trials + j,n,:], range = [x_min, x_max], bins = nbins)
                hist2_r, _ = np.histogram((pos + dither)[b*n_trials + j,:].numpy(), range = [x_min, x_max], bins = nbins)                
                hist_1 += hist1_r
                hist_2 += hist2_r

            space[n,b] = hist_1/hist_2
    return space

def angle_corrs(out):
    out_angle = np.arctan2(output_CW[:,t0:,0].detach(),output_CW[:,t0:,1].detach()) + np.pi
    angle_2 = np.arctan2(out[:,t0:,0].detach(),out[:,t0:,1].detach())
    
    corrs = np.zeros(batch_size)
    rs = np.zeros(batch_size)

    for b in range(batch_size):
        for t in range(tsteps-t0):
            diff = out_angle[b,t] - out_angle[b,t-1]
            diff_2 = angle_2[b,t] - angle_2[b,t-1]
            if diff <-5:
                out_angle[b,t:]+= 2*np.pi
            if diff_2 <-5:
                angle_2[b,t:]+= 2*np.pi
            
            res = sp.linregress(out_angle[b], angle_2[b])
            corrs[b] = res.slope
            rs[b] = res.rvalue
            
    return np.mean(corrs), np.min(rs), np.std(corrs)
#%%
model_name = 'model_space'
model = torch.load('Models/'+model_name+'.pt', map_location=torch.device(device))
sim_parameters = model.sim

hidden_size = model.hidden_size
act = np.arange(hidden_size)
batch_size = 2048

sim_parameters['batch_size'] = batch_size
input_size = model.input_size
v1 = 0
v2 = 3
dt = sim_parameters['dt']
model.func = torch.nn.Softmax(dim=1)

O = model.out.weight.detach()

cw_color = '#2278B5'
ccw_color = '#E5A024'
c_color = '#009E73'
#%%

tsteps = 20
t0 = 5

sim_parameters['tsteps'] = tsteps
model.tsteps = tsteps
model.batch_size = batch_size
t_vis = torch.from_numpy(np.zeros((batch_size,1), dtype = 'int'))
n_trials = 16 #trials con misma velocidad

p_0 = torch.rand(batch_size,1)*2*np.pi

#Generate CW trajectories from random initial positions
ins_CW, theta_CW, target_CW = gen_test_set_structured.inputs_and_targets_st(sim_parameters, t_vis,0, n_trials, v1, v2,p_0,t0)
#Generate structured CW trajectories
ins_CW_st, theta_CW_st, target_CW_st = gen_test_set_structured.inputs_and_targets_st(sim_parameters, t_vis,0, n_trials, v1, v2,0,t0)
#Generate structured CCW trajectories
ins_CCW, theta_CCW, target_CCW = gen_test_set_structured.inputs_and_targets_st(sim_parameters, t_vis,1, n_trials, v1, v2,p_0,t0)

#Evaluate model
output_CW, x_CW = model(ins_CW.to(device)[:,:,:input_size])
output_CW_st, x_CW_st = model(ins_CW_st.to(device)[:,:,:input_size])
output_CCW, x_CCW = model(ins_CCW.to(device)[:,:,:input_size])

vel_real = np.zeros(batch_size//n_trials)
for b in range(batch_size//n_trials):
    vel_real[b] = ins_CW[n_trials*b,-1,0]   
#%% Plot example trial
b = 1024
plt.figure()
plt.plot(target_CW_st[b,t0:,0], '--', linewidth = 5, color = 'gray', label = 'Target')
plt.plot(output_CW_st[b,t0:,0].detach(), 'o-', label = 'Output')

plt.ylim(bottom = -1.1, top = 1.1)
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

plt.figure()
plt.scatter(x1_max[cells_der_in],x0_max[cells_der_in], label = 'CW selective cells',zorder = 5, color = '#2278B5')
plt.scatter(x1_max[cells_izq_in],x0_max[cells_izq_in], label = 'CCW selective cells',zorder = 5, color = '#E5A024')
plt.scatter(x1_max[cells_c_in],x0_max[cells_c_in],label='Non-selective cells', zorder = 5, color = '#009E73')

plt.ylabel('Max firing clockwise')
plt.xlabel('Max firing counterclockwise')
plt.legend()
plt.xlim(left = -0.05,right = 1)
plt.ylim(bottom = -0.05, top = 1)

#%% speed sorted space ratemaps

cells = [40,121, 14,25]

ratemaps_pos = speed_pos_ratemaps(theta_CW_st[:,:],x_CW_st[:,cells,:])

vmax = np.nan_to_num(ratemaps_pos).max()
fig, axes = plt.subplots(2,2)
for i in range(4):
    ax = axes.flat[i]
    ax.imshow(ratemaps_pos[i],aspect = 0.5,vmin = 0, vmax = vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(i)
    
    
#%% speed sorted time ratemaps
fig, axes = plt.subplots(2,2)
ratemaps_t = np.zeros((4,batch_size//n_trials, tsteps-t0))
for n in range(4):
    ax = axes.flat[n]
    for b in range(batch_size//n_trials):
        hist1 = np.mean(x_CW_st[b*n_trials:(b+1)*n_trials,cells[n],t0:], axis = 0)
        ratemaps_t[n,b] = hist1
                
    ax.imshow(ratemaps_t[n], aspect = 0.2)
    ax.set_xticks([])
    ax.set_yticks([])

#%% FIG 3.d

max_r = np.nan_to_num(ratemaps_pos).max()
max_t = ratemaps_t.max()

fig, axes = plt.subplots(2,len(cells))

for i in range(4):
    ax1 = axes[0,i]
    im1 = ax1.imshow(ratemaps_pos[i], aspect = 0.5, vmax = max_r)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('N = ' + str(cells[i]))
    
    ax2 = axes[1,i]
    im2 = ax2.imshow(ratemaps_t[i], aspect = 0.3, vmax = max_t)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
fig.supylabel('Speed')

fig.colorbar(im1, ax=axes[0, :], shrink=0.8)
fig.colorbar(im2, ax=axes[1, :], shrink=0.8)

#%% FIG 3.f
vel_mean = vel_att(x_CW[:,:,t0:], batch_size, n_trials,tsteps - t0,ins_CW)
plt.figure()
plt.plot(vel_real,vel_mean)

plt.xlabel('Real speed')
plt.ylabel('Attractor speed')

#%% FIG 5

percents = [1,0.75,0.5,0.25]
alphas = [1,0.75,0.5,0.25]


lista_corrs = np.zeros(4)
lista_rs = np.zeros(4)
lista_sems = np.zeros(4)
with torch.no_grad():
   model.I_r.weight.data[:,0] -= model.I_r.weight.data[cells_c_in,0].max()
   buffer = model.I_r.weight.data[:,0].clone()

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, axes3 = plt.subplots()
fig4, axes4 = plt.subplots(2,2)

ratemap_list = np.zeros((4,batch_size//n_trials, 25))

for i in range(len(percents)):
    
    with torch.no_grad():
        model.I_r.weight.data[:,0] = buffer
        print(model.I_r.weight.data[0,0])
        model.I_r.weight.data[cells_c_in,0] = model.I_r.weight.data[cells_c_in,0]*percents[i]
        
    output_p,x_p = model(ins_CW.to(device)[:,:,:input_size])
    
    vel_mean = vel_att(x_p[:,:,t0:], batch_size, n_trials,tsteps,ins_CW)
    ax1.plot(vel_real,vel_mean, label = str(percents[i]))
    
    output_p_st,x_p_st = model(ins_CW_st.to(device)[:,:,:input_size])
    ratemap_list[i] = speed_pos_ratemaps(theta_CW_st[:,t0:],x_p_st[:,121:122,t0:])
    
    
    mean_centered, mean_cw, mean_ccw = pop_means(x_p, batch_size, n_trials,tsteps,cells_c_in, cells_der_in,cells_izq_in,t0)
    
    vel_mean_c = vel_att(x_p[:,cells_c_in,t0:], batch_size, n_trials,tsteps - t0,ins_CW)
    vel_mean_der = vel_att(x_p[:,cells_der_in,t0:], batch_size, n_trials,tsteps - t0,ins_CW)
    vel_mean_izq = vel_att(x_p[:,cells_izq_in,t0:], batch_size, n_trials,tsteps - t0,ins_CW)
    
    axes4.flat[i].plot(vel_real,vel_mean_der, color = cw_color)
    axes4.flat[i].plot(vel_real,vel_mean_c, color = c_color)
    axes4.flat[i].plot(vel_real,vel_mean_izq, color = ccw_color)
    axes4.flat[i].set_ylim(bottom = 0,top = 0.75)
    axes3.plot(vel_real,mean_cw, label = 'CW selective cells', alpha = alphas[i], color = cw_color)
    axes3.plot(vel_real,mean_ccw, label = 'CCW selective cells', alpha = alphas[i], color = ccw_color)
    axes3.plot(vel_real,mean_centered, label = 'Non-selective cells', alpha = alphas[i], color = c_color)
    axes3.set_ylim(top = 0.04)
    ax2.plot(output_p_st[-1,t0:,0].detach(), label = str(percents[i]), color = 'black', alpha = alphas[i])
    corrs, rs, sems = angle_corrs(output_p_st)
    lista_corrs[i] = corrs
    lista_sems[i] = sems
    lista_rs[i] = rs
    
    
ax2.legend()
ax1.set_xlabel('Animal speed')
ax1.set_ylabel('Attractor speed')
ax1.legend()


fig5,axes5 = plt.subplots(1,4)
vmax = np.nan_to_num(ratemap_list).max()
for i in range(4):
    ax = axes5.flat[i]
    im = ax.imshow(ratemap_list[i], vmax = vmax, aspect = 0.5)

    ax.set_ylim(ax.set_ylim()[::-1])
    ax.set_xticks([])
    ax.set_yticks([])
    
fig5.colorbar(im, ax=axes5, shrink=0.5)


plt.figure()
plt.bar(np.arange(3),lista_corrs[1:])
plt.errorbar(np.arange(3), lista_corrs[1:], yerr = lista_sems[1:], fmt = 'o', capsize = 5,color = 'black')
plt.xticks(np.arange(3))
plt.ylabel('Mean slope')
