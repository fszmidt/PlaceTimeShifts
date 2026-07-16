import numpy as np
import matplotlib.pyplot as plt
import Functions
import os
import torch
import scipy.stats as sp
from scipy.optimize import curve_fit
model_names = os.listdir('Simulations')
matplotlib.rcParams['pdf.fonttype'] = 42 
print(model_names)
#%%
model = 2
data = torch.load('Simulations/'+model_names[model])
print(model_names[model])
#%%
pos = data['pos']
v = data['ins'][:,:,0]
targets_t = data['targets'][:,:,-1]
firing = data['firing']
W = data['W'].numpy()

I = data['I'].numpy()
batch_size, hidden_size, tsteps = firing.shape
L = 2*np.pi
ins = data['ins']
#h = data['pre_firing']
input_size = ins.shape[-1]
times = np.tile(np.arange(tsteps), reps = (batch_size,1))
hd = (data['ins'][:,:,1]-1/2)*2

W_0 = W.max(axis = 1)
O = data['O'].numpy()

#%%
x_mean = firing.mean(axis=0)
x_std = firing.std(axis=0)


max_times = np.argmax(x_mean, axis = 1)
plt.hist(max_times, bins = 20)
plt.xlabel('Time of max firing')

plt.ylabel('Count')
#%% Fig 7.b
sorted_indices = np.argsort(max_times)
sorted_data = x_mean[sorted_indices]/(x_mean[sorted_indices].max(axis=1, keepdims=True))

plt.imshow(sorted_data, aspect = 0.5)
plt.xticks([])
plt.yticks([])
plt.colorbar()
#%% Fig 7.d

cells = [3,14,19,32]

Z = np.swapaxes(firing[:,:,:],1,2).reshape(batch_size*(tsteps), hidden_size)
pos_flat = pos[:,:].flatten()
time_flat = times.flatten()
hist2, x_edges2, y_edges2 = np.histogram2d(pos_flat,time_flat,bins = 20, range=[[0, L], [0, 20]])
maps_rt = np.zeros((hidden_size,20,20))

for j in range(hidden_size):
    hist, x_edges, y_edges = np.histogram2d(pos_flat,time_flat,bins = 20, range=[[0, L], [0, 20]],weights=Z[:,j])
    maps_rt[j] = hist/hist2

fig, axes = plt.subplots(1,4)
for i in range(4):
    ax = axes.flat[i]
    im = ax.imshow(maps_rt[cells[i]])
    #ax.set_title(i)
    ax.set_title('N = ' + str(cells[i]))
    #ax.set_title(np.round(space_std[i],3))
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax = axes)
#%%
width_t = np.zeros(hidden_size)

for i in range(hidden_size):    
   width_t[i] = (x_mean[i] > (x_mean[i].max()/2)).sum()

#%% Fig 7.i

plt.scatter(max_times, width_t)
plt.xlabel('Time of max firing')
plt.ylabel('Field width')

#%% Sorted W (Fig 7.e)
order_t = np.argsort(max_times)

w_sorted_time = W[order_t]
w_sorted_time = w_sorted_time[:,order_t]

plt.figure()
plt.imshow(w_sorted_time)
plt.xticks([])
plt.yticks([])
plt.colorbar()
#%% Fig 8.c

plt.scatter(np.arange(hidden_size),I[order_t,-1])

#%% Fig 7.f

coms = Functions.periodic_center_of_mass(w_sorted_time.T)
shift = coms - np.arange(hidden_size)

plt.figure()

plt.hist(shift)
plt.xlabel('Recurrent shift')
plt.ylabel('Cell count')
print(np.median(shift))
print(sp.wilcoxon(shift))
#%% Fig 7.h
plt.scatter(max_times, O[2])
plt.xlabel('Time of max firing')
plt.ylabel('Output weight')

print(sp.spearmanr(max_times,O[2]))