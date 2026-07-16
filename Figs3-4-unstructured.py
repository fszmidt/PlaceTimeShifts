import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import Functions
from mpl_toolkits import mplot3d
model_names = os.listdir('Simulations')
print(model_names)

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 
#%%
modelo = 0  #0 for space model
data = torch.load('Simulations/'+model_names[modelo])
print(model_names[modelo])


pos = data['pos']
v = data['ins'][:,:,0]
targets_t = data['targets'][:,:,-1]
firing = data['firing']
W = data['W'].numpy()
I = data['I'].numpy()
batch_size, hidden_size, tsteps = firing.shape
L = 2*np.pi
ins = data['ins']
input_size = ins.shape[-1]
times = np.tile(np.arange(tsteps), reps = (batch_size,1))
hd = (data['ins'][:,:,1]-1/2)*2

W_0 = W.max(axis = 0)
O = data['O'].numpy()

#%% sorted cells from analyze_model_space_unstructured

cells_der_in = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False,  True, False, False, False,
        True, False, False, False, False, False, False,  True, False,
       False, False, False, False, False,  True,  True, False, False,
       False, False,  True, False,  True, False, False, False, False,
        True, False, False, False, False, False,  True, False,  True,
       False, False, False, False, False, False, False, False, False,
       False,  True, False, False, False, False,  True,  True, False,
        True, False, False, False, False, False,  True, False,  True,
       False,  True, False, False, False,  True, False, False, False,
       False, False, False, False, False,  True, False, False, False,
       False, False, False, False, False, False, False,  True, False,
        True, False, False, False, False,  True, False, False, False,
       False,  True, False, False,  True, False, False, False, False,
       False,  True])

cells_izq_in = np.array([False,  True, False,  True, False, False, False,  True, False,
       False, False, False,  True, False, False, False, False, False,
       False, False, False, False, False, False, False, False,  True,
       False,  True, False, False,  True, False, False,  True, False,
       False, False, False, False, False, False, False, False, False,
       False,  True, False, False, False, False, False, False, False,
       False, False,  True, False, False, False, False, False,  True,
       False, False, False, False, False, False, False, False, False,
       False,  True,  True,  True, False, False, False, False, False,
        True, False, False,  True, False, False, False, False, False,
       False, False,  True,  True,  True, False, False,  True,  True,
       False,  True, False, False, False,  True, False, False, False,
       False, False, False, False,  True, False,  True, False, False,
       False, False, False, False, False,  True, False, False, False,
       False, False])

cells_c_in = (~cells_der_in)*(~cells_izq_in)


cw_color = '#2278B5'
ccw_color = '#E5A024'
c_color = '#009E73'
#%% calculates ratemaps in unstructured task

nbins = 50
ratemaps = np.zeros((hidden_size,nbins))

Z = np.swapaxes(firing[:batch_size,:,:],1,2).reshape(batch_size*(tsteps), hidden_size)
pos_flat = pos[:batch_size,:].flatten()

hist_2, bins = np.histogram(pos_flat, range = [0, L], bins = nbins)

for n in range(hidden_size):
        hist, x_edges = np.histogram(pos_flat,bins = nbins, range=[0, L],weights = Z[:,n])
        ratemaps[n,:] += np.nan_to_num(hist/hist_2)

fig, axes = plt.subplots(16,8,figsize=(10, 10), constrained_layout=True)
for n in range(hidden_size):
    ax = axes.flat[n]
    ax.plot(ratemaps[n])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(np.round(Z[:,n].max(), decimals = 4))
    #ax.set_title(n)

#%% theta_max
max_centers = np.argmax(ratemaps, axis = 1)

maxes = np.max(ratemaps, axis = 1)
plt.hist(max_centers, bins = 25)
plt.xlabel('Ángulos')
plt.ylabel('Ocurrencias')

sp.stats.kstest(max_centers/nbins, 'uniform')
#%% generates Sup9
print(sp.stats.kstest(max_centers[cells_c_in]/nbins, 'uniform'))
print(sp.stats.kstest(max_centers[cells_der_in]/nbins, 'uniform'))
print(sp.stats.kstest(max_centers[cells_izq_in]/nbins, 'uniform'))

fig, axes = plt.subplots(1,3)
axes[0].hist(2*np.pi*max_centers[cells_der_in]/nbins, color = cw_color, bins = 9, density = True, range = [0,2*np.pi])
axes[1].hist(2*np.pi*max_centers[cells_c_in]/nbins, color = c_color, bins = 9, density = True, range = [0,2*np.pi])
axes[2].hist(2*np.pi*max_centers[cells_izq_in]/nbins, color = ccw_color, bins = 9, density = True, range = [0,2*np.pi])

fig.supylabel('Density')
fig.supxlabel('Position')
#%% FIG 3.c
order = np.argsort(max_centers)

plt.imshow(ratemaps[order])
plt.xlabel('Position')
plt.ylabel('Cell')
plt.xticks([])
plt.yticks([])
plt.colorbar()

#%% FIG 4.a

w_sorted = W[order]
w_sorted = w_sorted[:,order]

plt.figure()
plt.imshow(w_sorted)
plt.xticks([])
plt.yticks([])
plt.colorbar()
#%% FIG 4.c
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(I[cells_der_in,0],I[cells_der_in,1],I[cells_der_in,2],'o', color = cw_color)
ax.plot(I[cells_izq_in,0],I[cells_izq_in,1],I[cells_izq_in,2],'o', color = ccw_color)
ax.plot(I[cells_c_in,0],I[cells_c_in,1],I[cells_c_in,2],'o', color = c_color)
ax.set_xlabel('Speed input')
ax.set_ylabel('Clockwise direction input')
ax.set_zlabel('Counterclockwise direction input')

#%% sub-population input medians
names_1 = ['w_speed','w_cw','w_ccw']
names_2 = ['CCW','Central','CW']

pobs = [cells_izq_in, cells_c_in, cells_der_in]

for j in range(3):
    for i in range(3):
        print(names_1[i] + ' ' + names_2[j] + ' = ' + str(np.round(np.median(I[pobs[j],i]),3)))

#%% FIG 4.e

plt.plot(O[0,order],'o',color = '#6A3D9A' , label = 'Sin weights')
plt.plot(O[1,order],'o',color = '#8C564B', label = 'Cos weights')

plt.xlabel('Sorted cells')
plt.ylabel('Output weight')
plt.legend()

out_angles = (np.arctan2(O[0],O[1]))%(2*np.pi)

# fig, ax = plt.subplots()
# ax.scatter(O[0],O[1], c = out_angles, cmap = 'viridis')
# ax.set_xlabel(r'$O_1$')
# ax.set_ylabel(r'$O_2$')
# ax.set_aspect('equal', adjustable='box')
#%% FIG 4.f
max_angles = 2*np.pi*max_centers/nbins

spearman_1 = sp.spearmanr(out_angles[order], max_angles[order])

plt.scatter(out_angles[order], max_angles[order], color ='#4D4D4D')
plt.plot([],alpha = 0, label = r'$\rho = $' + str(np.round(spearman_1[0],1)) + r', $p = $' + f"{spearman_1[1]:.0e}")
plt.xlabel(r'$\theta_{out}$')
plt.ylabel(r'$\theta_{max}$')
plt.legend()
#%% 

W_rolled = np.zeros((hidden_size,hidden_size))
coms = Functions.periodic_center_of_mass(w_sorted.T)
shift = coms - np.arange(hidden_size)

difs = (out_angles-max_angles)
#%% Center of mass shift median and test for each sub-population

print(sp.wilcoxon(shift[cells_der_in[order]]))
print(sp.wilcoxon(shift[cells_izq_in[order]]))
print(sp.wilcoxon(shift[cells_c_in[order]]))

print(np.median(shift[cells_der_in[order]]))
print(np.median(shift[cells_izq_in[order]]))
print(np.median(shift[cells_c_in[order]]))

bins = 28

fig, ax = plt.subplots()
ax.hist(shift[cells_der_in[order]], range = [-15,15], bins = bins, zorder = 5, density = True, color = cw_color)
ax.hist(shift[cells_izq_in[order]],range = [-15,15], bins = bins, zorder = 6,alpha = 0.5, density = True, color = ccw_color)
ax.hist(shift[cells_c_in[order]],range = [-15,15], bins = bins, zorder = 7,alpha = 0.5, density = True, color = c_color)
ax.set_xlabel('Shift')
ax.set_ylabel('Counts')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#%% FIG 4.g 

shift_ins = I[order,1] - I[order,2]

plt.scatter(shift_ins[cells_der_in[order]],difs[cells_der_in],color = cw_color)
plt.scatter(shift_ins[cells_izq_in[order]],difs[cells_izq_in], color= ccw_color)
plt.scatter(shift_ins[cells_c_in[order]],difs[cells_c_in], color = c_color)
spearman_3 = sp.spearmanr(shift_ins, difs[order])


plt.xlabel('CW - CCW input')
plt.ylabel(r'$\theta_{out} - \theta_{max}$')
plt.plot([],alpha = 0, label = r'$\rho = $' + str(np.round(spearman_3[0],1)) + r', $p = $' + f"{spearman_3[1]:.0e}")
plt.legend()
#%% FIG 4.d
plt.figure()
plt.scatter(shift_ins[cells_der_in[order]], shift[cells_der_in[order]], color = cw_color)
plt.scatter(shift_ins[cells_izq_in[order]], shift[cells_izq_in[order]], color = ccw_color)
plt.scatter(shift_ins[cells_c_in[order]], shift[cells_c_in[order]], color = c_color)
plt.xlabel(r'$W_{CW} - W_{CCW}$')
plt.ylabel('Shift')

spearman_4 = sp.spearmanr(shift_ins, shift)
plt.plot([],alpha = 0, label = r'$\rho = $' + str(np.round(spearman_4[0],1)) + r', $p = $' + f"{spearman_4[1]:.0e}")
plt.legend()


#%% Fig 3.E 

Z = np.swapaxes(firing[:,:,:],1,2).reshape(batch_size*(tsteps), hidden_size)
v_flat = ins[:,:,0].flatten()

time_flat = times.flatten()
hist2, x_edges2, y_edges2 = np.histogram2d(v_flat,time_flat,bins = 20, range=[[0, 1.5], [0, 20]])
maps_vt = np.zeros((hidden_size,20,20))

for j in range(hidden_size):
    hist, x_edges, y_edges = np.histogram2d(v_flat,time_flat,bins = 20, range=[[0, 1.5], [0, 20]],weights=Z[:,j])
    maps_vt[j] = hist/hist2


v_flat = ins[:,:,0].flatten()

hist2, x_edges2, y_edges2 = np.histogram2d(v_flat,pos_flat,bins = 20, range=[[0, 1.5], [0, L]])
maps_rv = np.zeros((hidden_size,20,20))

for j in range(hidden_size):
    hist, x_edges, y_edges = np.histogram2d(v_flat,pos_flat,bins = 20, range=[[0, 1.5], [0, L]],weights=Z[:,j])
    maps_rv[j] = hist/hist2


cells = [1,3,14,19]
fig, axes = plt.subplots(2,4)

for i in range(4):
    ax1 = axes[0,i]
    ax1.imshow(maps_rv[cells[i]])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('N = ' +str(cells[i]))
    
    ax2 = axes[1,i]
    ax2.imshow(maps_vt[cells[i]])
    ax2.set_xticks([])
    ax2.set_yticks([])
    
fig.supylabel('Speed')