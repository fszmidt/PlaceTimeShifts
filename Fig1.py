import numpy as np
import matplotlib.pyplot as plt
#%%

def gauss(x, x0, sigma):
    return np.exp(-(x-x0)**2/sigma**2)

#%% PLACE CELL
x0 = np.pi/2
sigma = 0.2

batch_size = 100
tsteps = 100

L = 2*np.pi

v = np.linspace(0,1, batch_size)
v = np.repeat(v[:,None], tsteps, axis = 1)

dt = 0.1

pos = np.cumsum((v*dt), axis = 1)

firing_t_pc = np.zeros((batch_size,tsteps))
times = np.zeros((batch_size,tsteps))
for b in range(batch_size):
    for t in range(tsteps):
        firing_t_pc[b,t] = gauss(pos[b,t], x0, sigma)
        times[b,t] = t
#%% speed pos ratemap
v_flat = v.flatten()
pos_flat = pos.flatten()

nbins = 50
hist2, x_edges2, y_edges2 = np.histogram2d(v_flat,pos_flat,bins = nbins, range=[[0, 1], [0, L]])


hist, x_edges, y_edges = np.histogram2d(v_flat,pos_flat,bins = nbins, range=[[0, 1], [0, L]],weights=firing_t_pc.flatten())
maps_rv_pc = np.nan_to_num(hist/hist2)

#%%  speed time ratemap


time_flat = times.flatten()

nbins = 40
hist2, x_edges2, y_edges2 = np.histogram2d(v_flat,time_flat,bins = nbins, range=[[0, 1], [0, tsteps]])


hist, x_edges, y_edges = np.histogram2d(v_flat,time_flat,bins = nbins, range=[[0, 1], [0, tsteps]],weights=firing_t_pc.flatten())
maps_vt_pc = np.nan_to_num(hist/hist2)    
#%% random trajectories
batch_size = 1000
random_v = np.random.normal(0,1, size = (batch_size,tsteps))
pos_0 = np.random.uniform(0, 2*np.pi, size = batch_size)

pos_random = np.zeros((batch_size,tsteps))
pos_random[:,0] = pos_0


firing_t_pc_random = np.zeros((batch_size, tsteps))

times_random = np.zeros((batch_size,tsteps))
for b in range(batch_size):
    for t in range(1,tsteps):
        pos_random[b,t] = (pos_random[b,t-1] + random_v[b,t]*dt)%L
        firing_t_pc_random[b,t] = gauss(pos_random[b,t], x0, sigma)
        times_random[b,t] = t
#%%

v_flat = random_v.flatten()
pos_flat = pos_random.flatten()

nbins = 50
hist2, x_edges2, y_edges2 = np.histogram2d(v_flat,pos_flat,bins = nbins, range=[[0, 1], [0, L]])

hist, x_edges, y_edges = np.histogram2d(v_flat,pos_flat,bins = nbins, range=[[0, 1], [0, L]],weights=firing_t_pc_random.flatten())
maps_rv_random_pc = np.nan_to_num(hist/hist2)

#%%

time_flat_random = times_random.flatten()

nbins = 50
hist2, x_edges2, y_edges2 = np.histogram2d(v_flat,time_flat_random,bins = nbins, range=[[0, 1], [0, tsteps]])


hist, x_edges, y_edges = np.histogram2d(v_flat,time_flat_random,bins = nbins, range=[[0, 1], [0, tsteps]],weights=firing_t_pc_random.flatten())
maps_vt_random_pc = np.nan_to_num(hist/hist2)


#%% TIME CELL

batch_size = 100
time = np.arange(tsteps)


firing_t_tc = np.zeros((batch_size,tsteps))
for b in range(batch_size):
    for t in range(tsteps):
        firing_t_tc[b,t] = gauss(time[t], tsteps//2, 5)

#%% speed pos ratemap
v_flat = v.flatten()
pos_flat = pos.flatten()

nbins = 50
hist2, x_edges2, y_edges2 = np.histogram2d(v_flat,pos_flat,bins = nbins, range=[[0, 1], [0, L]])


hist, x_edges, y_edges = np.histogram2d(v_flat,pos_flat,bins = nbins, range=[[0, 1], [0, L]],weights=firing_t_tc.flatten())
maps_rv_tc = np.nan_to_num(hist/hist2)
#%% speed time ratemap

time_flat = times.flatten()

nbins = 50
hist2, x_edges2, y_edges2 = np.histogram2d(v_flat,time_flat,bins = nbins, range=[[0, 1], [0, tsteps]])


hist, x_edges, y_edges = np.histogram2d(v_flat,time_flat,bins = nbins, range=[[0, 1], [0, tsteps]],weights=firing_t_tc.flatten())
maps_vt_tc = np.nan_to_num(hist/hist2)
#%% speed pos random

batch_size = 1000

firing_t_tc_random = np.zeros((batch_size,tsteps))

for b in range(batch_size):
    for t in range(tsteps):
        firing_t_tc_random[b,t] = gauss(t, tsteps//2, 5)        


v_flat = random_v.flatten()
pos_flat = pos_random.flatten()


nbins = 50
hist2, x_edges2, y_edges2 = np.histogram2d(v_flat,pos_flat,bins = nbins, range=[[0, 1], [0, L]])

hist, x_edges, y_edges = np.histogram2d(v_flat,pos_flat,bins = nbins, range=[[0, 1], [0, L]],weights=firing_t_tc_random.flatten())
maps_rv_random_tc = np.nan_to_num(hist/hist2)
#%%
nbins = 50
hist2, x_edges2, y_edges2 = np.histogram2d(v_flat,time_flat_random,bins = nbins, range=[[0, 1], [0, tsteps]])


hist, x_edges, y_edges = np.histogram2d(v_flat,time_flat_random,bins = nbins, range=[[0, 1], [0, tsteps]],weights=firing_t_tc_random.flatten())
maps_vt_random_tc = np.nan_to_num(hist/hist2)
#%%

fig, axes = plt.subplots(2,4)
plot_list = [maps_rv_pc,maps_vt_pc,maps_rv_random_pc, maps_vt_random_pc,
                  maps_rv_tc,maps_vt_tc,maps_rv_random_tc,maps_vt_random_tc]

for i in range(len(plot_list)):
    ax = axes.flat[i]
    ax.imshow(plot_list[i])
    ax.set_xticks([])
    ax.set_yticks([])
    
