import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 
#%%
def speed_pos_ratemaps(pos, x):
    x_min = 0
    x_max = pos.max().item()
    nbins = 30
    batch = pos.shape[0]
    
    space = np.zeros((N,batch, nbins))
    for n in range(N):
        
        #ax = axes.flat[n]
        for b in range(batch):
            hist_1 = np.zeros(nbins)
            hist_2 = np.zeros(nbins)
            for k in range(10):
                dither = np.random.normal(0,0.05, size = tsteps)
                hist1_r, bins = np.histogram(pos[b,:]+dither, weights = x[b,n,:], range = [x_min, x_max], bins = nbins)
                hist2_r, bins = np.histogram(pos[b,:]+dither, range = [x_min, x_max], bins = nbins)                
                hist_1 += hist1_r
                hist_2 += hist2_r

            space[n,b] = hist_1/hist_2
            
    return space

eps = 1e-5

def softmax(x):
    return np.exp(x) / (np.exp(x).sum() + eps)

func = softmax
#%% GENERATE MODEL and show examples

N = 129
s = 25
A_0 = 4.2
f = 4.5
tsteps = 30
beta = 17
batch_size = 100
D = 0.2
vs = np.linspace(0, 1,batch_size)
v_in = -0.15    


fs = [-f, 0, f]
As = [1,9.5,1]

rang = np.arange(N)
diff_C = np.abs(rang[:, None] - rang[None, :])
circ_diff_C = np.minimum(diff_C, N - diff_C)
W_C = A_0*np.exp(-(circ_diff_C/s)**2)

W = np.zeros_like(W_C)

for i in range(N):
    W[i] = As[i%3]*np.roll(W_C[i],fs[i%3])
    
W = W.T
L,R,V = np.zeros((3,N))

d = np.ones(N)*D

R[::3] = -D
L[2::3] = -D
V[1::3] = v_in

list_x = np.zeros((batch_size, N,tsteps))

for b in range(batch_size):
    x = np.zeros(N)
    x[1] = 1
    list_x[b,:,0] = x
    
    for t in range(1,tsteps):
        h = W@x + R + V*vs[b]
        x = func(beta*h)
        list_x[b,:,t] = x
        


fig, axes = plt.subplots(1,3)

ax0 = axes[0].imshow(W)
axes[0].set_xticks([])
axes[0].set_yticks([])
fig.colorbar(ax0, ax = axes[0], orientation='horizontal')

ax1 = axes[1].imshow(list_x[0,:20])
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.colorbar(ax1, ax = axes[1], orientation='horizontal')

ax2 = axes[2].imshow(list_x[10,:20])
axes[2].set_xticks([])
axes[2].set_yticks([])
fig.colorbar(ax2, ax = axes[2], orientation='horizontal')

dt = 1
pos = np.cumsum(np.repeat((vs[:,None]*dt),tsteps, axis = 1), axis = 1)
ratemaps = speed_pos_ratemaps(pos, list_x)


time = np.zeros((N, batch_size, tsteps))
for n in range(N):

    for b in range(batch_size):
        time[n,b] = list_x[b,n,:]
    
#%% RATEMAPS
cells = [11,17,32]

vmax_r = np.nan_to_num(ratemaps[cells]).max()
vmax_t = time[cells].max()

fig, axes = plt.subplots(2,len(cells), sharex = True)
for i in range(len(cells)):
    ax1 = axes[0,i]
    im1 = ax1.imshow(ratemaps[cells[i]], aspect = 0.5, origin = 'lower', vmax = vmax_r)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2 = axes[1,i]
    im2 = ax2.imshow(time[cells[i]], aspect = 0.5, origin = 'lower', vmax = vmax_t)
    ax2.set_xticks([])
    ax2.set_yticks([])

fig.colorbar(im1, ax=axes[0, :], shrink=1)
fig.colorbar(im2, ax=axes[1, :], shrink=1)

#%% ATTRACTOR SPEED

n_trials = 127
list_x = np.zeros((batch_size, n_trials, N, tsteps))

for k in range(n_trials):
    for b in range(batch_size):
        x = np.zeros(N)
        x[k] = 1
        list_x[b,k,:,0] = x
        
        for t in range(1,tsteps):
            h = W@x + R + V*vs[b]
            x = func(beta*h)
            list_x[b,k,:,t] = x
                        
list_x = list_x/list_x.max(axis = (0,1,3), keepdims = True)
diffs = np.linalg.norm(np.diff(list_x[:,:,:,1:], axis = 3), axis = 2).mean(axis=(1,2))

plt.figure()
plt.plot(vs,diffs)
plt.ylim(bottom = 0, top = 0.7)
plt.xlabel('V')
plt.ylabel(r'$\Delta x$')
