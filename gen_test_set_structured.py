import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import imageio

img = torch.from_numpy(imageio.v2.imread('visual_pattern.png'))/255
pos_nbins = img.shape[1]
#%%

def bin_pos(theta,pos_nbins):
    '''
    Binnarizes position to get corresponding pixel in the visual pattern

    Parameters
    ----------
    theta : Pytorch tensor
        Angle variable in the simulation, shape = (batch, tsteps)
    pos_nbins : INT
        Number of pixels in the image.


    Returns
    -------
    pos_bin : Pytorch tensor
        Binnarized index, shape = (batch,tsteps).

    '''
    bin_size = 2*np.pi/pos_nbins
    pos_bin = torch.clip(theta//bin_size, min = 0, max = pos_nbins-1).to(dtype=torch.int)
    return pos_bin

def calc_vis(theta, pos_nbins,hd,batch_size,tsteps):
    '''
    Selects visual input by binning angle, choosing the corresponding pixel in
    the image, and the corresponding pixels in front, to the movement direction

    Parameters
    ----------
    theta : Pytorch tensor
        Angle variable in the simulation, shape = (batch, tsteps)
    pos_nbins : INT
        Number of pixels in the image.
    hd : Pytorch tensor, shape = (batch, tsteps)
        Head direction in the simulation (0 = CW, 1 = CCW)
    batch_size : INT
        Number of trials per batch.
    tsteps : INT
        Number of time steps per trial.

    Returns
    -------
    vis: Pytorch tensor, shape = (batch, tsteps, 27)
        Flattened RGB 3x3 pixel visual input.
    '''
    pos_bin = bin_pos(theta, pos_nbins)
    
    vis = np.zeros((batch_size,tsteps,27))
    t = 0
    for b in range(batch_size):
        px_0 = pos_bin[b,t] 
        idx = (px_0 + np.arange(0,3)*(-2*hd[b,t,0] + 1))% pos_nbins 
        
        for c in range(3):
            vis[b,t,9*c:9*(c+1)] = img[:,idx.type(torch.int64).numpy(),c].flatten()# + noise[b,t,:,c].flatten()
    return torch.from_numpy(vis)
#%%
def sim_time(a, b, tsteps, batch):
    '''
    Generates inputs and targets for the timing task.
    

    Inputs:

    a : INT
        Minimum time length between cues
    b : INT
        Maximum time length between cues
    tsteps : INT
        Number of time steps per trial
    batch : int
        Number of trials per batch

    Returns
    -------
    cue_list: Pytorch tensor, shape = (batch, tsteps)
        Cue value (0 or 1)
    counts: Pytorch tensor, shape (batch, tsteps)
        Number of timesteps from the last cue
    '''
    
    
    cue_list = np.zeros((batch,tsteps))    
    count_list = np.zeros((batch,tsteps))

    for m in range(batch):
        random = np.random.uniform(a,b,tsteps).astype(int)
        random = random[random>0]

        cumsum = np.cumsum(random)
        cue_idx = cumsum[cumsum < tsteps]

        cue = np.zeros(tsteps)
        cue[cue_idx] = 1
        counts = np.zeros(tsteps)

        for i in range(tsteps):
            if i % tsteps == 0:
                cue[i] = 1

            if cue[i] == 1:
                j = 0
            else:
                j += 1
            counts[i] = j
        
        cue_list[m,:] = cue
        count_list[m,:] = counts
    return(torch.from_numpy(cue_list), torch.from_numpy(count_list))


def inputs_and_targets_st(sim_parameters, t_vis,direction, n_trials, v1, v2,p_0,t0):
    batch, tsteps, dt, v_min, v_max, L, pos_nbins, a,b, dist = sim_parameters.values()
    
    speed = np.linspace(v1,v2, batch//n_trials)*(-2*direction + 1)
    speed = np.repeat(speed[:,None], tsteps, axis = 1) #para todos los tiempos 
    speed = torch.from_numpy(np.repeat(speed, n_trials, axis = 0)) #varios batches con misma velocidad
    
    speed[:,:t0]*=0
    
    hd = np.zeros((batch, tsteps, 2))
    hd[:,:,direction] = 1
    
    ins_t = np.zeros((batch,tsteps))
    

    theta = (p_0 + torch.cumsum(speed*dt,axis = 1))
    vis = calc_vis(theta, pos_nbins, hd,batch, tsteps)
    target = torch.concatenate((torch.sin(theta).unsqueeze(dim=2), torch.cos(theta).unsqueeze(dim=2)), dim = 2)    
    ins = torch.concatenate((speed.unsqueeze(dim=2).abs(), torch.from_numpy(hd), vis, torch.from_numpy(ins_t).unsqueeze(dim=2)), dim = 2).float()
    
    return ins, theta, target      
