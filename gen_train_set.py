import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import imageio
img = torch.from_numpy(imageio.v2.imread('visual_pattern.png'))/255

pos_nbins = img.shape[1]

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
    t_vis = np.random.choice(np.arange(1,tsteps), size = (batch_size, 4))
    t_vis[:,0]*=0
    vis = np.zeros((batch_size,tsteps,27))

    for b in range(batch_size):
        for t in t_vis[b]:
            px_0 = pos_bin[b,t] 
            idx = (px_0 + np.arange(0,3)*(-2*hd[b,t,0] + 1))% pos_nbins 
            
            for c in range(3):
                vis[b,t,9*c:9*(c+1)] = img[:,idx.type(torch.int64).numpy(),c].flatten()# + noise[b,t,:,c].flatten()
    return torch.from_numpy(vis)

def sim_space(batch, tsteps, dt,p_0):
    '''
    Generates inputs and targets for the spatial task.

    Parameters
    ----------
    batch : INT
        Number of trials per batch
    tsteps : INT
        Number of time steps per trial
    dt : Float
        Time length of a time step.
    p_0 : Pytorch tensor
        Initial angle for each trial.

    Returns
    -------
    speed : Pytorch tensor, shape = (batch, tsteps,1)
        Absolute value of speed in the simulation (rad/s)
    hd : Pytorch tensor, shape = (batch, tsteps, 2)
        One-hot direction input (idx = 0 is CW, 1 is CCW)
    theta : Pytorch tensor, shape = (batch, tsteps)
        Simulated angle in the simulation
    target : Pytorch tensor, shape = (batch, tsteps, 2)
        Sine and cosine of theta, targets of the model
    '''
    zero_inf = np.random.choice([0,1], size = (batch,1), p = [0.1,0.9]) #prob of v=0
    speed = torch.randn((batch,tsteps))

    hd = torch.zeros(batch, tsteps, 2)
    hd[:,:,0] = (torch.sign(speed) + 1)/2
    hd[:,:,1] = 1 - hd[:,:,0]
    speed = speed*zero_inf
    hd = hd*np.abs(speed.unsqueeze(dim=2))

    theta = (p_0 + torch.cumsum(speed*dt,dim = 1))%(2*np.pi)
    target_space = torch.concatenate((torch.sin(theta).unsqueeze(dim=2), torch.cos(theta).unsqueeze(dim=2)), dim = 2)

    return(speed.unsqueeze(dim = 2).abs(), hd,theta, target_space)

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


def gen_inputs_and_targets(sim_parameters, p_0):
    '''
    

    Parameters
    ----------
    sim_parameters : Dictionary
        Contains parameters for the simulation (batch size, time steps,
        number of pixels of the visual input, min and max timesteps between cues)

    p_0 : Pytorch tensor
        Initial positions for all trials in the space task.

    Returns
    -------
    ins : Pytorch tensor, shape = (batch, tsteps, 31) (1 speed + 2 directions,
          + 27 flattened RGB 3x3 visual pattern + 1 timing cue)
        Tensor of inputs for the network, including spatial and timing tasks.
    theta : Pytorch tensor, shape = (batch, tsteps)
        Simulated angle in the simulation
    targets : Pytorch tensor, shape = (batch, tsteps, 3) (sin(theta), cos(theta)
        time from last cue)
        Targets for both the spatial and timing task.

    '''
    batch, tsteps, dt, pos_nbins, a, b = sim_parameters.values()
    speed, hd, theta, target_space = sim_space(batch, tsteps, dt, p_0)
    cues, target_time = sim_time(a,b,tsteps,batch) 
    vis = calc_vis(theta, pos_nbins, hd.numpy(), batch, tsteps)
    
    ins = torch.concatenate((speed,hd, vis, cues.unsqueeze(dim = 2)), dim = 2).float()
    targets = torch.concatenate((target_space, target_time.unsqueeze(dim=2)), dim = 2).float()

    return ins, theta, targets


batch_size = 100
tsteps = 40 #40 for the space model, 20 for the space+time model
dt = 0.2

a = 2
b = tsteps

sim_parameters = {'batch_size': batch_size, 'tsteps':tsteps,'dt':dt, 'pos_nbins':pos_nbins,
                  'a': a,'b': b}

task = np.random.choice([0,1], size = (batch_size))
p_0 = torch.rand(batch_size,1)*2*np.pi

ins, theta, target = gen_inputs_and_targets(sim_parameters, p_0)

data = {'ins':ins, 'target':target, 'img':img, 'sim_parameters': sim_parameters}
torch.save(data, 'train_set_space_time.pt')