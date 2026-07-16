from model import RNN
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



data = torch.load('train_set_space_time.pt', weights_only = False) #CAMBIAR

targets_data = data['target']
ins_data = data['ins']
sim_parameters = data['sim_parameters']
tot_size = ins_data.shape[0]
batch_size = 64
img = data['img']
pos_nbins = img.shape[1]

tsteps = sim_parameters['tsteps']
def get_ins_targets(batch_size):
    '''
    Selects a random subset of trials of size batch_size

    Parameters
    ----------
    batch_size : INT
        Number of trials per minibatch.

    Returns
    -------
    targets_batch : Pytorch tensor, shape = (batch_size, tsteps, 3)
        Selected trials for the minibatch.
    ins_batch : Pytorch tensor, shape = (batch_size, tsteps, 31)
        Selected inputs for the minibatch.

    '''
    batch = torch.randint(low = 0, high = tot_size, size = (batch_size,))
    targets_batch = targets_data[batch]
    ins_batch = ins_data[batch]
    return targets_batch, ins_batch


def train_net():
    model.train()
    model.to(device)
    model.batch_size = batch_size
    
    model.W.weight.data = torch.abs(model.W.weight.data)
    
    
    targets, ins = get_ins_targets(batch_size)
    output, u = model(ins.to(device))
    
    loss_r = mse_loss(output[:,:,:2],targets[:,:,:2].to(device))/(2*np.pi)
    loss_t = mse_loss(output[:,:,2],targets[:,:,2].to(device))/tsteps #can comment this line for training spatial model faster

    reg = u.mean(dim=(0, 2)).std()
    
    loss = loss_r + loss_t*0.1 + 0.5*reg   #for the space+time model
    #loss = loss_r + 0.5*reg                 #for the space model
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss_r.item(), loss_t.item()


hidden_size = 128
input_size = 31     #30 for the space model, 31 for the space+time model
net_parameters = {'input_size': input_size, 'hidden_size': hidden_size,'batch_size': batch_size, 'noise_std': 0.01}
lr = 1e-3


model = RNN(net_parameters, sim_parameters)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
mse_loss = nn.MSELoss()

error_space = []
error_time = []

#epochs = 200000
epochs = 100

for epoch in range(epochs):
    err_space, err_time = train_net()
    error_space.append(err_space)
    error_time.append(err_time)
    if epoch % 5000 ==0:
        plt.figure()
        plt.plot(error_space)
        plt.title('Space error')
        plt.yscale('log')

        plt.figure()
        plt.plot(error_time)
        plt.title('Error tiempo')
        plt.yscale('log')
        plt.show()
        
torch.save(model,'Models/model_space_time_test.pt')