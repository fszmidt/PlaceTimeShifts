import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
soft = nn.Softmax()
#%%

class RNN(nn.Module):                  
    def __init__(self,net_dict, sim_dict):
        super(RNN, self,).__init__()
        
        self.net = net_dict
        self.sim = sim_dict
    
        self.device = device
        self.input_size = self.net['input_size']
        self.hidden_size = self.net['hidden_size']
        self.batch_size = self.net['batch_size']
        self.noise = self.net['noise_std']
        self.tsteps = self.sim['tsteps']
        

        self.W = nn.Linear(self.hidden_size,self.hidden_size, bias = True)
        self.I_r = nn.Linear(self.input_size,self.hidden_size, bias = False)
        self.func = nn.Softmax(dim=1)

        self.beta = 20 #20 for the spatial model, 15 for the spatial + timing model
        self.out = nn.Linear(self.hidden_size,3, bias = False)
        
        

    def forward(self, input):
            noise = torch.normal(mean = 0, std = self.noise, size = (self.batch_size, self.hidden_size, self.tsteps)).to(device)
            h = torch.zeros(self.batch_size,self.hidden_size).to(device)
            x = self.func(self.beta*h)
                        
            out = torch.zeros(self.batch_size,self.tsteps,3).to(device)
            
            list_x = torch.zeros((self.batch_size,self.hidden_size,self.tsteps))
                        
            for t in range(self.tsteps):
                inputs = input[:,t,:self.input_size]
                
                W = self.W(x)
                I = self.I_r(inputs)
                
                h = I + W + noise[:,:,t]
                x = self.func(self.beta*h)
                
                list_x[:,:,t] = x
                out[:,t] = self.out(x)

            return(out,list_x)