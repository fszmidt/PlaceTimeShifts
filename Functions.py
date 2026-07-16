import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
#%%
def normalize(v,ax):
    '''
    
    Parameters
    ----------
    v : Numpy array
        Vector a normalizar
    ax : Int
        Dimensión del vector sobre la que normalizar

    Returns
    -------
    Numpy array
        vector normalizado

    '''
    return v/np.linalg.norm(v,axis = ax,keepdims = True)



def dist(v1,v2):
    d2 = ((v1-v2)**2).sum()
    return np.sqrt(d2)

def calc_dist_periodica(a,b, periodo):
    dist1 = np.abs(b-a)
    dist2 = np.abs(b - (a + periodo))
    return(np.min([dist1,dist2]))


def filtered_weights(w,mask_1,mask_2):
    '''
    Devuelve una matriz w cuyas filas fueron filtradas con mask_1 y sus columnas por mask_2
    
    Inputs:
        
    w (array de MxN)
    mask_1 (array bool de tamaño M)
    mask_2 (array bool de tamaño N)
    
    
    '''

    len_1 = len(mask_1)
    len_2 = len(mask_2)
    
    filtered = np.zeros((len_1,len_2))
    for i in range(len_1):
        for j in range(len_2):
            filtered[i,j] = w[mask_1[i],mask_2[j]]
    return(filtered)


def center_of_mass(x,f):
    return np.dot(x,f) / f.sum()

def chen_com(f):
    rows, cols = f.shape
    coms = np.zeros(rows)
    for row in range(rows):
        bins = np.arange(cols)
        coms[row] = center_of_mass(bins,f[row])
        
    return coms

def periodic_center_of_mass(matrix):
    """
    matrix: shape (n_rows, n_bins)
    returns: array of length n_rows with COM in [0, n_bins)
    """
    n_bins = matrix.shape[1]

    angles = 2 * np.pi * np.arange(n_bins) / n_bins

    sin_term = np.sum(matrix * np.sin(angles), axis=1)
    cos_term = np.sum(matrix * np.cos(angles), axis=1)

    mean_angle = np.arctan2(sin_term, cos_term)
    mean_angle = np.mod(mean_angle, 2*np.pi)

    return n_bins * mean_angle / (2*np.pi)


def vel_att(x0, batch_size, n_trials,tsteps,ins):
    
    x = x0/x0.max(axis=(0,2), keepdims = True)
    vel = np.linalg.norm((np.diff(x,axis = 2)),axis = 1)
    blocks = batch_size//n_trials
    vel_mean = np.zeros((blocks))
    for i in range(blocks):
        vel_mean[i] = vel[n_trials*i:n_trials*(i+1)].mean(axis = (0,1))
        
        
    return  vel_mean
    
def pop_means(x, batch_size, n_trials,tsteps,cells_c_in, cells_der_in,cells_izq_in,t0):    
    mean_centered = np.zeros(batch_size//n_trials)
    mean_cw = np.zeros(batch_size//n_trials)
    mean_ccw = np.zeros(batch_size//n_trials)
    for b in range(batch_size//n_trials):
        mean_centered[b] = x[b*n_trials:(b+1)*n_trials,cells_c_in,t0:].mean()    
        mean_cw[b] = x[b*n_trials:(b+1)*n_trials,cells_der_in,t0:].mean()    
        mean_ccw[b] = x[b*n_trials:(b+1)*n_trials,cells_izq_in,t0:].mean()    
    
    return mean_centered, mean_cw, mean_ccw    