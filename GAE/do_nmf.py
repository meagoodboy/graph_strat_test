import numpy as np
from libnmf.alsnmf import *
from libnmf.nmf import *
from libnmf.fpdnmf import *
from libnmf.pnmf import *
from libnmf.gnmf import *
import pandas as pd
import gower



def do_alsnmf(encoding, rank,max_iter):
    
    enc_np = np.array(encoding)
    enc_np.shape
    enc_np = gower.gower_matrix(enc_np)

    alsnmf= ALSNMF(enc_np, rank=rank)
    alsnmf.compute_factors(max_iter= max_iter)

    loss = alsnmf.frob_error

    W  = pd.DataFrame(alsnmf.W, index=encoding.index)

    return W, loss

def do_fpdnmf(encoding, rank,max_iter):
    
    enc_np = np.array(encoding)
    enc_np.shape
    enc_np = gower.gower_matrix(enc_np)

    fpdnmf= FPDNMF(enc_np, rank=rank)
    fpdnmf.compute_factors(max_iter=max_iter, nditer=5)

    loss = fpdnmf.div_error

    W  = pd.DataFrame(fpdnmf.W, index=encoding.index)

    return W, loss

def do_gnmf(encoding, rank,max_iter):
    
    enc_np = np.array(encoding)
    enc_np.shape

    gnmf= GNMF(enc_np, rank=rank)
    gnmf.compute_factors(max_iter= max_iter, lmd= 0.3, weight_type='heat-kernel', param= 0.4)
    
    loss = gnmf.frob_error

    W  = pd.DataFrame(gnmf.W, index=encoding.index)

    return W, loss

def do_pnmf(encoding, rank,max_iter):
    
    enc_np = np.array(encoding)
    enc_np.shape

    pnmf= PNMF(enc_np, rank=rank)
    pnmf.compute_factors(max_iter= max_iter)
    
    loss = pnmf.frob_error

    W  = pd.DataFrame(pnmf.W, index=encoding.index)

    return W, loss

def do_nmf(encoding, rank,max_iter):
    
    enc_np = np.array(encoding)
    enc_np.shape

    nmf= NMF(enc_np, rank=rank)
    nmf.compute_factors(max_iter= max_iter)
    
    loss = nmf.frob_error

    W  = pd.DataFrame(nmf.W, index=encoding.index)

    return W, loss