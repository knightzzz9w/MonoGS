import numpy as np
import torch

def inverse_Tmatrix(T):
    assert( T.shape == (4,4))
    R = T[:3 , :3] ; t = T[:3 , 3:]
    T_inv = torch.eye(4).to("cuda")
    T_inv[:3,  :3] = R.T
    T_inv[:3 , 3:] = R.T@(-t)
    return T_inv