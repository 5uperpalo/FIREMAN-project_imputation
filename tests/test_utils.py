from fireman_impupation.src.utils import HINTmatrix_gen, MCARgen
import numpy as np
import torch

def test_MCARgen_HINTmatrix():
    p = 0.2
    data = np.random.uniform(0,10,size=[1000,1000])
    data_missing, mask = MCARgen(data, probability=p)
    hint_matrix = HINTmatrix_gen(torch.from_numpy(mask), hint_rate=(1-p), orig_paper=False).numpy()
    assert round(abs(hint_matrix-mask).sum()/(mask.shape[0]*mask.shape[1]), 1) == p
