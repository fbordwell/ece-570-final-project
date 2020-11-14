import matplotlib.pyplot as plt
import pickle as pl
import numpy as np

from fid_score import calculate_frechet_distance

fig_handle = pl.load(open('results/556374_ffhq_unet_bce_noatt_cutmix_consist/logs/inception_metrics_556374_ffhq_unet_bce_noatt_cutmix_consist.p','rb'))
print(fig_handle)


def read_stats_file(filepath):
    f = np.load(filepath)
    m, s = f['mu'][:], f['sigma'][:]
    f.close()
    return m, s


m, s = read_stats_file('FFHQ_inception_moments.npz')
print(m, s)
