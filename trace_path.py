import numpy
import matplotlib.pyplot as plt
from v1 import *





def get_path(i):
    vid = load_vid(str(i) + '.mp4')
    vid = vid[:, 80:270]
    cloud = get_cloud(vid, thresh=1.5)
    cand = get_cloud_candidates(cloud, vid)
    cand = smooth_candidates(cand)
    fencers = naive_fencer_positions(cand)

    path = np.zeros_like(cand)

    for i, pair in enumerate(fencers):
        f1, f2 = int(pair[0]), int(pair[1])
        if f1 >= 0:
            path[i, f1] = 255
        if f2 >= 0:
            path[i, f2] = 255
        
    vid = draw_cloud_on_vid(vid, cloud, cand, fencers)

    return path, vid
    
