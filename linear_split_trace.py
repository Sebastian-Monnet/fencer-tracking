import numpy
from get_raw_path import *
import matplotlib.pyplot as plt

def get_path(i):
    vid = load_vid('Clips/' + str(i) + '.mp4')
    vid = vid[:, 80:270]
    cloud = get_cloud(vid, thresh=1.5)
    cand = get_cloud_candidates(cloud, vid)
    cand = smooth_candidates(cand)
    fencers = naive_fencer_positions(cand)

    for i in range(len(vid) - 1):
        vid[i] = draw_candidates_on_frame(vid[i], cand[i])

    path = np.zeros_like(cand)

    for i, pair in enumerate(fencers):
        f1, f2 = int(pair[0]), int(pair[1])
        if f1 >= 0:
            path[i, f1] = 255
        if f2 >= 0:
            path[i, f2] = 255

        

    return path[10 : -25], vid[10 : -25]


def get_mse(proj_slice):
    positions = np.array([[i for i in range(proj_slice.shape[1])]])
    est = np.sum(positions * proj_slice) / np.sum(proj_slice)
    return np.sum(proj_slice * (est - positions)**2) / np.sum(proj_slice), est

def get_sep_ind(path):
    proj = np.reshape(np.sum(path, axis=0), (1, -1))
    #proj = cv.GaussianBlur(proj, (1, 5), 0, 0)

    if np.sum(proj) == 0:
        return None

    scores = np.zeros(proj.shape[1] - 1)

    for i in range(len(scores)):
        a = proj[:, :i]
        b = proj[:, i:]
        if np.sum(a) == 0 or np.sum(b) == 0:
            scores[i] = 1e5
            continue
        left_mse, _ = get_mse(a)
        right_mse, _ = get_mse(b)


        scores[i] = left_mse + right_mse

    best_score = np.min(scores)
    
    best_inds = np.array([i for i in range(len(scores)) if scores[i] < 0.1 + best_score])
    margin = len(best_inds)
    
    sep_ind = best_inds[len(best_inds)//2]

    balance = np.sum(proj[:,:sep_ind]), np.sum(proj[:,sep_ind:])
    
    return sep_ind, best_score, margin, balance

path_arr = []
vid_arr = []
score_arr = []
margin_arr = []
balance_arr = []

good_paths = []

for i in range(1, 30):
    path, vid = get_path(i)
    sep_ind, score, margin, balance = get_sep_ind(path)
    #balance_arr.append(balance)
    #score_arr.append(score)
    #margin_arr.append(margin)
    path[:, sep_ind] = 255
    #path_arr.append(path)
    #vid_arr.append(vid)
    if margin > 3 and (max(balance) / min(balance)) < 3:
        good_paths.append(path)

for i in range(len(path_arr)):
    print('')
    print(i + 1)
    print(score_arr[i], margin_arr[i])
    plt.imshow(path_arr[i])
    plt.show()
     
        
       
