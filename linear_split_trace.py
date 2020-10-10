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

    #for i in range(len(vid) - 1):
        #vid[i] = draw_candidates_on_frame(vid[i], cand[i])

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



def get_series(path, sep_ind):
    left_path = path[:, :sep_ind]
    right_path = path[:, sep_ind:]

    left_pos_arr = []
    right_pos_arr = []
    left_t_arr = []
    right_t_arr = []

    for i in range(len(path)):
        if np.sum(left_path[i]) > 0:
            left_pos_arr.append(np.average([j for j in range(len(left_path[i]))
                                            if left_path[i, j] > 0]))
            left_t_arr.append(i)

        if np.sum(right_path[i]) > 0:
            right_pos_arr.append(sep_ind + np.average([j for j in range(len(right_path[i]))
                                            if right_path[i, j] > 0]))
            right_t_arr.append(i)

    return (left_t_arr, left_pos_arr), (right_t_arr, right_pos_arr)

def remove_outliers(t_arr, pos_arr, radius):
    t_arr = copy.copy(t_arr)
    pos_arr = copy.copy(pos_arr)
    to_kill = []
    for i in range(len(t_arr)):
        a = max(i - radius, 0)
        b = min(i + radius + 1, len(pos_arr))
        sub_pos = pos_arr[a : b]

        std = np.std(sub_pos)
        lq = np.quantile(sub_pos, 0.25)
        uq = np.quantile(sub_pos, 0.75)
        iqr = uq - lq
        
        median = np.median(sub_pos)

        if pos_arr[i] > uq + 1.5 * iqr or pos_arr[i] < lq - 1.5 * iqr:
            to_kill.append(i)

    to_kill.reverse()

    for i in to_kill:
        t_arr.pop(i)
        pos_arr.pop(i)

    return t_arr, pos_arr

def fill_series(t_arr, pos_arr, path):
    if len(t_arr) == 0:
        return np.zeros(len(path))
    total_length = len(path)
    series = np.zeros(total_length)

    for ind, t in enumerate(t_arr[:-1]):
        if t_arr[ind + 1] == t + 1:
            series[t] = pos_arr[ind]
        else:
            s = t_arr[ind + 1]
            for i in range(t, s):
                series[i] = pos_arr[ind] + (i - t)/(s - t) * \
                            (pos_arr[ind + 1] - pos_arr[ind])

    for i in range(t_arr[0]):
        series[i] = -1

    for i in range(t_arr[-1], total_length):
        series[i] = pos_arr[-1]
    return series

def has_teleports(series, thresh):
    for i in range(len(series) - 2):
        if np.abs(series[i + 2] - series[i]) > thresh and series[i] >= 0:
            return True
    return False


def process_series(series, path):
    series = remove_outliers(series[0], series[1], 20)
    series = remove_outliers(series[0], series[1], 10)

    series = fill_series(series[0], series[1], path)

    series = signal.medfilt(series, (5,))

    ma_smooth = (series[2:] + series[1 : -1] + series[:-2]) / 3

    for i in range(1, len(series) - 1):
        if series[i] != -1:
            series[i] = ma_smooth[i - 1]

    return series




       
