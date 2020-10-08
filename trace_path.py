import numpy
import matplotlib.pyplot as plt
from get_raw_path import *





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
    
def filter_isolated(path, x_rad=5, y_rad=5):
    path = copy.copy(path)
    for i in range(path.shape[0]):
        for j in range(path.shape[1]):
            if path[i, j] == 255:
                piece = path[i - y_rad : i + y_rad, j - x_rad : j + x_rad]
                if np.sum(piece) == 255:
                    piece[:] = 0
    return path



def kill_triples(path):
    path = copy.copy(path)
    for i in range(path.shape[0]):
        if np.sum(path[i]) > 2 * 255:
            path[i] = 0
    return path


def split_series(path):
    left_t = []
    right_t = []

    left_pos = []
    right_pos = []
    
    
    for i, row in enumerate(path):
        nonzero_inds = [k for k in range(len(row)) if row[k] != 0]

        if np.sum(row) == 0:
            continue
        one_fencer = False
        for k in range(len(row) - 15):
            if np.sum(row[k : k + 15]) == np.sum(row):
                one_fencer = True

        if one_fencer:
            pos = np.average(nonzero_inds)
            if left_t == [] or right_t == []:
                continue
            if np.abs(pos - left_pos[-1]) \
               < np.abs(pos - right_pos[-1]):
                left_t.append(i)
                left_pos.append(pos)
            else:
                right_t.append(i)
                right_pos.append(pos)
            continue
            
       
        losses = np.zeros(path.shape[1])
        for split in range(len(row)):
            left_inds = np.array([k for k in nonzero_inds if k <= split])
            right_inds = np.array([k for k in nonzero_inds if k > split])

            if len(left_inds) == 0:
                left_mse = -1e-10
            else:
                left_mse = np.mean((left_inds - np.mean(left_inds))**2)

            if len(right_inds) == 0:
                right_mse = -1e-10
            else:
                right_mse = np.mean(right_inds - np.mean((right_inds))**2)
           
            losses[split] = left_mse + right_mse
        split = np.argmin(losses)

        new_left_pos = np.mean([k for k in nonzero_inds if k <= split])
        new_right_pos = np.mean([k for k in nonzero_inds if k > split])

        left_t.append(i)
        right_t.append(i)
        left_pos.append(new_left_pos)
        right_pos.append(new_right_pos)
        
    return (left_t, left_pos), (right_t, right_pos)
            
def get_missing_sections(t_arr, path):
    # inclusive exclusive
    total_length = path.shape[0]
    missing_sections = []
    started = False
    
    for i in range(total_length):
        if i not in t_arr and not started:
            a = i
            started = True

        if i in t_arr and started:
            b = i
            started = False
            missing_sections.append((a, b))
    if started:
        missing_sections.append((a, total_length))

    return missing_sections

def remove_outliers(t_arr, pos_arr):
    t_arr = copy.copy(t_arr)
    pos_arr = copy.copy(pos_arr)
    to_kill = []
    for i in range(len(pos_arr) - 3, -1, -1):
        if np.abs(pos_arr[i] - pos_arr[i+2]) < 4 and \
           np.abs(pos_arr[i+1] - pos_arr[i]) > 3:
            to_kill.append(i + 1)
    t_arr = [t_arr[i] for i in range(len(t_arr)) if i not in to_kill]
    pos_arr = [pos_arr[i] for i in range(len(pos_arr)) if i not in to_kill]
    
    return t_arr, pos_arr

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

    
    
            
        
def fill_gaps(t_arr, pos_arr, path):
    series = np.zeros((path.shape[0],))
    for i, t in enumerate(t_arr):
        series[t] = pos_arr[i]

    missing_sections = get_missing_sections(t_arr, path)

    for a, b in missing_sections:
        pass

    



def get_neighbours(path, y, x):
    arr = []
    for i in range(y - 1, y + 2):
        for j in range(x - 1, x + 2):
            if path[i, j] > 0 and not (i == j == 0):
                arr.append((i, j))
    return arr

def get_segment(path, y, x):
    running = True
    points = [(y, x)]
    while running:
        y += 1
        children = []
        for j in range(x - 1, x + 2):
            if path[y, j] > 0:
                children.append((y, j))

        if len(children) == 1:
            points.append(children[0])
            x = children[0][1]
        else:
            running = False
    return points




raw_left_arr = []
raw_right_arr = []
left_arr = []
right_arr = []

for i in range(1, 16):
    path, vid = get_path(i)
    filt = filter_isolated(path)
    left, right = split_series(filt)
    raw_left_arr.append(copy.copy(left))
    raw_right_arr.append(copy.copy(right))

    left = remove_outliers(left[0], left[1], 20)
    left = remove_outliers(left[0], left[1], 10)
    right = remove_outliers(right[0], right[1], 20)
    right = remove_outliers(right[0], right[1], 10)

    left_arr.append(left)
    right_arr.append(right)


for i in range(15):
    raw_left, raw_right = raw_left_arr[i], raw_right_arr[i]

    left, right = left_arr[i], right_arr[i]
    print(i + 1)
    print('left')

    plt.scatter(raw_left[0], raw_left[1])
    plt.scatter(left[0], left[1])
    plt.show()

    print('right')

    plt.scatter(raw_right[0], raw_right[1])
    plt.scatter(right[0], right[1])
    plt.show()
