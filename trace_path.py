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
        count = np.sum(row) // 255
        if count == 0:
            continue
        if count == 1:
            pos = np.argmax(row)
            if left_t == [] or right_t == []:
                continue
            if np.abs(pos - left_pos[-1]) \
               < np.abs(pos - right_pos[-1]):
                left_t.append(i)
                left_pos.append(pos)
            else:
                right_t.append(i)
                right_pos.append(pos)

        else:
            pos1 = np.argmax(row)
            pos2 = pos1 + 1 + np.argmax(row[pos1 + 1:])

            left_t.append(i)
            right_t.append(i)

            left_pos.append(pos1)
            right_pos.append(pos2)

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
    for i in range(len(pos_arr) - 3, -1, -1):
        if np.abs(pos_arr[i] - pos_arr[i + 2]) < 4 and \
           np.abs(pos_arr[i+1] - pos_arr[i]) > 5:
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
        
