import cv2 as cv
import numpy as np
import copy
from scipy import signal

grid_width = grid_height = 10

def load_vid(filename):
    cap = cv.VideoCapture(filename)
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_list.append(frame)
        else:
            break
    return np.array(frame_list)

def get_vel_arr(old, new):
    p0 = []
    old = cv.GaussianBlur(cv.cvtColor(old, cv.COLOR_BGR2GRAY), (3, 3), 0, 0)
    new = cv.GaussianBlur(cv.cvtColor(new, cv.COLOR_BGR2GRAY), (3, 3), 0, 0)

    for i in range(old.shape[0] // grid_height):
        for j in range(old.shape[1] // grid_width):
            p0.append([[grid_width * j, grid_height * i]])

    p0 = np.array(p0).astype('float32')

    lk_params = dict( winSize = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 
                              10, 0.03))

    p1, st, err = cv.calcOpticalFlowPyrLK(old, new, p0,
                                          None, **lk_params)

    vels = p1 - p0

    w = int(p0[-1][0, 0] // grid_width)
    h = int(p0[-1][0, 1] // grid_height)
    
    x_vel_arr = np.zeros(int((h + 1) * (w + 1)),)

    for i in range(vels.shape[0]):
        vel = vels[i]
        x_vel = vel[0, 0]
        x_vel_arr[i] = x_vel

    x_vel_arr = x_vel_arr.reshape((h + 1, w + 1))

    return x_vel_arr

def smooth_vel_arr(x_vel_arr):
    return signal.medfilt(x_vel_arr, (5, 5))

def get_masked_vel_arr(x_vel_arr, thresh):
    mask = np.abs(x_vel_arr) > thresh
    return x_vel_arr * mask

def get_cloud(vid, thresh=1.5):
    cloud = []
    for i in range(len(vid) - 1):
        
        old = vid[i]
        new = vid[i + 1]
        vel_arr = get_vel_arr(old, new)
        vel_arr = smooth_vel_arr(vel_arr)
        vel_arr = vel_arr - np.median(vel_arr)
        vel_arr = get_masked_vel_arr(vel_arr, thresh)
        cloud.append(vel_arr)

    return np.array(cloud)
        

def draw_cloud_on_frame(frame, cloud_frame):
    new_frame = copy.copy(frame)
    for i in range(cloud_frame.shape[0]):
        for j in range(cloud_frame.shape[1]):
            y, x = i * grid_height, j * grid_width
            vel = cloud_frame[i, j]

            activ = 100 * np.sqrt(np.abs(vel))

            if vel > 0:
                colour = (0, 0, activ)
            else:
                colour = (activ, 0, 0)
            if activ < 1:
                continue

            try:
                new_frame[y, x - 3 : x + 4] = np.array(colour)
                new_frame[y - 3 : y + 4, x] = np.array(colour)
            except:
                pass
    return new_frame

def draw_cloud_on_vid(vid, cloud, candidates=None):
    new_vid = copy.copy(vid)
    for i in range(len(cloud)):
        frame = vid[i]
        cloud_frame = cloud[i]
        new_vid[i] = draw_cloud_on_frame(frame, cloud_frame)
        if candidates is not None:
            new_vid[i] = draw_candidates_on_frame(new_vid[i],
                                                  candidates[i])
        

    return new_vid[:-1]

def draw_candidates_on_frame(frame, candidates):
    new_frame = copy.copy(frame)
    for i in range(len(candidates)):
        x = i * grid_width
        activ = min(30 * candidates[i], 255)
      
        new_frame[- 7 : -1,
                  x - grid_width//2 : x + grid_width//2] = activ
    return new_frame
        

def play_vid(vid, wait=30):
    vid = vid.astype('uint8')
    for frame in vid:
        cv.imshow('a', frame)
        cv.waitKey(wait)
    
def load_and_play_all(num, wait=30, thresh=1.5):
    vid = load_vid(str(num) + '.mp4')
    vid = vid[:, 80:270]
    cloud = get_cloud(vid, thresh=thresh)
    candidates = get_cloud_candidates(cloud)
    new_vid = draw_cloud_on_vid(vid, cloud, candidates)
    play_vid(new_vid)

def get_row_votes(row):
    # row is a 1-dim np array. Returns indices central
    # to clusters that might be fencers
    components = {}
    cur_length = 0
    for i in range(len(row)):
        if row[i] != 0:
            cur_length += 1
            continue
        if cur_length > 0:
            components[i - cur_length] = cur_length
            cur_length = 0
    if cur_length > 0:
        components[len(row) - cur_length] = cur_length
        cur_length = 0

    if len(components) == 0:
        return []

    if len(components) <= 2:
        return list(components.items())
    
    key_list = list(components)
    key_1, key_2 = key_list[0], key_list[1]

    if components[key_1] < components[key_2]:
        key_1, key_2 = key_2, key_1

    val_1 = components[key_1]
    val_2 = components[key_2]
    
    for key in list(components.keys())[2:]:
        val = components[key]
        if val >= val_1:
            key_2 = key_1
            val_2 = val_1
            key_1 = key
            val_1 = val
        elif val > val_2:
            key_2 = key
            val_2 = val

    return [(key_1, val_1), (key_2, val_2)]

def get_cloud_frame_candidates(cloud_frame):
    vote_arr = []
    for row in cloud_frame:
        vote_arr.append(get_row_votes(row))

    return count_votes(vote_arr, cloud_width = cloud_frame.shape[1])

def get_cloud_candidates(cloud):
    cand_arr = []
    for frame in cloud:
        cand_arr.append(get_cloud_frame_candidates(frame))
    return np.array(cand_arr)
        

def count_votes(vote_arr, cloud_width=64):
    candidates = np.zeros(cloud_width)
    for elem in vote_arr:
        for pos, length in elem:
            candidates[pos : pos + length] += 1

    return candidates


    

    
