import cv2 as cv
import numpy as np
import copy
from scipy import signal

grid_width = grid_height = 10

def load_vid(filename):
    '''
    Returns 3-dim numpy array where each slice is a frame
    '''
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
    '''
    input: two consecutive frames of video
    output: 2-dim numpy array

    Calculates optical flow in a grid of pixels, and returns the x-componenets as a 2-dim numpy array
    '''
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
    '''
    input: 2-dim numpy array of the form output by get_vel_arr

    applies median smoothing for noise removal
    '''
    return signal.medfilt(x_vel_arr, (5, 5))

def get_masked_vel_arr(x_vel_arr, thresh):
    '''
    input:
        x_vel_arr: 2-dim numpy array of the form output by get_vel_arr, 
        thresh: float

    removes points with a magnitude below thresh    
    '''
    mask = np.abs(x_vel_arr) > thresh
    return x_vel_arr * mask

def get_cloud(vid, thresh=1.5):
    '''
    input:
        vid: 3-dim numpy array like the one output by load_vid

    returns a 3-dim numpy array whose slices are vector fields of the form output by get_vel_arr
    '''
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
    '''
    Takes a frame of video and the corresponding frame of x-velocities. Draws crosses on the frame at the
    positions corresponding to the velocities, colouring them red for positive velocity, and blue for negative.
    Colour intensity is proportional to size of velocity.
    '''
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

def draw_cloud_on_vid(vid, cloud, candidates=None, fencers=None):
    '''
    Takes a video and cloud, and draws the cloud on the video.
    
    Optional arguments are drawn on if given.
    
    candidates refers to the 1-dim array of 'votes' for each x-coordinate of the cloud frame, and represents the likelihood
    that there is a moving object with that x-coordinate.

    fencers is a list of two-element tuples, where each one is the x-coordinates of each fencer's position on the cloud.

    Note that cloud coordinates must be multiplied by grid_width and grid_height to get the actual coordinates on the frame.
    '''
    new_vid = copy.copy(vid)
    for i in range(len(cloud)):
        frame = vid[i]
        cloud_frame = cloud[i]
        new_vid[i] = draw_cloud_on_frame(frame, cloud_frame)
        if candidates is not None:
            new_vid[i] = draw_candidates_on_frame(new_vid[i],
                                                  candidates[i])
        if fencers is not None:
            new_vid[i] = draw_fencers_on_frame(new_vid[i],
                                               fencers[i])
        

    return new_vid[:-1]

def draw_fencers_on_vid(vid, left, right):
    '''
    Similar to draw_cloud_on_vid, but just draws the fencers, and uses two 1-dim arrays instead of a list of tuples
    '''
    vid = copy.copy(vid)
    for i in range(len(left)):
        vid[i] = draw_fencers_on_frame(vid[i], (left[i], right[i]))
    return vid

def draw_fencers_on_frame(frame, fencers):
    '''
    Takes a frame, and a list of 2-element tuples of fencer cloud x-coordinates.

    Draws the fencer positions on the x-axis of the frame, and returns a new frame.
    '''
    new_frame = copy.copy(frame)
    x1 = int(fencers[0] * grid_width)
    x2 = int(fencers[1] * grid_width)
    
    try:
        if x1 >= 0:
            new_frame[-14 : -7, x1 - grid_width//2 :
                      x1 + grid_width//2] = (0, 0, 255)
    except:
        pass
    try:
        if x2 >= 0:
            new_frame[-14 : -7, x2 - grid_width//2 :
                      x2 + grid_width//2] = (0, 255, 0)
    except:
        pass
    return new_frame

def draw_candidates_on_frame(frame, candidates):
    '''
    Candidates is a 1-dim array that represents the likelihood that there is a fencer at each cloud x-coordinate

    Draws these along the x-axis, showing the likelihood of a fencer at each position.
    '''
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
    
def load_and_do_all(num, wait=30, thresh=1.5, mode='play', smooth_cand=True):
    '''
    loads a video with filename 'num.mp4', draws on the cloud, candidates and fencer positions, and either
    returns or plays it depending on the mode argument.
    '''
    vid = load_vid(str(num) + '.mp4')
    vid = vid[:, 80:270]
    cloud = get_cloud(vid, thresh=thresh)
    candidates = get_cloud_candidates(cloud, vid)
    if smooth_cand:
        candidates = smooth_candidates(candidates)
    fencers = naive_fencer_positions(candidates)
    new_vid = draw_cloud_on_vid(vid, cloud, candidates, fencers)
    if mode == 'play':
        play_vid(new_vid, wait=wait)
    elif mode == 'return':
        return new_vid
    

def acceptable_surroundings(frame, row_ind, col_ind,
                            inten_thresh=70, white_thresh=50):
    '''
    takes a frame, together with cloud coordinates (row_ind and col_ind), and determines whether the frame meets certain
    requirements for being 'fencer-like'.
    '''
    x, y = col_ind * grid_width, row_ind * grid_height
    window = frame[y - 2 : y + 3, x - 2 : x + 3]
    colours = np.average(window, axis=(0, 1))
    
    ave_int = np.average(colours)

    # check if the point has at least a certain brightness. Fencers are brightly coloured, and we wouldl ike to ignore things
    # like suits of referees
    if ave_int <= inten_thresh:
        return False



    # check that the colour of the frame is close to white, since fencers generally wear white. 
    if np.max(colours) - np.min(colours) > white_thresh:
        return False

    return True
    
def get_row_votes(row, original_frame, row_ind):
    '''
    row is a 1-dim array. It is intended to be a horizontal slice of a cloud frame.
    original_frame is the actual frame from which the cloud is obtained.
    row_ind is the index of the row in the cloud frame

    We get up to two 'votes', which represent intervals where the row suggests there might be fencers.
    In particular, each vote is a 2-element tuple, where the first entry represents the start of the interval, and
    the second entry represents the length of the interval.
    '''
    components = {}
    cur_length = 0
    
    for i in range(len(row)):
        if row[i] != 0 and acceptable_surroundings(original_frame,
                                                   row_ind,
                                                   i,
                                                   white_thresh=50):
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

def get_cloud_frame_candidates(cloud_frame, original_frame):
    '''
    Takes a cloud frame and for each row, records the votes from that row. Then counts the votes and returns
    the final 1-dim array of candidate scores for the frame.
    '''
    vote_arr = []
    for i in range(len(cloud_frame)):
        row = cloud_frame[i]
        vote_arr.append(get_row_votes(row, original_frame, i))

    return count_votes(vote_arr, cloud_width = cloud_frame.shape[1])

def get_cloud_candidates(cloud, vid):
    '''
    Runs the get_cloud_frame_candidates on each frame of the cloud, returning a 2-dim array where each row is the candidate
    array for a frame.
    '''
    cand_arr = []
    for i in range(len(cloud)):
        cloud_frame = cloud[i]
        frame = vid[i]
        cand_arr.append(get_cloud_frame_candidates(cloud_frame,
                                                   frame))
    return np.array(cand_arr)
        

def count_votes(vote_arr, cloud_width=64):
    '''
    Takes a vote array of the form output by get_row_votes, and returns a 1-dim array of the total number of votes
    that was for each x-coordinate of the cloud.
    '''
    candidates = np.zeros(cloud_width)
    for elem in vote_arr:
        for pos, length in elem:
            candidates[pos : pos + length] += 1

    return candidates

def smooth_candidates(candidates):
    return signal.medfilt(candidates, (3, 3))

def remove_maximum(arr, max_ind):
    '''
    Takes a 1-dimensional array and the index of a local maximum.
    Sets points around that maximum to zero until they start increasing again.
    In other words, it removes the "bump" of the local maximum completely.
    '''
    new_arr = copy.copy(arr)
    i = max_ind
    while i < len(arr) - 1 and new_arr[i] >= new_arr[i + 1]:
        new_arr[i] = 0
        i += 1

    i = max_ind - 1
    while i > 1 and new_arr[i] >= new_arr[i - 1]:
        new_arr[i] = 0
        i -= 1

    return new_arr

def get_naive_fencer_positions_frame(cand_frame, thresh=9):
    '''
    Gets the two best local maxima of the candidate frame (which is a 1-dim numpy array). If the
    number of votes nearby is above a threshold, the location of the maximum is returned. Otherwise
    that maximum is replaced by -1, which is interpreted later as not having found a fencer.
    '''
    fencer_1 = np.argmax(cand_frame)
    
    dummy_frame = remove_maximum(cand_frame, fencer_1)

    radius = 10

    fencer_2 = np.argmax(dummy_frame)

    if np.sum(cand_frame[fencer_1 - 2: fencer_1 + 3]) < thresh:
        fencer_1 = fencer_2 = -1
    if np.sum(dummy_frame[fencer_2 - 2: fencer_2 + 3]) < thresh:
        fencer_2 = -1

    if fencer_1 > fencer_2:
        fencer_1, fencer_2 = fencer_2, fencer_1
    return fencer_1, fencer_2

def naive_fencer_positions(cand):
    '''
    Does the same as get_naive_fencer_positions_frame, but for every frame of the video.
    '''
    fencer_arr = np.zeros((cand.shape[0], 2))
    for i in range(len(cand)):
        cand_frame = cand[i]
        fencer_1, fencer_2 = get_naive_fencer_positions_frame(cand_frame)
        fencer_arr[i][0] = fencer_1
        fencer_arr[i][1] = fencer_2
    return fencer_arr
    
    


    

    
