from get_raw_path import *
from trace_path import *
from linear_split_trace import *


# Put a folder called 'Clips' inside your directory, and put 100 randomly
# selected clips inside it, called '1.mp4', '2.mp4', ....

# Then run this.

# Also, ignore the runtime warnings.

good_paths = []
good_vids = []
good_inds = []
good_series = []


for i in range(1, 101):
    if i % 10 == 0:
        print(i)
    path, vid = get_path(i)
    
    sep_ind, score, margin, balance = get_sep_ind(path)
    
    if margin > 3 and (max(balance) / min(balance)) < 3:
        
        left, right = get_series(path, sep_ind)
        left = process_series(left, path)
        right = process_series(right, path)
        if has_teleports(left[10:], 5) or has_teleports(right[10:], 5):
            continue
            

    
        vid = draw_fencers_on_vid(vid, left, right)
        good_inds.append(i)
        good_paths.append(path)
        good_vids.append(vid)
        good_series.append((left, right))


input('Press enter when ready to play videos.')
for i, vid in enumerate(good_vids):
    print(good_inds[i])
    left, right = good_series[i]
    print(has_teleports(left, 5), has_teleports(right, 5))
    play_vid(vid)
     
