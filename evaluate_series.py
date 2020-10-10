from trace_path import *
def mse_eval(left_series, right_series, raw_path):
    tot_se = 0
    n = 0
    path = filter_isolated(raw_path)
    for i in range(len(path)):
        left_val = left_series[i]
        right_val = right_series[i]
        for j in range(len(path[i])):
            if path[i, j] > 0:
                left_error = left_val - j
                right_error = right_val - j
                best_error = min(np.abs(left_error), np.abs(right_error))
                '''if best_error < 5:
                    best_error = 0'''
                tot_se += best_error**2
                n += 1

    return tot_se / n

def speed_eval(left_series, right_series):
    left_vel = left_series[1:] - left_series[:-1]
    right_vel = right_series[1:] - right_series[:-1]

    return np.average(left_vel ** 4) + np.average(right_vel ** 4)
        

tup_arr = []
good_arr = []
for i in range(10, 20):
    if i % 10 == 0:
        print('Progress:', i)
    vid, left, right, path = get_series(i)
    tup_arr.append((vid, left, right, path))
    mse = mse_eval(left, right, path)
    if mse < 5:
        good_arr.append((vid, left, right, path))

inputs = []
labels = []

for i, (vid, left, right, path) in enumerate(tup_arr):

    print('Playing clip' + str(i))

    mse = mse_eval(left, right, path)
    speed = speed_eval(left, right)

    inputs.append((mse, speed))

    #print(mse)
    #print(speed)

    play_vid(vid, wait=30)
    
    a = input('Enter verdict on previous clip: ')

    labels.append(int(a))
