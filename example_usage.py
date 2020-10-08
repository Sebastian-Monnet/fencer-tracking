from get_raw_path import *
from trace_path import *


path, vid = get_path(1)

play_vid(vid, wait=30)

plt.imshow(path)
plt.show()

filtered = filter_isolated(path)
left, right = split_series(filtered)

plt.scatter(left[0], left[1])
plt.scatter(right[0], right[1])

plt.show()


left = remove_outliers(left[0], left[1])
right = remove_outliers(right[0], right[1])

plt.scatter(left[0], left[1])
plt.scatter(right[0], right[1])
plt.show()
