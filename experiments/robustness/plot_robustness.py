# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import numpy as np
import matplotlib.pyplot as plt

times_robust = np.array([
    [10.05, 26.95, 2.10, 2.90],
    [18.56, 0.84, 1.20, 2.31],
    [61.10, 2.86, 3.85, 7.81],
    [87.70, 39.95, 9.95, 19.31],
    [119.04, 72.77, 47.18, 54.28],
    [125.61, 119.94, 75.46, 96.56],
])

failures_robust = np.array([
    [10, 16, 0, 2],
    [26, 1, 0, 0],
    [70, 9, 2, 6],
    [83, 52, 18, 17],
    [93, 89, 81, 74],
    [90, 83, 48, 62]
])

fig, ax1 = plt.subplots()

plt.plot(times_robust)
plt.legend(['h = 2', 'h = 10', 'h = 50', 'h = 100'])
plt.xticks(np.arange(6), ('10', '20', '50', '100', '200', '500'))
plt.xlabel('Radius')
plt.ylabel('Computational Time  [seconds]')
plt.grid()
bottom, top = plt.ylim()

ax2 = ax1.twinx()
ax2.set_ylim(bottom, top)
# plt.figure()
for idx in np.arange(4):
    plt.scatter(np.arange(6), failures_robust[:, idx], marker='*', s=100)
ax2.set_ylabel('Failures')
fig.tight_layout()


min_times = np.array([
    [0.05, 0.09, 0.15, 1.37],
    [0.09, 0.12, 0.90, 1.65],
    [0.52, 1.09, 1.45, 3.90],
    [0.21, 5.47, 5.56, 12.00],
    [0.40, 8.38, 14.11, 20.27],
    [7.83, 19.26, 20.39, 42.59]
])

plt.figure()
plt.plot(min_times/times_robust)
plt.legend(['h = 2', 'h = 10', 'h = 50', 'h = 100'])
plt.xticks(np.arange(6), ('10', '20', '50', '100', '200', '500'))
plt.xlabel('Radius')
plt.ylabel('Min Time over Avg Time')
plt.grid()


plt.show()
