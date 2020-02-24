import matplotlib.pyplot as plt
import numpy as np
import csv

import os

dir = os.getcwd()
dat_files = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == ".dat"]

class TimingResults:
    def __init__(self, dir, file_name):
        with open(os.path.join(dir, file_name)) as datf:
            rdr = csv.reader(datf)
            r = [row for row in rdr]

        if len(r) == 0:
            raise Exception("No data in file.")
        elif (len(r) - 1) % 3 != 0:
            raise Exception("Incorrect file format.")

        backend, kernel_type, device_name = file_name.split(".")[0].split("-")
        if r[0][0] != device_name:
            raise Exception("Device in file name ({}) does not match device in file ({}).".format(device_name, r[0]))

        self.backend = backend
        self.kernel_type = kernel_type
        self.device_name = device_name
        self.data = dict()
        for i in range(1, len(r), 3):
            work_group_size = tuple([int(x) for x in r[i]])
            timestamp_query_times = [float(x) for x in r[i + 1]]

            if work_group_size not in self.data.keys():
                self.data[work_group_size] = timestamp_query_times
            else:
                self.data[work_group_size] += timestamp_query_times



timing_results = [TimingResults(dir, f) for f in dat_files]

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots()

for tr in timing_results:
    ps = sorted([[wgsize[0]*wgsize[1], np.average(tr.data[wgsize]), np.std(tr.data[wgsize])] for wgsize in tr.data.keys()], key=lambda p: p[0])
    ax.errorbar([p[0] for p in ps], [p[1] for p in ps], yerr=[p[2] for p in ps], label="{}, {}, {}".format(tr.device_name, tr.backend, tr.kernel_type), marker=".", capsize=5, markersize=10)

ax.set_xticks([2**n for n in range(5, 6)] + [2**n for n in range(7, 11)])
ax.legend(bbox_to_anchor=(1.2, 0.5))
#ax.legend(loc="best")
fig.set_size_inches(11, 11)
fig.savefig(os.path.join(dir, "plot.png"))
