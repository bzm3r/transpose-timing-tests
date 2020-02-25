import matplotlib.pyplot as plt
import numpy as np
import csv

import os

print!("Please update known GPU info in plot.py if possible!")
# (gpu name, abbreviation, optimal tg size, line colour)
gpu_info = [\
    ("GeForce GTX 1060", "NVD GTX 1060", '#f80606'),\
    ("Intel(R) HD Graphics 630", "INT HD 630", '#0646f8')\
    ("Intel(R) Ivybridge Mobile", "INT IVYMOB 630", '#be06f8'),\
    ("Intel(R) Iris(TM) Plus Graphics 640", "INT Iris 640", '#06f8e5'),\
    ("Intel(R) HD Graphics 520", '#850f86'),\
    ("Radeon RX 570 Series", "AMD RX 570", #08f806'),\
    ]
knowns = [g[0] for g in gpu_info]
k_to_abbr = dict(zip(knowns, [g[1] for g in gpu_info]))
k_to_col = dict(zip(knowns, [g[3] for g in gpu_info]))
free_colors = ['#25871e', '#1e8787', '#871e1e', '#875b1e', '#c0b926', '#c08526']

kernel_line_styles = dict([("shuffle", "-"), ("hybrid shuffle", "--"), ("threadgroup", ":")])

dir = os.getcwd()
dat_files = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == ".dat"]

class TimingResults:
    def __init__(self, dir, file_name):
        with open(os.path.join(dir, file_name)) as datf:
            rdr = csv.reader(datf)
            r = [row for row in rdr]

        if len(r) == 0:
            raise Exception("No dat in file.")
        elif (len(r) - 1) % 3 != 0:
            raise Exception("Incorrect file format.")

        back, kernel, gpu = file_name.split(".")[0].split("-")
        if r[0][0] != gpu:
            raise Exception("Device in file name ({}) does not match device in file ({}).".format(gpu, r[0]))

        self.back = back
        self.kernel = kernel
        self.known_gpu = [gpu in knowns]
        if self.known_gpu:
            self.gpu = k_to_abbr[gpu]
        else:
            self.gpu = gpu

        self.dat_ts = dict()
        self.dat_insts = dict()
        self.dat = dict()

        for i in range(1, len(r), 3):
            # workgroup size x, workgroup size y, num dispatches
            key = tuple([int(x) for x in r[i])
            ts = [float(x) for x in r[i + 1]]
            insts = [float(x) for x in r[i + 2]]

            if key not in self.dat.keys():
                self.dat_ts[key] = ts
                self.dat_insts[key] = insts
            else:
                self.dat_ts[key] += ts
                self.dat_insts[key] += insts

        self.dat_ts_avg_std = dict([(k, (np.average(self.dat_ts[k]), np.std(self.dat_ts[k]))) for k in self.dat_ts.keys()])
        self.dat_ts_avg_std = dict([(k, (np.average(self.dat_insts[k]), np.std(self.dat_insts[k]))) for k in self.dat_insts.keys()])
        self.fallback_mode = "GPU"
        if np.average([x[1][0] for x in self.dat_ts_avg_std.values()]) < 1.0:
            self.fallback_mode = "CPU"
            
        self.dat_ts_sorted_by_tgs = sorted([(k[0]*k[1], v) for k, v in self.dat_ts], lambda x: x[0])
        self.dat_insts_sorted_by_tgs = sorted([(k[0]*k[1], v) for k, v in self.dat_insts], lambda x: x[0])

        self.xs_tg = [tup[0] for tup in self.dat_ts_sorted_by_tgs]
        self.yts_tg = [tup[1] for tup in self.dat_ts_sorted_by_tgs]
        self.yinsts_tg = [tup[1] for tup in self.dat_insts_sorted_by_tgs]
        
        self.opt_tg = max(self.dat_ts_sorted_by_tgs, key=lambda x: x[0])[0]

        self.dat_ts_sorted_by_nd = sorted([(k[2], v) for k, v in self.dat_ts if k[0]*k[1] == self.opt_tg], lambda x: x[0])
        self.dat_insts_sorted_by_nd = sorted([(k[2], v) for k, v in self.dat_insts if k[0]*k[1] == self.opt_tg], lambda x: x[0])

        self.xs_nd = [tup[0] for tup in self.dat_ts_sorted_by_nds]
        self.yts_nd = [tup[1] for tup in self.dat_ts_sorted_by_nds]
        self.yinsts_nd = [tup[1] for tup in self.dat_insts_sorted_by_nds]
        
        self.line_style = kernel_line_styles[self.kernel]
        if self.known_gpu:
            self.line_color = k_to_col[gpu]
        else:
            if len(free_colors) > 0:
                self.line_color = free_colors.pop()
            else:
                raise Exception("too many unregistered GPUs; not sure which color to pick")

trs = [TimingResults(dir, f) for f in dat_files]

def plot_varying_tg_using_gpu_queries(trs):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in trs:
        ax.errorbar(tr.xs_tg,\
                    [y[0] for y in tr.yts_tg],\
                    yerr=[y[1] for y in tr.yts_tg],\
                    label="{}, {}, {}".format(k_to_abbr[tr.gpu], tr.back, tr.kernel),\
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style\
                    )

    ax.set_xlabel("threadgroup size")
    ax.set_xticks([2**n for n in range(5, 11)])
    ax.set_ylabel("transpose/sec")
    ax.legend(loc="best")
    ax.set_title("GPU timing query results")
    fig.set_size_inches(11, 8.5)
    fig.savefig(os.path.join(dir, "plot-varying-tgs-gpu-queries.png"))

def plot_varying_tg_using_cpu_queries(trs):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in trs:
        ax.errorbar(tr.xs_tg,\
                    [y[0] for y in tr.yinsts_tg],\
                    yerr=[y[1] for y in tr.yinsts_tg],\
                    label="{}, {}, {}".format(k_to_abbr[tr.gpu], tr.back, tr.kernel),\
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style\
                    )

    ax.set_xlabel("threadgroup size")
    ax.set_xticks([2**n for n in range(5, 11)])
    ax.set_ylabel("transpose/sec")
    ax.legend(loc="best")
    ax.set_title("CPU timing query results")
    fig.set_size_inches(11, 8.5)
    fig.savefig(os.path.join(dir, "plot-varying-tgs-cpu-queries.png"))

def plot_varying_tg_using_gpu_queries_with_cpu_query_fallback(trs):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in trs:
        if tr.fallback_mode == "GPU":
            dat = tr.yts_tg
        else:
            dat = tr.yinsts_tg
            
        ax.errorbar(tr.xs_nd,\
                    [y[0][0] for y in dat],\
                    yerr=[y[0][1] for y in dat],\
                    label="{}, {}, {}, TGS={} ({})".format(k_to_abbr[tr.gpu], tr.back, tr.kernel, tr.opt_tg, tr.fallback_mode),\
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style\
                    )

    ax.set_xlabel("threadgroup size")
    ax.set_xticks([2**n for n in range(5, 11)])
    ax.set_ylabel("transpose/sec")
    ax.legend(loc="best")
    ax.set_title("GPU timing query results (w/ CPU fallback)")
    fig.set_size_inches(11, 8.5)
    fig.savefig(os.path.join(dir, "plot-varying-nds-fallback-queries.png"))
    
def plot_varying_nd_using_gpu_queries(trs):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in trs:
        ax.errorbar(tr.xs_nd,\
                    [y[0] for y in tr.yts_nd],\
                    yerr=[y[1] for y in tr.yts_nd],\
                    label="{}, {}, {}, TGS={}".format(k_to_abbr[tr.gpu], tr.back, tr.kernel, tr.opt_tg),\
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style\
                    )

    ax.set_xlabel("num dispatches")
    ax.set_ylabel("transpose/sec")
    ax.legend(loc="best")
    ax.set_title("GPU timing query results")
    fig.set_size_inches(11, 8.5)
    fig.savefig(os.path.join(dir, "plot-varying-nds-gpu-queries.png"))

def plot_varying_nd_using_cpu_queries(trs):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in trs:
        ax.errorbar(tr.xs_nd,\
                    [y[0] for y in tr.yinsts_nd],\
                    yerr=[y[1] for y in tr.yinsts_nd],\
                    label="{}, {}, {}, TGS={}".format(k_to_abbr[tr.gpu], tr.back, tr.kernel, tr.opt_tg),\
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style\
                    )

    ax.set_xlabel("num dispatches")
    ax.set_ylabel("transpose/sec")
    ax.legend(loc="best")
    ax.set_title("CPU timing query results")
    fig.set_size_inches(11, 8.5)
    fig.savefig(os.path.join(dir, "plot-varying-nds-cpu-queries.png"))

def plot_varying_nd_using_gpu_queries_with_cpu_query_fallback(trs):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in trs:
        if tr.fallback_mode == "GPU":
            dat = tr.yts_nd
        else:
            dat = tr.yinsts_nd
            
        ax.errorbar(tr.xs_nd,\
                    [y[0][0] for y in dat],\
                    yerr=[y[0][1] for y in dat],\
                    label="{}, {}, {}, TGS={} ({})".format(k_to_abbr[tr.gpu], tr.back, tr.kernel, tr.opt_tg, tr.fallback_mode),\
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style\
                    )

    ax.set_xlabel("num dispatches")
    ax.set_ylabel("transpose/sec")
    ax.legend(loc="best")
    ax.set_title("GPU timing query results (w/ CPU fallback)")
    fig.set_size_inches(11, 8.5)
    fig.savefig(os.path.join(dir, "plot-varying-nds-fallback-queries.png"))

plot_varying_tg_using_gpu_queries(trs)
plot_varying_tg_using_cpu_queries(trs)
plot_varying_nd_using_gpu_queries(trs)
plot_varying_nd_using_cpu_queries(trs)
plot_varying_tg_using_gpu_queries_with_cpu_query_fallback(trs)
plot_varying_nd_using_gpu_queries_with_cpu_query_fallback(trs)

