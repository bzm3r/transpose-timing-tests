import matplotlib.pyplot as plt
import numpy as np
import csv

import os

print("Please update known GPU info in plot.py if possible!")
# (gpu name, abbreviation, optimal tg size, line colour)
gpu_info = [("GeForce GTX 1060", "NVD GTX 1060", '#f80606'),
            ("GeForce RTX 2060", "NVD RTX 2060", '#f77304'),
            ("Intel(R) HD Graphics 630", "INT HD 630", '#0646f8'),
            ("Intel(R) Ivybridge Mobile", "INT IVYMOB 630", '#be06f8'),
            ("Intel(R) Iris(TM) Plus Graphics 640", "INT Iris 640", '#06f8e5'),
            ("Intel(R) HD Graphics 520", "INT HD 520", '#850f86'),
            ("Radeon RX 570 Series", "AMD RX 570", '#3c8521'),
            ]

knowns = [g[0] for g in gpu_info]
k_to_abbr = dict(zip(knowns, [g[1] for g in gpu_info]))
k_to_col = dict(zip(knowns, [g[2] for g in gpu_info]))
free_colors = ['#1e8787', '#871e1e', '#875b1e', '#c0b926', '#c08526']

kernel_line_styles = dict([("shuffle", {"ls": "-", "marker": "o"}), ("ballot", {"ls": "-", "marker": "s"}), ("hybrid shuffle", {"ls": "--", "marker": "d"}), ("threadgroup1D", {"ls": "--", "marker": "s"}), ("threadgroup2D", {"ls": "--", "marker": "o"})])

cwd = os.getcwd()
dat_files = [f for f in os.listdir(cwd) if os.path.splitext(f)[1] == ".dat"]

def maximally_utilized(wg_x, wg_y, num_bms):
    wg_size = wg_x*wg_y
    num_mats_per_wg = wg_size/32
    return num_bms > num_mats_per_wg

def total_threads_dispatched(wg_x, wg_y, num_bms):
    wg_size = wg_x*wg_y
    num_mats_per_wg = wg_size / 32
    num_dispatches = int(np.ceil(num_bms/num_mats_per_wg))
    print("num_bms: {} | wg_size: {}, {} | num_mats_per_wg: {} | num_dispatches: {} | num_threads: {}".format(num_bms, wg_x, wg_y, num_mats_per_wg, num_dispatches, wg_size*num_dispatches))
    return wg_size * num_dispatches

class TimingResults:
    def __init__(self, results_dir, file_name):
        with open(os.path.join(results_dir, file_name)) as datf:
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
            # workgroup size x, workgroup size y, num bms
            key = tuple([int(x) for x in r[i]])
            ts = [float(x) for x in r[i + 1]]
            insts = [float(x) for x in r[i + 2]]

            if key not in self.dat.keys():
                self.dat_ts[key] = ts
                self.dat_insts[key] = insts
            else:
                self.dat_ts[key] += ts
                self.dat_insts[key] += insts

        self.dat_ts_avg_std = dict(
            [(k, (np.average(self.dat_ts[k]), np.std(self.dat_ts[k]))) for k in self.dat_ts.keys()])
        self.dat_insts_avg_std = dict(
            [(k, (np.average(self.dat_insts[k]), np.std(self.dat_insts[k]))) for k in self.dat_insts.keys()])
        self.fallback_mode = "GPU"
        if np.average([x[1][0] for x in self.dat_ts_avg_std.items()]) < 1.0:
            self.fallback_mode = "CPU"

        self.fixed_bm_size = 2**14
        self.dat_ts_sorted_by_tgs = sorted([(k[0]*k[1], v) for k, v in self.dat_ts_avg_std.items() if k[2] == self.fixed_bm_size],
                                           key=lambda x: x[0])
        self.dat_insts_sorted_by_tgs = sorted([(k[0]*k[1], v) for k, v in self.dat_ts_avg_std.items() if k[2] == self.fixed_bm_size],
                                              key=lambda x: x[0])

        self.xs_tg = [tup[0] for tup in self.dat_ts_sorted_by_tgs]
        self.yts_tg = [tup[1] for tup in self.dat_ts_sorted_by_tgs]
        self.yinsts_tg = [tup[1] for tup in self.dat_insts_sorted_by_tgs]

        if self.fallback_mode == "GPU":
            opt_tg_vector = max(self.dat_ts_avg_std.items(), key=lambda x: x[1][0])[0]
        else:
            opt_tg_vector = max(self.dat_insts_avg_std.items(), key=lambda x: x[1][0])[0]
        self.opt_tg = opt_tg_vector[0]*opt_tg_vector[1]

        print("{} | {}".format(self.gpu, self.kernel))
        self.dat_ts_sorted_by_nd = sorted(
            [(total_threads_dispatched(k[0], k[1], k[2]), v) for k, v in self.dat_ts_avg_std.items() if k[0] * k[1] == self.opt_tg and maximally_utilized(k[0], k[1], k[2])], key=lambda x: x[0])
        self.dat_insts_sorted_by_nd = sorted(
            [(total_threads_dispatched(k[0], k[1], k[2]), v) for k, v in self.dat_insts_avg_std.items() if k[0] * k[1] == self.opt_tg and maximally_utilized(k[0], k[1], k[2])], key=lambda x: x[0])
        print("=============")

        self.xs_nd = [tup[0] for tup in self.dat_ts_sorted_by_nd]
        self.yts_nd = [tup[1] for tup in self.dat_ts_sorted_by_nd]
        self.yinsts_nd = [tup[1] for tup in self.dat_insts_sorted_by_nd]

        self.line_style = kernel_line_styles[self.kernel]
        if self.known_gpu:
            self.line_color = k_to_col[gpu]
        else:
            if len(free_colors) > 0:
                self.line_color = free_colors.pop()
            else:
                raise Exception("too many unregistered GPUs; not sure which color to pick")


trs = [TimingResults(cwd, f) for f in dat_files]


def plot_varying_tg_using_gpu_queries(timing_results):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in timing_results:
        ax.errorbar(tr.xs_tg,
                    [y[0] for y in tr.yts_tg],
                    yerr=[y[1] for y in tr.yts_tg],
                    label="{}, {}, {}".format(tr.gpu, tr.back, tr.kernel),
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style
                    )

    ax.set_xlabel("threadgroup size")
    ax.set_xticks([2 ** n for n in range(5, 11)])
    ax.set_ylabel("transpose/sec")
    ax.set_xscale("log", basex=2)
    ax.set_yscale("log", basey=10)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("GPU timing query results, num BMs={}".format(timing_results[0].fixed_bm_size))
    fig.set_size_inches(14, 8.5)
    fig.savefig(os.path.join(cwd, "plot-varying-tgs-gpu-queries.png"), bbox_inches="tight")


def plot_varying_tg_using_cpu_queries(timing_results):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in timing_results:
        ax.errorbar(tr.xs_tg,
                    [y[0] for y in tr.yinsts_tg],
                    yerr=[y[1] for y in tr.yinsts_tg],
                    label="{}, {}, {}".format(tr.gpu, tr.back, tr.kernel),
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style
                    )

    ax.set_xlabel("threadgroup size")
    ax.set_xticks([2 ** n for n in range(5, 11)])
    ax.set_ylabel("transpose/sec")
    ax.set_xscale("log", basex=2)
    ax.set_yscale("log", basey=10)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("CPU timing query results, num BMs={}".format(timing_results[0].fixed_bm_size))
    fig.set_size_inches(14, 8.5)
    fig.savefig(os.path.join(cwd, "plot-varying-tgs-cpu-queries.png"), bbox_inches="tight")

def plot_varying_tg_using_gpu_queries_with_cpu_query_fallback(timing_results):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in timing_results:
        if tr.fallback_mode == "GPU":
            dat = tr.yts_tg
        else:
            dat = tr.yinsts_tg

        ax.errorbar(tr.xs_nd,
                    [y[0] for y in dat],
                    yerr=[y[1] for y in dat],
                    label="{}, {}, {}, TGS={} ({})".format(tr.gpu, tr.back, tr.kernel, tr.opt_tg, tr.fallback_mode),
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style,
                    )

    ax.set_xlabel("threadgroup size")
    ax.set_xticks([2 ** n for n in range(5, 11)])
    ax.set_ylabel("transpose/sec")
    ax.set_xscale("log", basex=2)
    ax.set_yscale("log", basey=10)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("GPU timing query results (w/ CPU fallback)")
    fig.set_size_inches(14, 8.5)
    fig.savefig(os.path.join(cwd, "plot-varying-nds-fallback-queries.png"), bbox_inches="tight")


def plot_varying_nd_using_gpu_queries(timing_results):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in timing_results:
        ax.errorbar(tr.xs_nd,
                    [y[0] for y in tr.yts_nd],
                    yerr=[y[1] for y in tr.yts_nd],
                    label="{}, {}, {}, TGS={}".format(tr.gpu, tr.back, tr.kernel, tr.opt_tg),
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style
                    )

    ax.set_xlabel("theoretical num threads dispatched")
    ax.set_ylabel("transpose/sec")
    ax.set_xscale("log", basex=2)
    ax.set_yscale("log", basey=10)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("GPU timing query results")
    fig.set_size_inches(14, 8.5)
    fig.savefig(os.path.join(cwd, "plot-varying-nds-gpu-queries.png"), bbox_inches="tight")


def plot_varying_nd_using_cpu_queries(timing_results):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in timing_results:
        ax.errorbar(tr.xs_nd,
                    [y[0] for y in tr.yinsts_nd],
                    yerr=[y[1] for y in tr.yinsts_nd],
                    label="{}, {}, {}, TGS={}".format(tr.gpu, tr.back, tr.kernel, tr.opt_tg),
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style
                    )

    ax.set_xlabel("theoretical num threads dispatched")
    ax.set_ylabel("transpose/sec")
    ax.set_xscale("log", basex=2)
    ax.set_yscale("log", basey=10)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("CPU timing query results")
    fig.set_size_inches(14, 8.5)
    fig.savefig(os.path.join(cwd, "plot-varying-nds-cpu-queries.png"), bbox_inches="tight")


def plot_varying_nd_using_gpu_queries_with_cpu_query_fallback(timing_results):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for tr in timing_results:
        if tr.fallback_mode == "GPU":
            dat = tr.yts_nd
        else:
            dat = tr.yinsts_nd

        ax.errorbar(tr.xs_nd,
                    [y[0] for y in dat],
                    yerr=[y[1] for y in dat],
                    label="{}, {}, {}, TGS={} ({})".format(tr.gpu, tr.back, tr.kernel, tr.opt_tg, tr.fallback_mode),
                    marker=".", capsize=5, markersize=10, color=tr.line_color, ls=tr.line_style
                    )

    ax.set_xlabel("theoretical num threads dispatched")
    ax.set_ylabel("transpose/sec")
    ax.set_xscale("log", basex=2)
    ax.set_yscale("log", basey=10)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("GPU timing query results (w/ CPU fallback)")
    fig.set_size_inches(14, 8.5)
    fig.savefig(os.path.join(cwd, "plot-varying-nds-fallback-queries.png"), bbox_inches="tight")


plot_varying_tg_using_gpu_queries(trs)
plot_varying_tg_using_cpu_queries(trs)
plot_varying_nd_using_gpu_queries(trs)
plot_varying_nd_using_cpu_queries(trs)
# plot_varying_tg_using_gpu_queries_with_cpu_query_fallback(trs)
# plot_varying_nd_using_gpu_queries_with_cpu_query_fallback(trs)

plt.close("all")

