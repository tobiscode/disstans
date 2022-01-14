"""
This is a tutorial creating a synthetic network to test how an increasing
number of stations witnessing the same transient signal can aid in its
recovery if the signal is below the noise floor.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import detrend
from scipy.special import binom
from itertools import permutations, product
from tqdm import tqdm
import disstans
from disstans import Network, Station, Timeseries
from disstans.tools import parallelize
from disstans.solvers import InverseReweighting
from disstans.models import ISpline

# make it prettier
from matplotlib import rcParams
rcParams['font.sans-serif'] = ["NewComputerModernSans10"]
rcParams['font.size'] = "14"

# output
outdir = "out/tutorial_5"
os.makedirs(outdir, exist_ok=True)


# make single checking function
def run_single(all_inputs):
    stations_to_use, station_names, all_stations, pen, rw_func = all_inputs
    i = stations_to_use.size
    net = Network(name="BoxNet")
    for istat in stations_to_use:
        net.add_station(name=station_names[istat],
                        station=all_stations[station_names[istat]])
    net.spatialfit("Synthetic",
                   penalty=float(pen),
                   spatial_reweight_models=["Transient"],
                   spatial_reweight_iters=20,
                   local_reweight_func=rw_func,
                   spatial_reweight_max_rms=1e-6,
                   dist_weight_min=80,
                   no_pbar=True)
    net.evaluate("Synthetic", no_pbar=True)
    for stat in net:
        stat[f"Err_N{i}"] = stat["Truth"] - stat.fits["Synthetic"]["Transient"]
    return [np.sqrt(np.mean(net[station_names[istat]][f"Err_N{i}"].data.values**2))
            for istat in stations_to_use], net, i


def get_perms(i, max_num_stations, num_samples):
    if (binom(max_num_stations, i) < 1e3) and (i < max_num_stations // 2):
        all_perms = np.unique(np.sort(np.array(
            list(permutations(np.arange(max_num_stations), i))), axis=1), axis=0)
        station_combos = all_perms[rng.choice(all_perms.shape[0],
                                              min(all_perms.shape[0], num_samples),
                                              replace=False), :]
    elif i == max_num_stations - 1:
        station_combos = np.array([np.arange(max_num_stations)
                                   [np.arange(max_num_stations) != j]
                                   for j in range(max_num_stations)])
    elif i == max_num_stations:
        station_combos = np.arange(max_num_stations).reshape(1, -1)
    else:
        station_combos = np.array([rng.choice(max_num_stations, i, replace=False)
                                   for _ in range(num_samples)])
    return station_combos


if __name__ == "__main__":

    # don't let OpenMP mess with NumPy and set parallelization
    os.environ['OMP_NUM_THREADS'] = '1'
    disstans.defaults["general"]["num_threads"] = 30

    # initialize stuff
    rng = np.random.default_rng(0)
    num_samples = 50
    noise_sds = ["0.1", "0.3", "1", "3", "10"]
    penalties = ["1", "10", "30"]
    rw_func_scales = ["1e-2", "1e-1", "1", "10"]

    # create random locations in a uniform square
    max_num_stations = 20
    station_names = [f"S{i:02d}" for i in range(1, max_num_stations + 1)]
    latlons = rng.uniform([-1, -1], [1, 1], (max_num_stations, 2))
    x_range = np.arange(1, max_num_stations+1)

    # create timevector
    t_start_str = "2000-01-01"
    t_end_str = "2001-01-01"
    timevector = pd.date_range(start=t_start_str, end=t_end_str, freq="1D")

    # create an I-Spline model
    ispl = ISpline(degree=2, scale=367/20, t_reference=t_start_str,
                   time_unit="D", num_splines=21)
    # add a reversing transient in the middle
    ispl_params = np.zeros(ispl.num_parameters)
    ispl_params[ispl.num_parameters//2-2] = 1
    ispl_params[ispl.num_parameters//2+2] = -1
    ispl.read_parameters(ispl_params)
    # create all true timeseries
    truth = ispl.evaluate(timevector)["fit"]

    # create single noise realization for all stations
    noise = detrend(rng.normal(scale=1, size=(truth.size, max_num_stations)), axis=0)

    def run_test(noise_sd, pen, rw_func):
        synth = truth + float(noise_sd) * noise

        # create all Station objects with true and synthetic data
        all_stations = {}
        for i, (stat_name, lat, lon) in enumerate(zip(station_names,
                                                      latlons[:, 0], latlons[:, 1])):
            stat = Station(name=stat_name, location=[lat, lon, 0])
            stat.add_timeseries("Truth",
                                Timeseries.from_array(timevector=timevector, data=truth,
                                                      src="synth", data_unit="mm",
                                                      data_cols=["up"]))
            stat.add_timeseries("Synthetic",
                                Timeseries.from_array(timevector=timevector,
                                                      data=synth[:, i].reshape(-1, 1),
                                                      var=float(noise_sd)*np.ones((truth.size, 1)),
                                                      src="synth", data_unit="mm",
                                                      data_cols=["up"]))
            stat.add_local_model("Synthetic", "Transient",
                                 ISpline(degree=1, scale=367/20, t_reference=t_start_str,
                                         time_unit="D", num_splines=21))
            all_stations[stat_name] = stat

        # make a double loop, one about the max number station,
        # and one about sampling the subset of stations
        all_iter_inputs = []
        all_iter_rmses = [[] for _ in range(2, max_num_stations + 1)]
        for i in range(2, max_num_stations + 1):
            # get some possible permutations and create the function input tuples
            station_combos = get_perms(i, max_num_stations, num_samples)
            all_iter_inputs.extend([(station_combos[j, :], station_names, all_stations,
                                     float(pen), rw_func)
                                    for j in range(station_combos.shape[0])])
        for r in tqdm(parallelize(run_single, all_iter_inputs), ascii=True,
                      desc="Looping over station/sample combos", total=len(all_iter_inputs)):
            i = r[2]
            all_iter_rmses[i-2].append(r[0])
            last_net = r[1]

        # since spatialfit doesn't work with one station, do the first fits manually
        # (after the parallel run as to not confuse the all_stations variable)
        net = Network(name="BoxNet")
        for i in range(max_num_stations):
            net.add_station(name=station_names[i], station=all_stations[station_names[i]])
        net.fit("Synthetic", solver="lasso_regression", penalty=float(pen),
                reweight_max_iters=5, reweight_func=rw_func, no_pbar=True)
        net.evaluate("Synthetic", no_pbar=True)
        for stat in net:
            stat["Err_N1"] = stat["Truth"] - stat.fits["Synthetic"]["Transient"]
        rmses_local = [[np.sqrt(np.mean(net[stat_name]["Err_N1"].data.values**2))]
                       for i, stat_name in enumerate(station_names)]

        return [rmses_local] + all_iter_rmses, net, last_net

    # loop over cases
    results = {}
    all_noise_pen_rw_combos = list(product(noise_sds, penalties, rw_func_scales))
    for noise_sd, pen, rw_func_scale in tqdm(all_noise_pen_rw_combos, ascii=True,
                                             desc="Looping over noise/penalty combos"):
        tqdm.write(f"noise={noise_sd}, pen={pen}, rw_func_scale={rw_func_scale}")
        rw_func = InverseReweighting(eps=1e-4, scale=float(rw_func_scale))
        last_rmse, net_local, net_spatial = run_test(noise_sd, pen, rw_func)
        results[f"{noise_sd}_{pen}_{rw_func_scale}"] = last_rmse

        # # save individual result
        # with open(f"{outdir}/S{max_num_stations}N{num_samples}_"
        #           f"{noise_sd}_{pen}_{rw_func_scale}.pkl", "wb") as fp:
        #     pickle.dump(last_rmse, fp)

        # # plot single rmse progression
        # mean_mean = np.array([np.mean(np.mean(rr, axis=1)) for rr in last_rmse])
        # mean_sd = np.array([np.std(np.mean(rr, axis=1)) for rr in last_rmse])
        # fig, ax = plt.subplots()
        # ax.errorbar(x=x_range, y=mean_mean, yerr=mean_sd)
        # ax.set_xlim([0.5, 20.5])
        # ax.set_xticks([1, 5, 10, 15, 20])
        # ax.set_xlabel("Number of Stations")
        # ax.set_ylabel("Mean of Transient RMS Error")
        # ax.set_title(f"Noise S.D. = {noise_sd}, Penalty = {pen}, Scale = {rw_func_scale}")
        # fig.savefig(f"{outdir}/S{max_num_stations}N{num_samples}_"
        #             f"{noise_sd}_{pen}_{rw_func_scale}.png")
        # plt.close(fig)

    # save entire dict
    with open(f"{outdir}/S{max_num_stations}N{num_samples}_dict.pkl", "wb") as fp:
        pickle.dump(results, fp)

    # # load results
    # all_noise_pen_rw_combos = list(product(noise_sds, penalties, rw_func_scales))
    # results = {}
    # for noise_sd, pen, rw_func_scale in all_noise_pen_rw_combos:
    #     with open(f"{outdir}/S{max_num_stations}N{num_samples}_"
    #               f"{noise_sd}_{pen}_{rw_func_scale}.pkl", "rb") as fp:
    #         results[f"{noise_sd}_{pen}_{rw_func_scale}"] = pickle.load(fp)

    # plot for each penalty
    x_range_small = np.linspace(0.5, max_num_stations+0.5, num=100)
    cycleable_colors = ["#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252",
                        "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94",
                        "#843c39", "#ad494a", "#d6616b", "#e7969c", "#7b4173", "#a55194",
                        "#ce6dbd", "#de9ed6", "#3182bd", "#6baed6", "#9ecae1", "#c6dbef",
                        "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2", "#31a354", "#74c476",
                        "#a1d99b", "#c7e9c0", "#756bb1", "#9e9ac8", "#bcbddc", "#dadaeb",
                        "#636363", "#969696", "#bdbdbd", "#d9d9d9"]  # tab20b + tab20c
    for pen in penalties:
        fig, ax = plt.subplots(figsize=(6, 8))
        for i, noise_sd in enumerate(noise_sds):
            for j, rw_func_scale in enumerate(rw_func_scales):
                col = cycleable_colors[i*4 + j]
                ecol = col + "30"
                last_rmse = results[f"{noise_sd}_{pen}_{rw_func_scale}"]
                mean_mean = np.array([np.mean(np.mean(rr, axis=1)) for rr in last_rmse])
                mean_sd = np.array([np.std(np.mean(rr, axis=1)) for rr in last_rmse])
                ax.errorbar(x=x_range, y=mean_mean, yerr=mean_sd, color=col, ecolor=ecol,
                            label=f"$\\sigma$={noise_sd}, $\\alpha$={rw_func_scale}")
        ax.plot(x_range_small, 1/np.sqrt(x_range_small) * 2, c="0.5", ls=":", lw=1)
        ax.axhline(np.sqrt(np.mean(truth**2)), color="0.5", linewidth=1, linestyle="--", zorder=-1)
        ax.set_xscale("log")
        ax.set_xlim([0.8, max_num_stations + 3])
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_yscale("log")
        ax.set_ylim([1e-2, 3])
        ax.set_yticks([0.01, 0.1, 1])
        ax.set_yticklabels(["0.01", "0.1", "1"])
        ax.set_xlabel("Number of Stations")
        ax.set_ylabel("Mean of Transient RMS Error")
        ax.set_title(f"Penalty = {pen}")
        ax.legend(ncol=len(noise_sds), loc="upper right", fontsize=6)
        fig.savefig(f"{outdir}/tutorial_5_S{max_num_stations}N{num_samples}_{pen}.png", dpi=300)
        fig.savefig(f"{outdir}/tutorial_5_S{max_num_stations}N{num_samples}_{pen}.pdf", dpi=300)
        plt.close(fig)

    # use only j=1, rw_func_scale=1e-1 for single plot
    pen = penalties[0]
    j = 1
    rw_func_scale = rw_func_scales[j]
    fig, ax = plt.subplots(figsize=(4, 4))
    for i, noise_sd in enumerate(noise_sds):
        if i == 0:
            continue
        col = cycleable_colors[i*4 + j]
        ecol = col + "30"
        last_rmse = results[f"{noise_sd}_{pen}_{rw_func_scale}"]
        mean_mean = np.array([np.mean(np.mean(rr, axis=1)) for rr in last_rmse])
        mean_sd = np.array([np.std(np.mean(rr, axis=1)) for rr in last_rmse])
        ax.errorbar(x=x_range, y=mean_mean, yerr=mean_sd, color=col, ecolor=ecol,
                    label=f"$\\sigma$={noise_sd}")
    ax.plot(x_range_small, 1/np.sqrt(x_range_small) * 2, c="0.5", ls=":", lw=1)
    ax.axhline(np.sqrt(np.mean(truth**2)), color="0.5", linewidth=1, linestyle="--", zorder=-1)
    ax.set_xscale("log")
    ax.set_xlim([0.8, max_num_stations + 3])
    ax.set_xticks([1, 5, 10, 20])
    ax.set_xticklabels(["1", "5", "10", "20"])
    ax.set_yscale("log")
    ax.set_ylim([0.02, 3])
    ax.set_yticks([0.1, 1])
    ax.set_yticklabels(["0.1", "1"])
    ax.set_xlabel("Number of Stations")
    ax.set_ylabel("Mean of Transient RMS Error")
    ax.legend(ncol=2, loc="upper right", fontsize=10)
    fig.savefig(f"{outdir}/tutorial_5_single.png", dpi=300)
    fig.savefig(f"{outdir}/tutorial_5_single.pdf", dpi=300)
    plt.close(fig)
