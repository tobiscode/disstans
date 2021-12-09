"""
First example of the disstans documentation:
Long Valley Caldera

This example aims to demonstrate the downloading of timeseries from UNR's
data repository and modeling spatiotemporally differing seasonal and
transient signals.
"""

# import the modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import lstsq
from matplotlib.lines import Line2D
import disstans
from disstans.tools import tvec_to_numpycol, full_cov_mat_to_columns

# make it look better
from matplotlib import rcParams
rcParams['font.sans-serif'] = ["NewComputerModernSans10"]
rcParams['font.size'] = "14"
disstans.defaults["gui"]["wmts_show"] = True
disstans.defaults["gui"]["wmts_alpha"] = 0.5
fmt = "png"

# Python is petty about this format, this is somehow super important
if __name__ == "__main__":

    # preparation
    main_dir = Path("proj_dir").resolve()
    data_dir = main_dir / "data/gnss"
    gnss_dir = data_dir / "longvalley"
    plot_dir = Path("../img").resolve()
    os.makedirs(plot_dir, exist_ok=True)
    os.chdir(plot_dir)

    # let's have multiprocessing
    os.environ['OMP_NUM_THREADS'] = '1'
    disstans.defaults["general"]["num_threads"] = 30

    # let's define a region of interest
    center_lon = -118.884167  # [°]
    center_lat = 37.716667  # [°]
    radius = 100  # [km]
    station_bbox = [center_lon, center_lat, radius]

    # # then download the data for the stations within
    # stations_df = disstans.tools.download_unr_data(station_bbox, gnss_dir,
    #                                                min_solutions=600, verbose=2)
    # stations_df.to_pickle(f"{gnss_dir}/downloaded.pkl.gz")

    # if not redownloading, just load the subset DataFrame
    stations_df = pd.read_pickle(f"{gnss_dir}/downloaded.pkl.gz")

    # create an empty Network instance
    net = disstans.Network("LVC")

    # add the stations we downloaded
    for _, row in stations_df.iterrows():
        # get name and location of station
        name = row["Sta"]
        loc = [row["Lat(deg)"], row["Long(deg)"], row["Hgt(m)"]]
        # make a timeseries object to check availability metric
        tspath = f"{gnss_dir}/{name}.tenv3"
        loaded_ts = disstans.timeseries.UNRTimeseries(tspath)
        # make a station and add the timeseries only if two quality metrics are met
        if (loaded_ts.reliability > 0.5) and (loaded_ts.length > pd.Timedelta(365, "D")):
            net[name] = disstans.Station(name, loc)
            net[name]["raw"] = loaded_ts

    # print the station map and example timeseries
    net.gui(save=True, save_map=True,
            station="CASA",
            annotate_stations=False,
            save_kw_args={"format": fmt, "dpi": 300})
    os.rename(next(plot_dir.glob(f"ts_CASA*.{fmt}")),
              (plot_dir / f"example_1a_ts.{fmt}"))
    os.rename(next(plot_dir.glob(f"map_CASA*.{fmt}")),
              (plot_dir / f"example_1a_map.{fmt}"))

    # low-pass filter using a median function, then clean the timeseries
    # (using the settings in the config file)
    # compute reference
    net.call_func_ts_return("median", ts_in="raw", ts_out="raw_filt", kernel_size=7)
    # remove outliers
    net.call_func_no_return("clean", ts_in="raw", reference="raw_filt", ts_out="raw_clean")
    # get the residual for each station
    net.math("raw_filt_res", "raw_clean", "-", "raw_filt")
    # remove obsolete timeseries
    net.remove_timeseries("raw_filt")

    # estimate the common mode, either with a visualization of the result or not
    # (same underlying function)
    # net.graphical_cme(ts_in="raw_filt_res", ts_out="common", method="ica")
    # calculate common mode
    net.call_netwide_func("decompose", ts_in="raw_filt_res", ts_out="common", method="ica")
    # now remove the common mode, call it the "intermed" timeseries,
    for station in net:
        station.add_timeseries("intermed", station["raw_clean"] - station["common"],
                               override_data_cols=station["raw"].data_cols)
    # remove obsolete timeseries
    net.remove_timeseries("common", "raw_clean")

    # clean again
    net.call_func_ts_return("median", ts_in="intermed",
                            ts_out="intermed_filt", kernel_size=7)
    net.call_func_no_return("clean", ts_in="intermed",
                            reference="intermed_filt", ts_out="final")
    net.remove_timeseries("intermed", "intermed_filt")

    # use uncertainty data from raw timeseries
    net.copy_uncertainties(origin_ts="raw", target_ts="final")

    # add simple models
    models = {"Annual": {"type": "Sinusoid",
                         "kw_args": {"period": 365.25,
                                     "t_reference": "2000-01-01"}},
              "Biannual": {"type": "Sinusoid",
                           "kw_args": {"period": 365.25/2,
                                       "t_reference": "2000-01-01"}},
              "Linear": {"type": "Polynomial",
                         "kw_args": {"order": 1,
                                     "t_reference": "2000-01-01",
                                     "time_unit": "Y"}}}
    net.add_local_models(models=models, ts_description="final")

    # first fit (no earthquakes, no maintenance, no transients, no regularized models,
    # so just linear regression)
    net.fitevalres("final", solver="linear_regression",
                   use_data_covariance=False, output_description="model_noreg",
                   residual_description="resid_noreg")

    # run the step detector to find any steps
    stepdet = disstans.processing.StepDetector(kernel_size=61, kernel_size_min=21)
    step_table, _ = stepdet.search_network(net, "resid_noreg")

    # quick view of step_table
    print(step_table)

    # look at a map to gauge a good cutoff for places with extreme residuals,
    # and therefore what a good first cutoff is for steps
    # I already know this data a little so we can already go for varred > 0.9
    # (step detector is way too sensitive)
    step_table_above90 = step_table[step_table["varred"] > 0.9]
    print(step_table_above90)
    # net.gui(station="CASA", timeseries="final",
    #         rms_on_map={"ts": "resid_noreg"},
    #         mark_events=step_table_above90)

    # save figures into main plotting folder
    for stat_name in step_table_above90["station"].unique():
        net.gui(station=stat_name, timeseries="final",
                mark_events=step_table_above90, save=True,
                save_kw_args={"format": fmt, "dpi": 300})
        os.rename(next(plot_dir.glob(f"ts_{stat_name}*.{fmt}")),
                  (plot_dir / f"example_1b_{stat_name}.{fmt}"))

    # delete noisy parts or add models manually
    net["TILC"].add_local_model(ts_description="final",
                                model_description="Unknown",
                                model=disstans.models.Step(["2008-07-26"]))
    net["DOND"].add_local_model(ts_description="final",
                                model_description="Unknown",
                                model=disstans.models.Step(["2016-04-20"]))
    net["WATC"].add_local_model(ts_description="final",
                                model_description="Unknown",
                                model=disstans.models.Step(["2002-04-04", "2002-06-18"]))
    net["LINC"]["final"].cut(t_min="1998-09-13", keep_inside=True)
    net["P636"]["final"].cut(t_min="2011-08-03", t_max="2011-09-14", keep_inside=False)
    del net["P628"]["final"]

    # now, let's make a really long-term transient before we check again for steps
    # to make sure we're not getting contaminated by big signals
    longterm_transient_mdl = \
        {"Longterm": {"type": "SplineSet",
                      "kw_args": {"degree": 2,
                                  "t_center_start": net["CASA"]["final"].time.min(),
                                  "t_center_end": net["CA99"]["final"].time.max(),
                                  "list_num_knots": [5, 9]}}}
    net.add_local_models(models=longterm_transient_mdl, ts_description="final")

    # let's fit everything again, without regularization
    net.fitevalres("final", solver="linear_regression",
                   use_data_covariance=False, output_description="model_noreg_2",
                   residual_description="resid_noreg_2")

    # get residuals overview
    resids_df = net.analyze_residuals("resid_noreg_2", rms=True)
    resids_df["total"] = np.linalg.norm(resids_df.values, axis=1)
    resids_df.sort_values("total", inplace=True, ascending=False)
    print(resids_df["total"].head())

    # # quick look
    # net.gui(station="CASA", timeseries="final", rms_on_map={"ts": "resid_noreg_2"})

    # save figures of top-5 worst residual timeseries
    for stat_name in resids_df.index[:5].tolist():
        net.gui(station=stat_name, timeseries="final", save=True,
                save_kw_args={"format": fmt, "dpi": 300})
        os.rename(next(plot_dir.glob(f"ts_{stat_name}*.{fmt}")),
                  (plot_dir / f"example_1c_{stat_name}.{fmt}"))

    # cut again
    net["P723"]["final"].cut(t_min="2010-12-18", t_max="2011-04-18", keep_inside=False)
    net["P723"]["final"].cut(t_min="2017-01-09", t_max="2017-05-24", keep_inside=False)
    net["P723"]["final"].cut(t_min="2019-02-02", t_max="2019-04-02", keep_inside=False)
    net["P723"]["final"].cut(t_min="2019-12-02", t_max="2020-04-02", keep_inside=False)
    net["MUSB"]["final"].cut(t_min="1998-02-15", t_max="1998-04-19", keep_inside=False)
    net["KNOL"]["final"].cut(t_min="2017-01-22", t_max="2017-03-16", keep_inside=False)

    # other places where we remove the first couple of (very uncertain) observations
    net["MINS"]["final"].cut(t_min="1997-06-01", keep_inside=True)
    net["MWTP"]["final"].cut(t_min="1999-01-01", keep_inside=True)
    net["KNOL"]["final"].cut(t_min="1999-01-01", keep_inside=True)
    net["RDOM"]["final"].cut(t_min="1999-09-01", keep_inside=True)
    net["SHRC"]["final"].cut(t_min="2006-03-01", keep_inside=True)

    # let's fit everything again, without regularization
    net.fitevalres("final", solver="linear_regression",
                   use_data_covariance=False, output_description="model_noreg_3",
                   residual_description="resid_noreg_3")

    # before we plot again, we want to run the step detector again
    step_table, _ = stepdet.search_network(net, "resid_noreg_3")

    # download and parse step file
    # only set check_update to True when downloading new data
    unr_maint_table, _, unr_eq_table, _ = \
        disstans.tools.parse_unr_steps(f"{data_dir}/unr_steps.txt",
                                       verbose=True, check_update=False,
                                       only_stations=net.station_names)

    # use the step detector to also assess the step probabilites for the
    # possible earthquake- and maintenance-related events, including reduced variance
    maint_table, _ = stepdet.search_catalog(net, "resid_noreg_3", unr_maint_table)
    eq_table, _ = stepdet.search_catalog(net, "resid_noreg_3", unr_eq_table)

    # make a new merged table with steps that are not in maint_table or eq_table
    # merge the two catalog tables
    maint_or_eq = pd.merge(maint_table[["station", "time"]],
                           eq_table[["station", "time"]], how="outer")
    # merge with step_table
    merged_table = step_table.merge(maint_or_eq, on=["station", "time"], how="left",
                                    indicator="merged")
    # drop rows where the indicators are not only in step_table
    unknown_table = merged_table. \
        drop(merged_table[merged_table["merged"] != "left_only"].index)

    # the dropped rows are
    print(merged_table[merged_table["merged"] != "left_only"])

    # make a plot of the curve of probability values to find an L-curve
    # inflection point
    plt.plot(np.arange(unknown_table.shape[0])/unknown_table.shape[0],
             unknown_table["probability"].values, label="Unknown")
    plt.plot(np.arange(maint_table.shape[0]) /
             np.isfinite(maint_table["probability"].values).sum(),
             maint_table["probability"].values, label="Maintenance")
    plt.plot(np.arange(eq_table.shape[0]) /
             np.isfinite(eq_table["probability"].values).sum(),
             eq_table["probability"].values, label="Earthquake")
    plt.ylabel("Probability")
    plt.xlabel("Normalized number of events")
    plt.xticks(ticks=[], labels=[])
    plt.legend()
    plt.savefig(plot_dir / f"example_1d.{fmt}")
    plt.close()

    # # show the GUI for the unknown events
    # # similar for maint_table or unknown_table
    # net.gui(station="CASA", timeseries="final", rms_on_map={"ts": "resid_noreg_3"},
    #         mark_events=eq_table[eq_table["probability"] > 10])

    # add the steps
    eq_steps_dict = dict(eq_table[eq_table["probability"] > 15]
                         .groupby("station")["time"].unique().apply(list))
    for stat, steptimes in eq_steps_dict.items():
        net[stat].add_local_model_kwargs(
            ts_description="final",
            model_kw_args={"Earthquake": {"type": "Step",
                                          "kw_args": {"steptimes": steptimes}}})
    maint_steps_dict = dict(maint_table[maint_table["probability"] > 15]
                            .groupby("station")["time"].unique().apply(list))
    for stat, steptimes in maint_steps_dict.items():
        net[stat].add_local_model_kwargs(
            ts_description="final",
            model_kw_args={"Maintenance": {"type": "Step",
                                           "kw_args": {"steptimes": steptimes}}})

    # add step-and-reverse model as a special way of using Polynomial (could also use Step)
    net["KRAC"].add_local_model_kwargs(
        ts_description="final",
        model_kw_args={"Offset": {"type": "Polynomial",
                                  "kw_args": {"order": 0,
                                              "t_start": "2002-02-17",
                                              "t_reference": "2002-02-17",
                                              "t_end": "2002-03-17",
                                              "zero_before": True,
                                              "zero_after": True}}})

    # don't need the longterm transient model anymore
    for stat in net:
        stat.remove_local_models("final", "Longterm")

    # define and add transient and new seasonal models
    new_models = \
        {"Transient": {"type": "SplineSet",
                       "kw_args": {"degree": 2,
                                   "t_center_start": net["CASA"]["final"].time.min(),
                                   "t_center_end": net["CA99"]["final"].time.max(),
                                   "list_num_knots": [int(1+2**n) for n in range(4, 8)]}},
         "AnnualDev": {"type": "AmpPhModulatedSinusoid",
                       "kw_args": {"period": 365.25,
                                   "degree": 2,
                                   "num_bases": 29,
                                   "t_start": "1994-01-01",
                                   "t_end": "2022-01-01"}},
         "BiannualDev": {"type": "AmpPhModulatedSinusoid",
                         "kw_args": {"period": 365.25/2,
                                     "degree": 2,
                                     "num_bases": 29,
                                     "t_start": "1994-01-01",
                                     "t_end": "2022-01-01"}}}
    net.add_local_models(new_models, "final")

    # make a single solution
    rw_func = disstans.solvers.InverseReweighting(eps=1e-4, scale=1e-2)
    stats = net.spatialfit("final",
                           penalty=[10, 10, 1],
                           spatial_reweight_models=["Transient"],
                           spatial_reweight_iters=20,
                           local_reweight_func=rw_func,
                           formal_covariance=True,
                           use_data_covariance=True,
                           verbose=True,
                           extended_stats=True,
                           keep_mdl_res_as=("model_srw", "resid_srw"))

    # make CASA plots
    net.gui(station="CASA", timeseries="final", save=True,
            save_kw_args={"format": fmt, "dpi": 300},
            scalogram_kw_args={"ts": "final", "model": "Transient", "cmaprange": 60})
    os.rename(next(plot_dir.glob(f"ts_CASA*.{fmt}")),
              (plot_dir / f"example_1e_ts.{fmt}"))
    os.rename(next(plot_dir.glob(f"scalo_CASA*.{fmt}")),
              (plot_dir / f"example_1e_scalo.{fmt}"))
    net.gui(station="CASA", timeseries="final", save=True,
            save_kw_args={"format": fmt, "dpi": 300},
            sum_models=False, fit_list=["Transient"], gui_kw_args={"plot_sigmas": 0})
    os.rename(next(plot_dir.glob(f"ts_CASA*.{fmt}")),
              (plot_dir / f"example_1e_transient.{fmt}"))
    net.gui(station="CASA", timeseries="final", save=True,
            save_kw_args={"format": fmt, "dpi": 300}, sum_models=True,
            fit_list=["Annual", "AnnualDev", "Biannual", "BiannualDev"],
            gui_kw_args={"plot_sigmas": 0})
    os.rename(next(plot_dir.glob(f"ts_CASA*.{fmt}")),
              (plot_dir / f"example_1e_seasonal.{fmt}"))
    net["CASA"].models["final"].plot_covariance(fname=plot_dir / "example_1e_corr",
                                                save_kw_args={"format": fmt, "dpi": 300},
                                                plot_empty=False, use_corr_coef=True)

    # save secular parameters
    poly_stat_names = [stat_name for stat_name in net.station_names
                       if "final" in net[stat_name].models]
    # use the covariance estimates
    all_poly_pars_covs = np.concatenate((
        np.stack([net[stat_name].models["final"]["Linear"].par.ravel()
                  for stat_name in poly_stat_names], axis=0),
        np.stack([full_cov_mat_to_columns(net[stat_name].models["final"]["Linear"].cov,
                                          num_components=3, include_covariance=True,
                                          return_single=True)[1, :]
                  for stat_name in poly_stat_names], axis=0)), axis=1)
    # # if formal_covariance=False and/or use_data_covariance=False, set covariances to zero
    # all_poly_pars_covs = np.concatenate((
    #     np.stack([net[stat_name].models["final"]["Linear"].par.ravel()
    #                 for stat_name in poly_stat_names], axis=0),
    #     np.zeros((len(poly_stat_names), 6))), axis=1)
    all_poly_df = pd.DataFrame(data=all_poly_pars_covs, index=poly_stat_names,
                               columns=["off_e", "off_n", "off_u", "vel_e", "vel_n", "vel_u",
                                        "sig_vel_e", "sig_vel_n", "sig_vel_u", "corr_vel_en",
                                        "corr_vel_eu", "corr_vel_nu"])
    all_poly_df[["sig_vel_e", "sig_vel_n", "sig_vel_u"]] **= 0.5
    all_poly_df["corr_vel_en"] /= (all_poly_df["sig_vel_e"] * all_poly_df["sig_vel_n"])
    all_poly_df["corr_vel_eu"] /= (all_poly_df["sig_vel_e"] * all_poly_df["sig_vel_u"])
    all_poly_df["corr_vel_nu"] /= (all_poly_df["sig_vel_n"] * all_poly_df["sig_vel_u"])
    all_poly_df.to_csv(plot_dir / "secular_velocities.csv", index_label="station")

    # make wormplot
    subset_stations = ["RDOM", "KRAC", "SAWC", "MWTP", "CASA", "CA99", "P639", "HOTK",
                       "P646", "P638", "DDMN", "P634", "KNOL", "MINS", "LINC", "P630",
                       "SHRC", "P631", "TILC", "P642", "BALD", "P648", "WATC", "P632",
                       "P643", "P647", "PMTN", "P635", "P645"]
    plot_t_start, plot_t_end = "2012-01-01", "2015-01-01"
    # if colorbar is added, the animation is a bit buggy, so make the animation
    # without it, then overwrite the final image with one that has a colorbar
    net.wormplot(ts_description=("final", "Transient"),
                 fname=plot_dir / "example_1f",
                 save_kw_args={"format": fmt, "dpi": 300},
                 fname_animation=plot_dir / "example_1f.mp4",
                 t_min=plot_t_start, t_max=plot_t_end, scale=2e2,
                 subset_stations=subset_stations,
                 lat_min=37.52, lat_max=37.87, lon_min=-119.18, lon_max=-118.56,
                 annotate_stations="small",
                 legend_ref_dict={"location": [-118.685, 37.832],
                                  "length": 30,
                                  "label": "30 mm",
                                  "rect_args": [(-118.7, 37.8), 0.1, 0.05],
                                  "rect_kw_args": {"facecolor": [1, 1, 1, 0.15],
                                                   "edgecolor": [0, 0, 0, 0.6]}})
    net.wormplot(ts_description=("final", "Transient"),
                 fname=plot_dir / "example_1f",
                 save_kw_args={"format": fmt, "dpi": 300},
                 t_min=plot_t_start, t_max=plot_t_end, scale=2e2,
                 subset_stations=subset_stations,
                 lat_min=37.52, lat_max=37.87, lon_min=-119.18, lon_max=-118.56,
                 annotate_stations="small",
                 colorbar_kw_args={"shrink": 0.5},
                 legend_ref_dict={"location": [-118.685, 37.832],
                                  "length": 30,
                                  "label": "30 mm",
                                  "rect_args": [(-118.7, 37.8), 0.1, 0.05],
                                  "rect_kw_args": {"facecolor": [1, 1, 1, 0.15],
                                                   "edgecolor": [0, 0, 0, 0.6]}})

    # print some secular linear velocities
    print(f"{'Station':>10s} {'East [m/a]':>12s} {'North [m/a]':>12s} {'Up [m/a]':>12s}")
    for stat_name in ["P308", "DOND", "KRAC", "CASA", "CA99", "P724", "P469", "P627"]:
        ve, vn, vu = net[stat_name].models["final"]["Linear"].par[1, :] / 1000
        print(f"{stat_name:>10s} {ve:>12f} {vn:>12f} {vu:>12f}")

    # define rotation matrix function
    def R(angle):
        angle = np.deg2rad(angle)
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # make a plot of the displacement in the dominant direction during the timespan
    # of the wormplot
    # first, get relevant timeseries
    lvc_names = ["TILC", "SHRC", "MWTP", "KNOL", "CASA", "CA99", "DDMN",
                 "BALD", "SAWC", "HOTK", "RDOM", "KRAC", "LINC", "WATC"]
    # now, get quick access to data
    lvc_data = {name: net[name]["final"].data for name in lvc_names}
    # quick access to Linear + Unknown + Earthquake + Maintenance + Offset
    lvc_background = {name: net[name].fits["final"]["Linear"].data.values.copy()
                      for name in lvc_names}
    for name in lvc_names:
        for test_ts in ["Unknown", "Earthquake", "Maintenance", "Offset"]:
            if test_ts in net[name].fits["final"]:
                lvc_background[name] += net[name].fits["final"][test_ts].data.values
    # quick access to transients and seasonals
    lvc_transient = {name: net[name].fits["final"]["Transient"].data.values
                     for name in lvc_names}
    lvc_seasonal = {name: (net[name].fits["final"]["Annual"].data.values +
                           net[name].fits["final"]["Biannual"].data.values +
                           net[name].fits["final"]["AnnualDev"].data.values +
                           net[name].fits["final"]["BiannualDev"].data.values)
                    for name in lvc_names}
    # remove fits from data
    lvc_data_transient = {name: (lvc_data[name].values - lvc_background[name] -
                                 lvc_seasonal[name])
                          for name in lvc_names}
    lvc_data_seasonal = {name: (lvc_data[name].values - lvc_background[name] -
                                lvc_transient[name])
                         for name in lvc_names}

    # get the main direction of the transient displacements from the fits
    rots = {}
    for name in lvc_names:
        if name == "CASA":
            continue
        intimespan = np.logical_and(pd.Timestamp(plot_t_start) <= net[name]["final"].time,
                                    net[name]["final"].time <= pd.Timestamp(plot_t_end))
        finitedata = np.all(np.isfinite(lvc_data_transient[name][:, :2]), axis=1)
        valid = np.logical_and(intimespan, finitedata)
        G = np.ones((lvc_data[name].index.size, 2))
        G[:, 1] = tvec_to_numpycol(lvc_data[name].index)
        G = G[valid, :]
        GtWG = G.T @ G
        GtWd_E = G.T @ lvc_data_transient[name][:, 0][valid]
        GtWd_N = G.T @ lvc_data_transient[name][:, 1][valid]
        trend = [lstsq(GtWG, GtWd_E)[0][1], lstsq(GtWG, GtWd_N)[0][1]]
        rots[name] = (np.rad2deg(np.arctan2(trend[1], trend[0])) + 360) % 360
    # give CASA the CA99 direction because it doesn't have data in that timespan
    rots["CASA"] = rots["CA99"]
    rots_keys = list(rots.keys())
    # sort by azimuth
    rots_sort_azim = dict(sorted(rots.items(), key=lambda kv: kv[1]))

    # rotate east-north to main direction
    lvc_data_main = {name: (lvc_data[name].values[:, :2] @ R(rots[name]))[:, 0]
                     for name in lvc_names}
    lvc_background_main = {name: (lvc_background[name][:, :2] @ R(rots[name]))[:, 0]
                           for name in lvc_names}
    lvc_transient_main = {name: (lvc_transient[name][:, :2] @ R(rots[name]))[:, 0]
                          for name in lvc_names}
    lvc_data_transient_main = {name: (lvc_data_transient[name][:, :2] @ R(rots[name]))[:, 0]
                               for name in lvc_names}
    # start transient plot by azimuth
    offsets = [0, 65, 80, 110, 140, 180, 235, 265, 295, 285, 380, 395, 425, 455]
    fig, ax = plt.subplots(figsize=(5, 8))
    for i, name in enumerate(rots_sort_azim.keys()):
        ax.plot(lvc_data[name].index,
                lvc_data_transient_main[name] - lvc_data_transient_main[name][0] + offsets[i],
                "k.", markersize=3, rasterized=True)
        ax.plot(lvc_data[name].index,
                lvc_transient_main[name] - lvc_data_transient_main[name][0] + offsets[i])
        if name == "CASA":
            ax.annotate(name, (lvc_data[name].index[0] - pd.Timedelta(50, "D"), offsets[i] + 30),
                        ha="left", va="bottom")
        elif name == "CA99":
            ax.annotate(name, (lvc_data[name].index[0] - pd.Timedelta(50, "D"), offsets[i] + 10),
                        ha="center", va="bottom")
        else:
            ax.annotate(name, (lvc_data[name].index[0] - pd.Timedelta(100, "D"), offsets[i]),
                        ha="right", va="center")
        if name != "CASA":
            ax.annotate(f"{rots[name]:.0f}°",
                        (lvc_data[name].index[-1] + pd.Timedelta(100, "D"),
                         lvc_transient_main[name][-1] - lvc_data_transient_main[name][0]
                         + offsets[i]),
                        ha="left", va="center", color="0.3")
    ax.set_xlim([pd.Timestamp("1993-01-01"), pd.Timestamp("2025-01-01")])
    yearticks = ["1996", "2000", "2004", "2008", "2012", "2016", "2020"]
    ax.set_xticks([pd.Timestamp(f"{year}-01-01") for year in yearticks])
    ax.set_xticklabels(yearticks)
    ax.set_xlabel("Time")
    ax.set_ylabel("Displacement [mm]")
    ax.grid(axis="x", color="0.8")
    fig.savefig(plot_dir / f"example_1g_expansion_azim.{fmt}", dpi=300)
    plt.close(fig)

    # for all plotted stations from the wormplot, get the transient, detrended and
    # seasonal model fits, as well as the residuals
    ylabels_data = ["East [mm]", "North [mm]", "Up [mm]"]
    ylabels_comp = ["Nominal [mm]", "Deviation [mm]", "Total [mm]"]
    for stat_name in subset_stations:
        # get the noisy data, subtract seasonal, secular and steps
        temp_ts = net[stat_name]["final"]
        temp_fits = net[stat_name].fits["final"]
        temp_mdl = net[stat_name].models["final"]
        seas_ann = temp_fits["Annual"].data.values + temp_fits["AnnualDev"].data.values
        seas_biann = temp_fits["Biannual"].data.values + temp_fits["BiannualDev"].data.values
        trans_data = (temp_ts.data.values - seas_ann - seas_biann -
                      temp_fits["Linear"].data.values)
        trans_fit = temp_fits["Transient"].data.values.copy()
        detrend_data = temp_ts.data.values - temp_fits["Linear"].data.values
        detrend_fit = temp_fits.allfits.data.values - temp_fits["Linear"].data.values
        temp_offsets = []
        for other_model in ["Unknown", "Earthquake", "Maintenance", "Offset"]:
            if other_model in temp_fits:
                trans_fit += temp_fits[other_model].data.values
                detrend_fit -= temp_fits[other_model].data.values
                detrend_data -= temp_fits[other_model].data.values
                if other_model == "Offset":
                    temp_offsets.extend([temp_mdl[other_model].t_start,
                                         temp_mdl[other_model].t_end])
                else:
                    temp_offsets.extend(temp_mdl[other_model].steptimes)
        # plot transient
        fig, ax = plt.subplots(nrows=3, sharex=True)
        for i in range(3):
            for step in temp_offsets:
                ax[i].axvline(pd.Timestamp(step), color="C0", linewidth=1, zorder=-1)
            ax[i].plot(temp_ts.time, trans_data[:, i], rasterized=True,
                       c="k", ls="none", marker=".", markersize=1)
            ax[i].plot(temp_ts.time, trans_fit[:, i], c="C1")
            ax[i].set_ylabel(ylabels_data[i])
        ax[2].set_xlabel("Time")
        custom_lines = [Line2D([0], [0], c="k", marker=".", markersize=5, linestyle='none'),
                        Line2D([0], [0], c="C1")]
        ax[2].legend(custom_lines, ["Transient + Steps Residual", "Transient + Steps Model"])
        fig.suptitle(stat_name)
        fig.savefig(plot_dir / f"example_1h_transient_{stat_name}.{fmt}", dpi=300)
        plt.close(fig)
        # plot detrended fit
        fig, ax = plt.subplots(nrows=3, sharex=True)
        for i in range(3):
            for step in temp_offsets:
                ax[i].axvline(pd.Timestamp(step), color="C0", linewidth=1, zorder=-1)
            ax[i].plot(temp_ts.time, detrend_data[:, i], rasterized=True,
                       c="k", ls="none", marker=".", markersize=1)
            ax[i].plot(temp_ts.time, detrend_fit[:, i], c="C2")
            ax[i].set_ylabel(ylabels_data[i])
        ax[2].set_xlabel("Time")
        custom_lines = [Line2D([0], [0], c="k", marker=".", markersize=5, linestyle='none'),
                        Line2D([0], [0], c="C2")]
        ax[2].legend(custom_lines, ["Detrended Residual", "Detrended Model"])
        fig.suptitle(stat_name)
        fig.savefig(plot_dir / f"example_1h_detrended_{stat_name}.{fmt}", dpi=300)
        plt.close(fig)
        # plot seasonal
        fig, ax = plt.subplots(nrows=3, sharex=True)
        seas_max = np.ceil(np.amax(np.abs(seas_ann + seas_biann)))
        for i in range(3):
            ax[i].plot(temp_ts.time, seas_ann[:, i], "C0", alpha=0.5)
            ax[i].plot(temp_ts.time, seas_biann[:, i], "C1", alpha=0.5)
            ax[i].plot(temp_ts.time, seas_ann[:, i] + seas_biann[:, i], "k")
            ax[i].set_ylim(-seas_max, seas_max)
            ax[i].set_ylabel(ylabels_data[i])
        ax[2].set_xlabel("Time")
        ax[2].legend([Line2D([0], [0], c="C0", alpha=0.5), Line2D([0], [0], c="C1", alpha=0.5),
                      Line2D([0], [0], c="k")], ["Annual", "Biannual", "Sum"])
        fig.suptitle(stat_name)
        fig.savefig(plot_dir / f"example_1h_seasonal_{stat_name}.{fmt}", dpi=300)
        plt.close(fig)
        # plot seasonal by component
        for i in range(3):
            fig, ax = plt.subplots(nrows=3, sharex=True)
            seas_max = np.ceil(np.amax(np.abs(seas_ann[:, i] + seas_biann[:, i])))
            ax[0].plot(temp_ts.time, temp_fits["Annual"].data.values[:, i], "C0", alpha=0.5)
            ax[0].plot(temp_ts.time, temp_fits["Biannual"].data.values[:, i], "C1", alpha=0.5)
            ax[0].plot(temp_ts.time, temp_fits["Annual"].data.values[:, i] +
                       temp_fits["Biannual"].data.values[:, i], "k")
            ax[1].plot(temp_ts.time, temp_fits["AnnualDev"].data.values[:, i], "C0", alpha=0.5)
            ax[1].plot(temp_ts.time, temp_fits["BiannualDev"].data.values[:, i], "C1", alpha=0.5)
            ax[1].plot(temp_ts.time, temp_fits["AnnualDev"].data.values[:, i] +
                       temp_fits["BiannualDev"].data.values[:, i], "k")
            ax[2].plot(temp_ts.time, temp_fits["Annual"].data.values[:, i] +
                       temp_fits["AnnualDev"].data.values[:, i], "C0", alpha=0.5)
            ax[2].plot(temp_ts.time, temp_fits["Biannual"].data.values[:, i] +
                       temp_fits["BiannualDev"].data.values[:, i], "C1", alpha=0.5)
            ax[2].plot(temp_ts.time, seas_ann[:, i] + seas_biann[:, i], "k")
            for j in range(3):
                ax[j].set_ylim(-seas_max, seas_max)
                ax[j].set_ylabel(ylabels_comp[j])
            ax[2].set_xlabel("Time")
            ax[1].legend([Line2D([0], [0], c="C0", alpha=0.5), Line2D([0], [0], c="C1", alpha=0.5),
                         Line2D([0], [0], c="k")], ["Annual", "Biannual", "Sum"])
            fig.suptitle(f"{stat_name}: {ylabels_data[i][:-5]}")
            fig.savefig(plot_dir / (f"example_1h_seasonal_{stat_name}_" +
                                    f"{ylabels_data[i][:-5].lower()}.{fmt}"), dpi=300)
            plt.close(fig)
        # plot residuals
        fig, ax = plt.subplots(nrows=3, sharex=True)
        for i in range(3):
            ax[i].plot(temp_ts.time, net[stat_name]["resid_srw"].data.values[:, i],
                       linestyle="None", color="0.3", marker="o", markersize=1)
            ax[i].set_ylabel(ylabels_data[i])
        ax[2].set_xlabel("Time")
        fig.suptitle(stat_name)
        fig.savefig(plot_dir / f"example_1h_residuals_{stat_name}.{fmt}", dpi=300)
        plt.close(fig)
