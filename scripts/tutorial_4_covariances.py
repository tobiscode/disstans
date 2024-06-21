"""
This is a tutorial creating a synthetic network to test the use of data
variance-covariance and estimating the formal model error associated with the
different components.
"""

if __name__ == "__main__":

    # don't let OpenMP mess with NumPy and set parallelization
    import os

    # initialize RNG
    import numpy as np
    rng = np.random.default_rng(0)

    # set output directory
    from pathlib import Path
    curdir = Path()  # returns current directory
    outdir = Path("out/tutorial_4")
    os.makedirs(outdir, exist_ok=True)

    # create network in shape of a circle
    from disstans import Network, Station
    net_name = "CircleVolcano"
    num_stations = 8
    station_names = [f"S{i:1d}" for i in range(1, num_stations + 1)]
    angles = np.linspace(0, 2 * np.pi, num=num_stations, endpoint=False)
    lons, lats = np.cos(angles) / 10, np.sin(angles) / 10
    net = Network(name=net_name)
    for (istat, stat_name), lon, lat in zip(enumerate(station_names), lons, lats):
        net[stat_name] = Station(name=stat_name, location=[lat, lon, 0.0])

    # create timevector
    import pandas as pd
    t_start_str = "2000-01-01"
    t_end_str = "2001-01-01"
    timevector = pd.date_range(start=t_start_str, end=t_end_str, freq="1D")

    # define a function that takes the station's angle and returns
    # the transient, volcanic motion (outwards radially from center)
    # as well as a linear, secular motion,
    # together with the correlated noise in the direction of the displacement
    def generate_model_and_noise(angle, rng):
        # get dimensions
        n1 = int(np.floor(timevector.size / 2))
        n2 = int(np.ceil(timevector.size / 2))
        # rotation matrix
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # secular motion
        p_sec = np.array([[0, 0], [6, 0]])
        # transient motion, only in first half
        p_vol = np.array([[8, 0]])
        p_vol_rot = p_vol @ R.T
        # correlated error in direction of angle in first half
        cov_cor = np.diag([10, 3])
        cov_rot = R @ cov_cor @ R.T
        noise_cor = np.empty((timevector.size, 2))
        noise_cor[:n1, :] = rng.multivariate_normal(mean=(0, 0), cov=cov_rot, size=n1)
        # not correlated in second half
        cov_uncor = np.diag([3, 3])
        noise_cor[-n2:, :] = rng.multivariate_normal(mean=(0, 0), cov=cov_uncor, size=n2)
        # make uncertainty columns
        var_en = np.empty((timevector.size, 2))
        cov_en = np.empty((timevector.size, 1))
        var_en[:n1, :] = np.diag(cov_rot).reshape(1, 2) * np.ones((n1, 2))
        var_en[-n2:, :] = np.diag(cov_uncor).reshape(1, 2) * np.ones((n2, 2))
        cov_en[:n1, :] = cov_rot[0, 1] * np.ones((n1, 1))
        cov_en[-n2:, :] = cov_uncor[0, 1] * np.ones((n2, 1))
        return p_sec, p_vol_rot, noise_cor, var_en, cov_en

    # now add them to all stations
    from copy import deepcopy
    from disstans import Timeseries
    from disstans.models import Arctangent, Polynomial, SplineSet
    mdl_coll, mdl_coll_synth = {}, {}  # containers for the model objects
    synth_coll = {}  # dictionary of synthetic data & noise for each stations
    for station, angle in zip(net, angles):
        # think of some model parameters
        gen_data = {}
        p_sec, p_vol, gen_data["noise"], var_en, cov_en = \
            generate_model_and_noise(angle, rng)
        # create model objects
        mdl_sec = Polynomial(order=1, time_unit="Y", t_reference=t_start_str)
        # Arctangent is for the truth, SplineSet are for how we will estimate them
        mdl_vol = Arctangent(tau=20, t_reference="2000-03-01")
        mdl_trans = SplineSet(degree=2,
                              t_center_start=t_start_str,
                              t_center_end=t_end_str,
                              list_num_knots=[7, 13])
        # collect the models in the dictionary
        mdl_coll_synth[station.name] = {"Secular": mdl_sec}
        mdl_coll[station.name] = deepcopy(mdl_coll_synth[station.name])
        mdl_coll_synth[station.name].update({"Volcano": mdl_vol})
        mdl_coll[station.name].update({"Transient": mdl_trans})
        # only the model objects that will not be associated with the station
        # get their model parameters input
        mdl_sec.read_parameters(p_sec)
        mdl_vol.read_parameters(p_vol)
        # now, evaluate the models...
        # gen_data["truth"] = mdl_sec.evaluate(timevector)["fit"]
        gen_data["truth"] = (mdl_sec.evaluate(timevector)["fit"] +
                             mdl_vol.evaluate(timevector)["fit"])
        gen_data["data"] = gen_data["truth"] + gen_data["noise"]
        synth_coll[station.name] = gen_data
        # ... and assign them to the station as timeseries objects
        station["Truth"] = \
            Timeseries.from_array(timevector=timevector,
                                  data=gen_data["truth"],
                                  src="synthetic",
                                  data_unit="mm",
                                  data_cols=["E", "N"])
        station["Displacement"] = \
            Timeseries.from_array(timevector=timevector,
                                  data=gen_data["data"],
                                  var=var_en,
                                  cov=cov_en,
                                  src="synthetic",
                                  data_unit="mm",
                                  data_cols=["E", "N"])
        # finally, we give the station the models to fit
        station.add_local_model_dict(ts_description="Displacement",
                                     model_dict=mdl_coll[station.name])

    # save a map and timeseries plot
    net.gui(station="S1", save=True, save_map=True)

    # move the map and timeseries to the output folder
    os.rename(next(curdir.glob('map_*.png')),
              (outdir / "tutorial_4a_map.png"))
    os.rename(next(curdir.glob('ts_*.png')),
              (outdir / "tutorial_4b_ts_S1.png"))

    # define a reweighting function
    from disstans.solvers import LogarithmicReweighting
    rw_func = LogarithmicReweighting(1e-8, scale=10)

    # solve without using the data variance
    # using non-spatial L0 will make the fits worse, since there's more amibguity as
    # to where the transients should be fitted - probably optimizable by choosing a
    # good penalty etc., but much easier with spatial L0
    print("\nSpatial L0, only data\n")
    stats = net.spatialfit("Displacement",
                           penalty=10,
                           spatial_l0_models=["Transient"],
                           spatial_reweight_iters=20,
                           reweight_func=rw_func,
                           use_data_variance=False,
                           use_data_covariance=False,
                           formal_covariance=True,
                           verbose=True)
    net.evaluate("Displacement", output_description="Fit_onlydata")

    # save estimated velocity components
    vel_en_est = {}
    vel_en_est["onlydata"] = \
        np.stack([s.models["Displacement"]["Secular"].parameters[1, :] for s in net])

    # save estimated velocity covariances
    cov_en_est = {}
    cov_en_est["onlydata"] = \
        np.stack([s.models["Displacement"]["Secular"].cov[[2, 3, 2], [2, 3, 3]] for s in net])

    # solve using the data variance
    print("\nSpatial L0, with variance\n")
    stats = net.spatialfit("Displacement",
                           penalty=10,
                           spatial_l0_models=["Transient"],
                           spatial_reweight_iters=20,
                           reweight_func=rw_func,
                           use_data_variance=True,
                           use_data_covariance=False,
                           formal_covariance=True,
                           verbose=True)
    net.evaluate("Displacement", output_description="Fit_withvar")

    # save estimated velocity components
    vel_en_est["withvar"] = \
        np.stack([s.models["Displacement"]["Secular"].parameters[1, :] for s in net])

    # save estimated velocity covariances
    cov_en_est["withvar"] = \
        np.stack([s.models["Displacement"]["Secular"].cov[[2, 3, 2], [2, 3, 3]] for s in net])

    # solve with the data variance and covariance
    print("\nSpatial L0, with variance and covariance\n")
    stats = net.spatialfit("Displacement",
                           penalty=10,
                           spatial_l0_models=["Transient"],
                           spatial_reweight_iters=20,
                           reweight_func=rw_func,
                           use_data_variance=True,
                           use_data_covariance=True,
                           formal_covariance=True,
                           verbose=True)
    net.evaluate("Displacement", output_description="Fit_withvarcov")

    # save estimated velocity components
    vel_en_est["withvarcov"] = \
        np.stack([s.models["Displacement"]["Secular"].parameters[1, :] for s in net])

    # save estimated velocity covariances
    cov_en_est["withvarcov"] = \
        np.stack([s.models["Displacement"]["Secular"].cov[[2, 3, 2], [2, 3, 3]] for s in net])

    # plot a timeseries and scalogram
    net.gui(station="S2", save=True, timeseries=["Displacement"],
            scalogram_kw_args={"ts": "Displacement", "model": "Transient", "cmaprange": 3})
    # move to output folder and rename
    os.rename(next(curdir.glob('ts_*.png')),
              (outdir / "tutorial_4c_ts_S2.png"))
    os.rename(next(curdir.glob('scalo_*.png')),
              (outdir / "tutorial_4c_scalo_S2.png"))

    # collect true velocity values
    vel_en_true = np.stack([mdl_coll_synth[s]["Secular"].parameters[1, :]
                            for s in station_names])
    norm_true = np.sqrt(np.sum(vel_en_true**2, axis=1))
    # calculate and print statistics for each station
    err_stats = {}
    for title, case in \
        zip(["Data", "Data + Variance", "Data + Variance + Covariance"],
            ["onlydata", "withvar", "withvarcov"]):
        # error statistics
        print(f"\nError Statistics for {title}:")
        # get amplitude errors
        norm_est = np.sqrt(np.sum(vel_en_est[case]**2, axis=1))
        err_amp = norm_est - norm_true
        # get absolute angle error by calculating the angle between the two vectors
        dotprodnorm = (np.sum(vel_en_est[case] * vel_en_true, axis=1) /
                       (norm_est * norm_true))
        err_angle = np.rad2deg(np.arccos(dotprodnorm))
        # make error dataframe and print
        err_df = pd.DataFrame(index=station_names,
                              data={"Amplitude": err_amp, "Angle": err_angle})
        print(err_df)
        # print rms of both
        print(f"RMS Amplitude: {np.sqrt(np.mean(err_amp**2)):.11g}")
        print(f"RMS Angle: {np.sqrt(np.mean(err_angle**2)):.11g}")
        # print covariances
        print("Secular (Co)Variances")
        print(pd.DataFrame(index=station_names,
                           data=cov_en_est[case],
                           columns=["E-E", "N-N", "E-N"]))

    # save current covariance
    spat_cov = {sta_name: net[sta_name].models["Displacement"].cov
                for sta_name in station_names}

    # get some imports for the plotting to happen
    import matplotlib.pyplot as plt
    from cmcrameri import cm as scm
    from disstans.tools import cov2corr

    # define a plotting function for repeatability
    def corr_plot(cov, title, fname_corr):
        plt.imshow(cov2corr(cov), cmap=scm.roma, vmin=-1, vmax=1)
        plt.colorbar(label="Correlation Coefficient $[-]$")
        plt.title("Correlation: " + title)
        plt.xlabel("Parameter index")
        plt.ylabel("Parameter index")
        plt.savefig(fname_corr)
        plt.close()

    # plot the covariance & correlation matrix of S2 to show the degeneracy
    corr_plot(spat_cov["S2"], "Spatial L0 at S2",
              outdir / "tutorial_4d_corr_S2.png")

    # show that the subsetted estimation with linear regression would yield
    # a very different covariance matrix
    # but this would also be the one if we repeated the spatial L0 many times,
    # since it would always be the same splines
    print("\nLocal linear regression with frozen dictionary\n")
    net.freeze("Displacement", model_list=["Transient"], zero_threshold=1e-6)
    net.fitevalres("Displacement", solver="linear_regression",
                   use_data_variance=True, use_data_covariance=True,
                   formal_covariance=True)
    net.unfreeze("Displacement")
    sub_param = {sta_name: net[sta_name].models["Displacement"].par.ravel()
                 for sta_name in station_names}
    sub_cov = {sta_name: net[sta_name].models["Displacement"].cov
               for sta_name in station_names}
    # plot
    corr_plot(sub_cov["S2"], "Frozen local L0 at S2",
              outdir / "tutorial_4e_corr_S2.png")

    # repeat the full inversion many times with new noise realizations
    # to estimate a covariance matrix to compare the estimated ones to
    # need to use local L0 to make noise matter
    num_repeat = 100
    stacked_params_tru = \
        {sta_name: np.empty((num_repeat,
                             net[sta_name].models["Displacement"].num_parameters * 2))
         for sta_name in station_names}

    # loop
    print("\nLocal lasso regression with different noise, truth-based\n")
    for i in range(num_repeat):
        # change noise
        for station, angle in zip(net, angles):
            station["Displacement"].data = \
                station["Truth"].data + generate_model_and_noise(angle, rng)[2]
        # solve, same reweight_func, same penalty = easy
        net.fit("Displacement", solver="lasso_regression", penalty=10,
                reweight_max_iters=5, reweight_func=rw_func,
                use_data_variance=True, use_data_covariance=True,
                formal_covariance=True, progress_desc=f"Fit {i}")
        # save
        for sta_name in station_names:
            stacked_params_tru[sta_name][i, :] = net[sta_name].models["Displacement"].par.ravel()

    # calculate empirical covariance
    emp_cov_tru = {sta_name: np.cov(stacked_params_tru[sta_name], rowvar=False)
                   for sta_name in station_names}

    # plot the covariance matrix of S2 to show a cleaner one
    corr_plot(emp_cov_tru["S2"], "Truth-based Empirical Local L0 at S2",
              outdir / "tutorial_4f_corr_S2.png")

    # repeat the loop, but this time resampling data from its assumed covariance
    num_repeat = 100
    stacked_params_dat = \
        {sta_name: np.empty((num_repeat,
                             net[sta_name].models["Displacement"].num_parameters * 2))
         for sta_name in station_names}
    orig_data = {sta_name: net[sta_name]["Displacement"].data.values.copy()
                 for sta_name in station_names}

    # loop
    print("\nLocal lasso regression with different noise, data-based\n")
    for i in range(num_repeat):
        # change noise
        for station, angle in zip(net, angles):
            station["Displacement"].data = \
                orig_data[station.name] + generate_model_and_noise(angle, rng)[2]
        # solve, same reweight_func, same penalty = easy
        net.fit("Displacement", solver="lasso_regression", penalty=10,
                reweight_max_iters=5, reweight_func=rw_func,
                use_data_variance=True, use_data_covariance=True,
                formal_covariance=True, progress_desc=f"Fit {i}")
        # save
        for sta_name in station_names:
            stacked_params_dat[sta_name][i, :] = net[sta_name].models["Displacement"].par.ravel()

    # calculate empirical covariance
    emp_cov_dat = {sta_name: np.cov(stacked_params_dat[sta_name], rowvar=False)
                   for sta_name in station_names}

    # plot the covariance matrix of S2 to show a cleaner one
    corr_plot(emp_cov_dat["S2"], "Data-based Empirical Local L0 at S2",
              outdir / "tutorial_4g_corr_S2.png")

    # # show
    # net.gui(timeseries=["Displacement"],
    #         scalogram_kw_args={"ts": "Displacement", "model": "Transient", "cmaprange": 3})
