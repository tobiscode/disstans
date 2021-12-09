"""
This is a tutorial creating a synthetic network to test
the temporal and spatial sparsity enforcing methods.
"""

if __name__ == "__main__":

    # don't let OpenMP mess with NumPy
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    import disstans
    disstans.defaults["general"]["num_threads"] = 16

    # decide output quality, looks & folder
    from pathlib import Path
    from matplotlib import rcParams
    outdir = Path("out/tutorial_3")
    rcParams['font.sans-serif'] = ["NewComputerModernSans10"]
    rcParams['font.size'] = "14"
    fmt = "png"
    include_resids = True
    os.makedirs(outdir, exist_ok=True)

    # initialize RNG
    import numpy as np
    rng = np.random.default_rng(0)

    # create network
    # Null Island https://en.wikipedia.org/wiki/Null_Island
    # names from https://aiweirdness.com/post/162396324452/neural-networks-kittens
    from disstans import Network, Station
    net_name = "NullIsland"
    station_names = ["Jeckle", "Cylon", "Marper", "Timble",
                     "Macnaw", "Colzyy", "Mrror", "Mankith",
                     "Lingo", "Marvish", "Corko", "Kogon",
                     "Malool", "Aarla", "Tygrar", "Jozga"]
    nlon, nlat = 16, 1
    num_stations = nlon * nlat
    lons, lats = np.meshgrid(np.linspace(0, 1, num=nlon),
                             np.linspace(-0.1, 0.1, num=nlat))
    net = Network(name=net_name)
    for (istat, stat_name), lon, lat in zip(enumerate(station_names),
                                            lons.ravel(), lats.ravel()):
        temp_loc = [lat + rng.normal()*0.02 + int(istat % 2 == 0)*0.1,
                    lon + rng.normal()*0.01, 0]
        net[stat_name] = Station(name=stat_name,
                                 location=temp_loc)

    # quick test
    assert num_stations == net.num_stations == len(station_names)

    # create timevector
    import pandas as pd
    t_start_str = "2000-01-01"
    t_end_str = "2010-01-01"
    timevector = pd.date_range(start=t_start_str, end=t_end_str, freq="1D")

    # create CME
    cme_noise = rng.normal(size=(timevector.size, 2)) * 0.2

    # define noise covariance matrix
    # (parameters come from histogram fitting UNR Long Valley data)
    from scipy.stats import invgamma, laplace
    var_e, var_n, cov_en = 0.354, 0.538, 0.015
    invgamma_e_alpha, invgamma_e_scale = 2.569, 0.274
    invgamma_n_alpha, invgamma_n_scale = 3.054, 0.536
    laplace_en_scale = 0.031
    noise_cov = np.array([[var_e, cov_en], [cov_en, var_n]])

    # for synthetic data network, define a function that takes location
    # and returns common signal parameters and the noise vector
    def generate_parameters_noise(loc, rng):
        lon, lat = loc[1], loc[0]  # noqa: F841
        p_sec = np.array([[0, 0], [1, -1]])
        p_seas = rng.uniform(-0.3, 0.3, size=(2, 2))
        p_sse1 = np.array([[6, -6]])*np.exp(-(3 * lon**2))  # from the west
        p_sse2 = np.array([[4, -4]])*np.exp(-(3 * lon**2))  # from the west
        p_sse3 = np.array([[8, -8]])*np.exp(-(3 * lon**2))  # from the west
        p_eq = np.array([[-3, 3]])
        meas_noise = rng.multivariate_normal(mean=(0, 0), cov=noise_cov,
                                             size=timevector.size)
        noisevec = meas_noise + cme_noise
        estim_var_cov = np.stack([invgamma.rvs(invgamma_e_alpha, loc=var_e,
                                               scale=invgamma_e_scale,
                                               size=timevector.size, random_state=rng),
                                  invgamma.rvs(invgamma_n_alpha, loc=var_n,
                                               scale=invgamma_n_scale,
                                               size=timevector.size, random_state=rng),
                                  laplace.rvs(loc=cov_en, scale=laplace_en_scale,
                                              size=timevector.size, random_state=rng)], axis=1)
        return p_sec, p_seas, p_eq, p_sse1, p_sse2, p_sse3, noisevec, estim_var_cov

    # now add them to all stations
    from copy import deepcopy
    from disstans import Timeseries
    from disstans.models import HyperbolicTangent, Polynomial, Sinusoid, Step, \
        SplineSet, Logarithmic
    from disstans.tools import create_powerlaw_noise
    mdl_coll, mdl_coll_synth = {}, {}  # containers for the model objects
    synth_coll = {}  # dictionary of synthetic data & noise for each stations
    for station in net:
        # think of some model parameters
        p_sec, p_seas, p_eq, p_sse1, p_sse2, p_sse3, noisevec, estim_var_cov = \
            generate_parameters_noise(station.location, rng)
        # create model objects
        mdl_sec = Polynomial(order=1, time_unit="Y", t_reference=t_start_str)
        mdl_seas = Sinusoid(period=1, time_unit="Y", t_reference=t_start_str)
        mdl_eq = Step(["2002-07-01"])
        mdl_post = Logarithmic(tau=20, t_reference="2002-07-01")
        # HyperbolicTangent (no long tails!) is for the truth, SplineSet are for how
        # we will estimate them.
        # We could align the HyperbolicTangents with the spline center times but that would
        # never happen in real life so it would just unrealistically embellish our results
        mdl_sse1 = HyperbolicTangent(tau=50, t_reference="2001-07-01")
        mdl_sse2 = HyperbolicTangent(tau=50, t_reference="2003-07-01")
        mdl_sse3 = HyperbolicTangent(tau=300, t_reference="2007-01-01")
        mdl_trans = SplineSet(degree=2,
                              t_center_start=t_start_str,
                              t_center_end=t_end_str,
                              list_num_knots=[int(1+2**n) for n in range(3, 8)])
        # collect the models in the dictionary
        mdl_coll_synth[station.name] = {"Secular": mdl_sec,
                                        "Seasonal": mdl_seas,
                                        "Earthquake": mdl_eq,
                                        "Postseismic": mdl_post}
        mdl_coll[station.name] = deepcopy(mdl_coll_synth[station.name])
        mdl_coll_synth[station.name].update({"SSE1": mdl_sse1,
                                             "SSE2": mdl_sse2,
                                             "SSE3": mdl_sse3})
        mdl_coll[station.name].update({"Transient": mdl_trans})
        # only the model objects that will not be associated with the station
        # get their model parameters read in
        mdl_sec.read_parameters(p_sec)
        mdl_seas.read_parameters(p_seas)
        mdl_eq.read_parameters(p_eq)
        mdl_post.read_parameters(p_eq/5)
        mdl_sse1.read_parameters(p_sse1)
        mdl_sse2.read_parameters(p_sse2)
        mdl_sse3.read_parameters(p_sse3)
        # now, evaluate the models
        # noise will be white + colored
        gen_data = \
            {"seas+sec+eq": (mdl_sec.evaluate(timevector)["fit"] +
                             mdl_seas.evaluate(timevector)["fit"] +
                             mdl_eq.evaluate(timevector)["fit"] +
                             mdl_post.evaluate(timevector)["fit"]),
             "trans": (mdl_sse1.evaluate(timevector)["fit"] +
                       mdl_sse2.evaluate(timevector)["fit"] +
                       mdl_sse3.evaluate(timevector)["fit"]),
             "noise": noisevec}
        # for one station, we'll add a colored noise process such that the resulting
        # noise variance is the same as before
        # but: only in the second half, where there are no strong, short-term signals
        if station.name == "Cylon":
            gen_data["noise"][timevector.size//2:, :] = \
                (gen_data["noise"][timevector.size//2:, :] +
                 create_powerlaw_noise(size=(timevector.size // 2, 2),
                                       exponent=1, seed=rng
                                       ) * np.sqrt(np.array([[var_e, var_n]]))
                 ) / np.sqrt(2)
        # for one special station, we add the maintenance step
        # repeating all steps above
        if station.name == "Corko":
            # time and amplitude
            p_maint = np.array([[-2, 0]])
            mdl_maint = Step(["2005-01-01"])
            mdl_maint.read_parameters(p_maint)
            # add to station and synthetic data
            mdl_coll_synth[station.name].update({"Maintenance": mdl_maint})
            gen_data["seas+sec+eq"] += mdl_maint.evaluate(timevector)["fit"]
        # now we sum the components up...
        gen_data["truth"] = gen_data["seas+sec+eq"] + gen_data["trans"]
        gen_data["data"] = gen_data["truth"] + gen_data["noise"]
        synth_coll[station.name] = gen_data
        # ... and assign them to the station as timeseries objects
        station["Truth"] = \
            Timeseries.from_array(timevector=timevector,
                                  data=gen_data["truth"],
                                  src="synthetic",
                                  data_unit="mm",
                                  data_cols=["E", "N"])
        station["Raw"] = \
            Timeseries.from_array(timevector=timevector,
                                  data=gen_data["data"],
                                  var=estim_var_cov[:, :2],
                                  cov=estim_var_cov[:, 2],
                                  src="synthetic",
                                  data_unit="mm",
                                  data_cols=["E", "N"])

    # print the summary of a station
    print(net["Jeckle"])

    # save a map and timeseries plot
    net.gui(station="Jeckle", timeseries=["Raw"], save=True, save_map=True,
            save_kw_args={"format": fmt, "dpi": 300})

    # make an output folder and move the map and timeseries there
    curdir = Path()  # returns current directory
    os.rename(next(curdir.glob(f"map_*.{fmt}")),
              (outdir / f"tutorial_3a_map.{fmt}"))
    os.rename(next(curdir.glob(f"ts_*.{fmt}")),
              (outdir / f"tutorial_3a_ts_Jeckle.{fmt}"))

    # estimate and remove CME
    # low-pass filter using a median function, then clean the timeseries
    # (using the settings in the config file)
    net.call_func_ts_return("median", ts_in="Raw", ts_out="Filtered", kernel_size=7)
    # get the residual for each station
    net.math("Residual", "Raw", "-", "Filtered")
    # estimate the common mode, either with a visualization of the result or not
    # (same underlying function)
    net.graphical_cme(ts_in="Residual", ts_out="CME", method="ica",
                      save=True, save_kw_args={"format": fmt, "dpi": 300})
    # net.call_netwide_func("decompose", ts_in="Residual", ts_out="CME", method="ica")
    # now remove the common mode, call it the "Displacement" timeseries,
    for station in net:
        # calculate the clean timeseries
        station.add_timeseries("Displacement", station["Raw"] - station["CME"],
                               override_data_cols=station["Raw"].data_cols)
        # copy over the uncertainties
        station["Displacement"].add_uncertainties(timeseries=station["Raw"])
        # give the station the models to fit
        station.add_local_model_dict(ts_description="Displacement",
                                     model_dict=mdl_coll[station.name])
    # remove unnecessary intermediate results
    net.remove_timeseries("Filtered", "CME", "Residual")
    # move plots
    os.rename(f"cme_spatial.{fmt}", outdir / f"tutorial_3b_cme_spatial.{fmt}")
    os.rename(f"cme_temporal.{fmt}", outdir / f"tutorial_3b_cme_temporal.{fmt}")

    # print the summary of a station again
    print(net["Jeckle"])

    # fit everything with L1 regularization and create residuals
    print()
    net.fitevalres(ts_description="Displacement", solver="lasso_regression",
                   penalty=1, output_description="Fit_L1", residual_description="Res_L1")
    for stat in net:
        stat["Trans_L1"] = stat.fits["Displacement"]["Transient"].copy(only_data=True)
    net.math("Err_L1", "Fit_L1", "-", "Truth")

    # quick looks
    figure_stations = ["Jeckle", "Cylon", "Marvish", "Mankith", "Corko", "Tygrar", "Jozga"]
    for s in figure_stations:
        net.gui(station=s, save="base",
                timeseries=["Displacement", "Res_L1"]
                if include_resids else ["Displacement"],
                save_kw_args={"format": fmt, "dpi": 300},
                scalogram_kw_args={"ts": "Displacement", "model": "Transient",
                                   "cmaprange": 2})

    # get number of (unique) non-zero parameters
    ZERO = 1e-4  # this is from the default in Network.spatialfit
    num_total = sum([s.models["Displacement"]["Transient"].parameters.size for s in net])
    num_uniques_base = \
        np.sum(np.any(np.stack([np.abs(s.models["Displacement"]["Transient"].parameters)
                                > ZERO for s in net]), axis=0), axis=0)
    num_nonzero_base = sum([(s.models["Displacement"]["Transient"].parameters.ravel() > ZERO
                             ).sum() for s in net])
    print(f"Number of reweighted non-zero parameters: {num_nonzero_base}/{num_total}")
    print("Number of unique reweighted non-zero parameters per component: "
          + str(num_uniques_base.tolist()))

    # get spatial correlation matrix of the transient model for later (just first component)
    cor_base = np.corrcoef(np.stack([s.fits["Displacement"]["Transient"].data.values[:, 1]
                                     for s in net]))

    # repeat everything with reweighted L1 regularization
    print()
    net.fitevalres(ts_description="Displacement", solver="lasso_regression",
                   penalty=1, reweight_max_iters=5,
                   output_description="Fit_L1R5", residual_description="Res_L1R5")
    for stat in net:
        stat["Trans_L1R5"] = stat.fits["Displacement"]["Transient"].copy(only_data=True)
    net.math("Err_L1R5", "Fit_L1R5", "-", "Truth")

    # get spatial correlation matrix for later
    cor_localiters = np.corrcoef(np.stack([s.fits["Displacement"]["Transient"].data.values[:, 1]
                                           for s in net]))

    # quick looks
    for s in figure_stations:
        net.gui(station=s, save="local", save_kw_args={"format": fmt, "dpi": 300},
                timeseries=["Displacement", "Res_L1R5"]
                if include_resids else ["Displacement"],
                scalogram_kw_args={"ts": "Displacement", "model": "Transient",
                                   "cmaprange": 2})

    # get number of (unique) non-zero parameters
    num_uniques_local = \
        np.sum(np.any(np.stack([np.abs(s.models["Displacement"]["Transient"].parameters)
                                > ZERO for s in net]), axis=0), axis=0)
    num_nonzero_local = sum([(s.models["Displacement"]["Transient"].parameters.ravel() > ZERO
                              ).sum() for s in net])
    print(f"Number of reweighted non-zero parameters: {num_nonzero_local}/{num_total}")
    print("Number of unique reweighted non-zero parameters per component: "
          + str(num_uniques_local.tolist()))

    # define reweighting function for spatial L0
    rw_func = disstans.solvers.InverseReweighting(eps=1e-4, scale=0.1)

    # lasso fit, 1 spatial reweighting, 1 local reweightings
    print()
    stats = net.spatialfit("Displacement",
                           penalty=1,
                           spatial_reweight_models=["Transient"],
                           spatial_reweight_iters=1,
                           local_reweight_func=rw_func,
                           formal_covariance=True,
                           verbose=True)
    net.evaluate("Displacement", output_description="Fit_L1R1S1")
    for stat in net:
        stat["Trans_L1R1S1"] = stat.fits["Displacement"]["Transient"].copy(only_data=True)

    # residual
    net.math("Res_L1R1S1", "Displacement", "-", "Fit_L1R1S1")
    net.math("Err_L1R1S1", "Fit_L1R1S1", "-", "Truth")

    # get spatial correlation matrix for later
    cor_spatialiters1 = \
        np.corrcoef(np.stack([s.fits["Displacement"]["Transient"].data.values[:, 1]
                              for s in net]))

    # plot
    for s in figure_stations:
        net.gui(station=s, save="spatial1", save_kw_args={"format": fmt, "dpi": 300},
                timeseries=["Displacement", "Res_L1R1S1"]
                if include_resids else ["Displacement"],
                scalogram_kw_args={"ts": "Displacement", "model": "Transient",
                                   "cmaprange": 2})

    # repeat with 20 spatial iterations
    print()
    stats = net.spatialfit("Displacement",
                           penalty=1,
                           spatial_reweight_models=["Transient"],
                           spatial_reweight_iters=20,
                           local_reweight_func=rw_func,
                           formal_covariance=True,
                           verbose=True)
    net.evaluate("Displacement", output_description="Fit_L1R1S20")
    for stat in net:
        stat["Trans_L1R1S20"] = stat.fits["Displacement"]["Transient"].copy(only_data=True)

    # residual
    net.math("Res_L1R1S20", "Displacement", "-", "Fit_L1R1S20")
    net.math("Err_L1R1S20", "Fit_L1R1S20", "-", "Truth")

    # get spatial correlation matrix for later
    cor_spatialiters20 = \
        np.corrcoef(np.stack([s.fits["Displacement"]["Transient"].data.values[:, 1]
                              for s in net]))

    # plot
    for s in figure_stations:
        net.gui(station=s, save="spatial20",
                save_kw_args={"format": fmt, "dpi": 300},
                timeseries=["Displacement", "Res_L1R1S20"]
                if include_resids else ["Displacement"],
                scalogram_kw_args={"ts": "Displacement", "model": "Transient",
                                   "cmaprange": 2})

    # for each solution case, and each station, move the files to the output folder and rename
    from itertools import product
    for case, station in product(["base", "local", "spatial1", "spatial20"],
                                 figure_stations):
        for imgfile in curdir.glob(f"*_{station}_{case}_*.{fmt}"):  # ts_, scalo_ and map_
            prefix = imgfile.name.split("_")[0]
            newfile = outdir / f"tutorial_3c_{prefix}_{station}_{case}.{fmt}"
            os.rename(imgfile, newfile)

    # now add the maintenance step
    new_maint_mdl = {"Maintenance": Step(["2005-01-01"])}
    mdl_coll["Corko"].update(new_maint_mdl)
    net["Corko"].add_local_model_dict("Displacement", new_maint_mdl)

    # do L1 again
    print()
    net.fitevalres(ts_description="Displacement", solver="lasso_regression",
                   penalty=1, output_description="Fit_L1M", residual_description="Res_L1M")
    for stat in net:
        stat["Trans_L1M"] = stat.fits["Displacement"]["Transient"].copy(only_data=True)
    net.math("Err_L1M", "Fit_L1M", "-", "Truth")
    # quick looks
    for s in figure_stations:
        net.gui(station=s, save="baseM", save_kw_args={"format": fmt, "dpi": 300},
                timeseries=["Displacement", "Res_L1M"]
                if include_resids else ["Displacement"],
                scalogram_kw_args={"ts": "Displacement", "model": "Transient",
                                   "cmaprange": 2})
    # get number of (unique) non-zero parameters
    num_uniques_base_M = \
        np.sum(np.any(np.stack([np.abs(s.models["Displacement"]["Transient"].parameters)
                                > ZERO for s in net]), axis=0), axis=0)
    num_nonzero_base_M = sum([(s.models["Displacement"]["Transient"].parameters.ravel() > ZERO
                               ).sum() for s in net])
    print(f"Number of reweighted non-zero parameters: {num_nonzero_base_M}/{num_total}")
    print("Number of unique reweighted non-zero parameters per component: "
          + str(num_uniques_base_M.tolist()))
    # get spatial correlation matrix of the transient model for later (just first component)
    cor_base_M = np.corrcoef(np.stack([s.fits["Displacement"]["Transient"].data.values[:, 1]
                                       for s in net]))

    # do the local-L0 fit again
    print()
    net.fitevalres(ts_description="Displacement", solver="lasso_regression",
                   penalty=1, reweight_max_iters=5,
                   output_description="Fit_L1R5M", residual_description="Res_L1R5M")
    for stat in net:
        stat["Trans_L1R5M"] = stat.fits["Displacement"]["Transient"].copy(only_data=True)
    net.math("Err_L1R5M", "Fit_L1R5M", "-", "Truth")
    # get spatial correlation matrix for later
    cor_localiters_M = np.corrcoef(
        np.stack([s.fits["Displacement"]["Transient"].data.values[:, 1] for s in net]))
    # quick looks
    for s in figure_stations:
        net.gui(station=s, save="localM", save_kw_args={"format": fmt, "dpi": 300},
                timeseries=["Displacement", "Res_L1R5M"]
                if include_resids else ["Displacement"],
                scalogram_kw_args={"ts": "Displacement", "model": "Transient",
                                   "cmaprange": 2})
    # get number of (unique) non-zero parameters
    num_uniques_local_M = \
        np.sum(np.any(np.stack([np.abs(s.models["Displacement"]["Transient"].parameters)
                                > ZERO for s in net]), axis=0), axis=0)
    num_nonzero_local_M = sum([(s.models["Displacement"]["Transient"].parameters.ravel() > ZERO
                                ).sum() for s in net])
    print(f"Number of reweighted non-zero parameters: {num_nonzero_local_M}/{num_total}")
    print("Number of unique reweighted non-zero parameters per component: "
          + str(num_uniques_local_M.tolist()))

    # and now for 20 spatial iterations
    print()
    stats = net.spatialfit("Displacement",
                           penalty=1,
                           spatial_reweight_models=["Transient"],
                           spatial_reweight_iters=20,
                           local_reweight_func=rw_func,
                           formal_covariance=True,
                           verbose=True)
    net.evaluate("Displacement", output_description="Fit_L1R1S20M")
    for stat in net:
        stat["Trans_L1R1S20M"] = stat.fits["Displacement"]["Transient"].copy(only_data=True)
    # residual
    net.math("Res_L1R1S20M", "Displacement", "-", "Fit_L1R1S20M")
    net.math("Err_L1R1S20M", "Fit_L1R1S20M", "-", "Truth")
    # get spatial correlation matrix for later
    cor_spatialiters20_M = np.corrcoef(
        np.stack([s.fits["Displacement"]["Transient"].data.values[:, 1] for s in net]))
    # plot again
    for s in figure_stations:
        if s == "Corko":
            net.gui(station=s, save="spatial20M", save_map=True,
                    save_kw_args={"format": fmt, "dpi": 300},
                    timeseries=["Displacement", "Res_L1R1S20M"]
                    if include_resids else ["Displacement"],
                    scalogram_kw_args={"ts": "Displacement", "model": "Transient",
                                       "cmaprange": 2},
                    rms_on_map={"ts": "Res_L1R1S20M", "comps": [0], "c_max": 1})
        else:
            net.gui(station=s, save="spatial20M",
                    save_kw_args={"format": fmt, "dpi": 300},
                    timeseries=["Displacement", "Res_L1R1S20M"]
                    if include_resids else ["Displacement"],
                    scalogram_kw_args={"ts": "Displacement", "model": "Transient",
                                       "cmaprange": 2})

    # for each new solution case, and each station, move & rename again
    for case, station in product(["baseM", "localM", "spatial20M"], figure_stations):
        for imgfile in curdir.glob(f"*_{station}_{case}_*.{fmt}"):  # ts_, scalo_ and map_
            prefix = imgfile.name.split("_")[0]
            newfile = outdir / f"tutorial_3c_{prefix}_{station}_{case}.{fmt}"
            os.rename(imgfile, newfile)

    # calculate and print residual statistics for each residual timeseries,
    # together with the median spatial correlation and its visualization
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from disstans.processing import StepDetector
    stepdet = StepDetector(kernel_size=51)
    for title, case, res_ts, err_ts, cormat in \
        zip(["No reweighting (no M)", "5 Local Reweightings (no M)",
             "1 Local, 1 Spatial Reweighting (no M)", "1 Local, 20 Spatial Reweighting (no M)",
             "No reweighting", "5 Local Reweightings", "1 Local, 20 Spatial Reweighting"],
            ["base", "local", "spatial1", "spatial20", "baseM", "localM", "spatial20M"],
            ["Res_L1", "Res_L1R5", "Res_L1R1S1", "Res_L1R1S20",
             "Res_L1M", "Res_L1R5M", "Res_L1R1S20M"],
            ["Err_L1", "Err_L1R5", "Err_L1R1S1", "Err_L1R1S20",
             "Err_L1M", "Err_L1R5M", "Err_L1R1S20M"],
            [cor_base, cor_localiters, cor_spatialiters1, cor_spatialiters20,
             cor_base_M, cor_localiters_M, cor_spatialiters20_M]):
        print(f"\nStatistics for {title}:")
        # residuals statistics
        print(f"\nResiduals for {res_ts}:")
        stat = net.analyze_residuals(res_ts, mean=True, rms=True)
        print(stat)
        print(stat.mean())
        # error statistics
        print(f"\nErrors for {err_ts}:")
        stat = net.analyze_residuals(err_ts, mean=True, rms=True)
        print(stat)
        print(stat.mean())
        # median spatial correlation of transient timeseries
        medcor = np.ma.median(np.ma.masked_equal(np.triu(cormat, 1), 0))
        print(f"\nMedian spatial correlation = {medcor}\n")
        # run the stepdetector on the residuals
        print(stepdet.search_network(net, res_ts)[0])
        # spatial correlation visualization
        plt.figure()
        plt.title(title)
        plt.imshow(cormat, vmin=-1, vmax=1, interpolation="none")
        plt.yticks(ticks=range(num_stations),
                   labels=net.station_names)
        plt.xticks([])
        plt.savefig(outdir / f"tutorial_3f_corr_{case}.{fmt}", dpi=300)
        plt.close()
        # error visualization
        for s in figure_stations:
            stat = net[s]
            # plot fit and residual
            fig, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].set_title(title)
            ax[0].plot(stat[res_ts].data.iloc[:, 0], c='0.3',
                       ls='none', marker='.', markersize=0.5)
            ax[0].plot(stat[res_ts].time, synth_coll[s]["noise"][:, 0], c='C1',
                       ls='none', marker='.', markersize=0.5)
            ax[0].plot(stat[err_ts].data.iloc[:, 0], c="C0")
            ax[0].set_ylim(-3, 3)
            ax[0].set_ylabel("East [mm]")
            ax[1].plot(stat[res_ts].data.iloc[:, 1], c='0.3',
                       ls='none', marker='.', markersize=0.5)
            ax[1].plot(stat[res_ts].time, synth_coll[s]["noise"][:, 1], c='C1',
                       ls='none', marker='.', markersize=0.5)
            ax[1].plot(stat[err_ts].data.iloc[:, 1], c="C0")
            ax[1].set_ylim(-3, 3)
            ax[1].set_ylabel("North [mm]")
            custom_lines = [Line2D([0], [0], c="0.3", marker=".", linestyle='none'),
                            Line2D([0], [0], c="C1", marker=".", linestyle='none'),
                            Line2D([0], [0], c="C0")]
            ax[0].legend(custom_lines, ["Residual", "Noise", "Error"],
                         loc="upper right", ncol=3)
            ax[1].legend(custom_lines, ["Residual", "Noise", "Error"],
                         loc="upper right", ncol=3)
            fig.savefig(outdir / f"tutorial_3d_{s}_{case}.{fmt}")
            plt.close(fig)

    # plot the iteration statistics for the last spatialfit call
    # first figure is for num_total, arr_uniques, list_nonzeros
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(stats["list_nonzeros"], c="k", marker=".")
    ax1.scatter(-0.1, num_nonzero_base_M, s=100, c="k")
    ax1.scatter(-0.1, num_nonzero_local_M, s=60, c="k", marker="D")
    ax1.set_ylim([0, 1000])
    ax1.set_yticks(range(0, 1200, 200))
    ax2.plot(stats["arr_uniques"][:, 0], c="C0", marker=".")
    ax2.plot(stats["arr_uniques"][:, 1], c="C1", marker=".")
    ax2.scatter(23, num_uniques_base_M[0], s=100, c="C0")
    ax2.scatter(23, num_uniques_local_M[0], s=60, c="C0", marker="D")
    ax2.scatter(23, num_uniques_base_M[1], s=100, c="C1")
    ax2.scatter(23, num_uniques_local_M[1], s=60, c="C1", marker="D")
    ax2.set_ylim([0, 250])
    ax2.set_yticks(range(0, 300, 50))
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_xlim([-0.1, 23])
    ax1.set_xticks([0, 1, 5, 10, 15, 20])
    ax1.set_xticklabels(["0", "1", "5", "10", "15", "20"])
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Total number of non-zero parameters")
    ax2.set_ylabel("Unique number of non-zero parameters")
    custom_lines = [Patch(color="k",),
                    Patch(color="C0"),
                    Patch(color="C1"),
                    Line2D([0], [0], c=[1, 1, 1, 0], mfc="0.7", marker="o"),
                    Line2D([0], [0], c=[1, 1, 1, 0], mfc="0.7", marker="D"),
                    Line2D([0], [0], c="0.7", marker=".")]
    ax1.set_title(f"Number of available parameters: {stats['num_total']}")
    ax1.legend(custom_lines, ["Total", "Unique East", "Unique North",
                              "L1", "Local L0", "Spatial L0"], loc=(0.56, 0.53), ncol=1)
    fig.savefig(outdir / f"tutorial_3e_numparams.{fmt}")
    plt.close(fig)
    # second figure is for dict_rms_diff, dict_num_changed
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(1, 21), stats["dict_rms_diff"]["Transient"], c="C0", marker=".")
    ax1.set_yscale("log")
    ax1.set_ylim([1e-4, 10])
    ax2.plot(range(1, 21), stats["dict_num_changed"]["Transient"], c="C1", marker=".")
    ax2.set_yscale("symlog", linthresh=10)
    ax2.set_ylim([0, 1000])
    ax2.set_yticks([0, 2, 4, 6, 8, 10, 100, 1000])
    ax2.set_yticklabels([0, 2, 4, 6, 8, 10, 100, 1000])
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_xlim([-0.1, 23])
    ax1.set_xticks([0, 1, 5, 10, 20])
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("RMS difference of parameters")
    ax2.set_ylabel("Number of changed parameters")
    custom_lines = [Line2D([0], [0], c="C0", marker="."),
                    Line2D([0], [0], c="C1", marker=".")]
    ax1.legend(custom_lines, ["RMS Difference", "Changed Parameters"])
    fig.savefig(outdir / f"tutorial_3e_diffs.{fmt}")
    plt.close(fig)
    # third figure is just the total number of parameters
    # can use horizontal line instead of markers on the side
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(stats["list_nonzeros"], c="k", marker=".")
    ax.axhline(num_nonzero_local_M, ls="--", lw=1, c="0.5", zorder=-1)
    ax.axhline(num_nonzero_base_M, ls=":", lw=1, c="0.5", zorder=-1)
    ax.set_ylim([0, 1000])
    ax.set_yticks(range(0, 1200, 200))
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim([-0.1, 23])
    ax.set_xticks([0, 1, 5, 10, 20])
    ax.set_xticklabels(["0", "1", "5", "10", "20"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Non-zero parameters")
    custom_lines = [Line2D([0], [0], c="0.5", ls=":", lw=1),
                    Line2D([0], [0], c="0.5", ls="--", lw=1),
                    Line2D([0], [0], c="k", marker=".")]
    ax.legend(custom_lines, ["L1", "Local L0", "Spatial L0"], loc=(0.49, 0.55))
    fig.savefig(outdir / f"tutorial_3e_numparams_total.{fmt}")
    plt.close(fig)

    # make correlation plot, showing the sparsity
    net["Jeckle"].models["Displacement"].plot_covariance(
        save_kw_args={"format": fmt, "dpi": 300},
        fname=outdir / "tutorial_3g_Jeckle_corr_sparse",
        use_corr_coef=True)
    # make correlation plot with only non-empty parameters
    net["Jeckle"].models["Displacement"].plot_covariance(
        save_kw_args={"format": fmt, "dpi": 300},
        fname=outdir / "tutorial_3g_Jeckle_corr_dense",
        plot_empty=False, use_corr_coef=True)

    # plot example comparison between truth & fit
    sta = net["Jeckle"]
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(6, 4))
    # top left is overall fit
    axes[0, 0].plot(timevector, synth_coll["Jeckle"]["data"][:, 0], "k.", markersize=3,
                    label="Data")
    axes[0, 0].plot(timevector, synth_coll["Jeckle"]["truth"][:, 0], "C1")
    axes[0, 0].plot(timevector, sta.fits["Displacement"].allfits.data.iloc[:, 0], "C0")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].set_ylabel("Displacement [mm]")
    axes[0, 0].set_title("Total")
    # bottom left is secular + seasonal
    axes[1, 0].plot(timevector,
                    mdl_coll_synth["Jeckle"]["Secular"].evaluate(timevector)["fit"][:, 0] +
                    mdl_coll_synth["Jeckle"]["Seasonal"].evaluate(timevector)["fit"][:, 0], "C1")
    axes[1, 0].plot(timevector,
                    sta.fits["Displacement"]["Seasonal"].data.iloc[:, 0] +
                    sta.fits["Displacement"]["Secular"].data.iloc[:, 0], "C0")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Displacement [mm]")
    axes[1, 0].set_title("Secular + Seasonal")
    # top right is transient
    axes[0, 1].plot(timevector, synth_coll["Jeckle"]["trans"][:, 0], "C1")
    axes[0, 1].plot(timevector, sta.fits["Displacement"]["Transient"].data.iloc[:, 0], "C0")
    axes[0, 1].set_title("Transient")
    # bottom right is earthquake (incl. postseismic)
    axes[1, 1].plot(timevector,
                    mdl_coll_synth["Jeckle"]["Earthquake"].evaluate(timevector)["fit"][:, 0] +
                    mdl_coll_synth["Jeckle"]["Postseismic"].evaluate(timevector)["fit"][:, 0],
                    "C1", label="Truth")
    axes[1, 1].plot(timevector,
                    sta.fits["Displacement"]["Earthquake"].data.iloc[:, 0] +
                    sta.fits["Displacement"]["Postseismic"].data.iloc[:, 0], "C0", label="Fit")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_title("Earthquake")
    fig.savefig(outdir / f"tutorial_3i_fits_Jeckle.{fmt}")
    plt.close(fig)

    # worm plots (animated only for the final cases with the maintenance step)
    for title, case, trans_ts in \
        zip(["No reweighting (no M)", "5 Local Reweightings (no M)",
             "1 Local, 1 Spatial Reweighting (no M)", "1 Local, 20 Spatial Reweighting (no M)",
             "No reweighting", "5 Local Reweightings", "1 Local, 20 Spatial Reweighting"],
            ["base", "local", "spatial1", "spatial20", "baseM", "localM", "spatial20M"],
            ["Trans_L1", "Trans_L1R5", "Trans_L1R1S1", "Trans_L1R1S20",
             "Trans_L1M", "Trans_L1R5M", "Trans_L1R1S20M"]):
        print(f"Wormplot for {title}")
        net.wormplot(ts_description=trans_ts,
                     fname=outdir / f"tutorial_3h_worm_{case}",
                     fname_animation=outdir / f"tutorial_3h_worm_{case}.mp4"
                     if case[-1] == "M" else None,
                     save_kw_args={"format": fmt, "dpi": 300},
                     colorbar_kw_args={"orientation": "horizontal", "shrink": 0.5},
                     scale=1e3, annotate_stations=False,
                     lon_min=-0.1, lon_max=1.1, lat_min=-0.3, lat_max=0.1)

    # make a difference wormplot
    net.math("diff-local-spatial", "Trans_L1R5M", "-", "Trans_L1R1S20M")
    print("Wormplot for Local-Spatial")
    net.wormplot(ts_description="diff-local-spatial",
                 fname=outdir / "tutorial_3h_worm_diff",
                 save_kw_args={"format": fmt, "dpi": 300},
                 colorbar_kw_args={"orientation": "horizontal", "shrink": 0.5},
                 scale=1e3, annotate_stations=False,
                 lon_min=-0.1, lon_max=1.1, lat_min=-0.3, lat_max=0.1)

    # make a truth wormplot
    for station in net:
        station["truth-trans"] = \
            Timeseries.from_array(timevector=timevector,
                                  data=synth_coll[station.name]["trans"],
                                  src="synthetic",
                                  data_unit="mm",
                                  data_cols=["E", "N"])
    print("Wormplot for Truth")
    net.wormplot(ts_description="truth-trans",
                 fname=outdir / "tutorial_3h_worm_truth",
                 save_kw_args={"format": fmt, "dpi": 300},
                 colorbar_kw_args={"orientation": "horizontal", "shrink": 0.5},
                 scale=1e3, annotate_stations=False,
                 lon_min=-0.1, lon_max=1.1, lat_min=-0.3, lat_max=0.1)
