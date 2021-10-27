"""
This script aims to approximately recreate the synthetic example
from Bryan Riel's paper using disstans:

Riel, B., Simons, M., Agram, P., & Zhan, Z. (2014).
Detecting transient signals in geodetic time series using sparse estimation techniques.
Journal of Geophysical Research: Solid Earth, 119(6), 5140–5160.
https://doi.org/10.1002/2014JB011077
"""

# make output folder
import os
outdir = "out/tutorial_2"
os.makedirs(outdir, exist_ok=True)

# create timevector
import pandas as pd  # noqa: E402
t_start_str = "2000-01-01"
t_end_str = "2020-01-01"
timevector = pd.date_range(start=t_start_str, end=t_end_str, freq="1D")

# create a model collection (see Tutorial)
from disstans.models import Arctangent, Polynomial, Sinusoid  # noqa: E402
mdl_secular = Polynomial(order=1, t_reference=t_start_str)
mdl_annual = Sinusoid(period=365.25, t_reference=t_start_str)
mdl_semiannual = Sinusoid(period=365.25/2, t_reference=t_start_str)
mdl_transient_1 = Arctangent(tau=100, t_reference="2002-07-01")
mdl_transient_2 = Arctangent(tau=50, t_reference="2010-01-01")
mdl_transient_3 = Arctangent(tau=300, t_reference="2016-01-01")
mdl_coll_synth = {"Secular": mdl_secular,
                  "Annual": mdl_annual,
                  "Semi-Annual": mdl_semiannual,
                  "Transient_1": mdl_transient_1,
                  "Transient_2": mdl_transient_2,
                  "Transient_3": mdl_transient_3}

# make a deep copy of the models we want to compare later
from copy import deepcopy  # noqa: E402
mdl_coll = deepcopy({"Secular": mdl_secular,
                     "Annual": mdl_annual,
                     "Semi-Annual": mdl_semiannual})

# add parameters
import numpy as np  # noqa: E402
mdl_secular.read_parameters(np.array([-20, 200/(20*365.25)]))
mdl_annual.read_parameters(np.array([-5, 0]))
mdl_semiannual.read_parameters(np.array([0, 5]))
mdl_transient_1.read_parameters(np.array([40]))
mdl_transient_2.read_parameters(np.array([-4]))
mdl_transient_3.read_parameters(np.array([-20]))

# evaluate the synthetic models
sum_seas_sec = mdl_secular.evaluate(timevector)["fit"] \
               + mdl_annual.evaluate(timevector)["fit"] \
               + mdl_semiannual.evaluate(timevector)["fit"]
sum_transient = mdl_transient_1.evaluate(timevector)["fit"] \
                + mdl_transient_2.evaluate(timevector)["fit"] \
                + mdl_transient_3.evaluate(timevector)["fit"]
sum_all_models = sum_seas_sec + sum_transient

# create white and colored noise
from disstans.tools import create_powerlaw_noise  # noqa: E402
rng = np.random.default_rng(0)
white_noise = rng.normal(scale=2, size=(timevector.size, 1))
colored_noise = create_powerlaw_noise(size=(timevector.size, 1),
                                      exponent=1.5, seed=0) * 2
sum_noise = white_noise + colored_noise

# create the synthetic data
synth_data = sum_all_models + sum_noise

# plot
import matplotlib.pyplot as plt  # noqa: E402
from pandas.plotting import register_matplotlib_converters  # noqa: E402
register_matplotlib_converters()  # improve how time data looks
plt.plot(timevector, sum_seas_sec, c='C1', label="Seasonal + Secular")
plt.plot(timevector, sum_transient, c='k', label="Transient")
plt.plot(timevector, sum_noise, c='0.5', lw=0.3, label="Noise")
plt.plot(timevector, synth_data, c='C0', ls='none', marker='.',
         markersize=2, alpha=0.5, label="Synthetic Data")
plt.xlabel("Time")
plt.ylim(-50, 250)
plt.ylabel("Displacement [mm]")
plt.legend(loc="upper left")
plt.savefig(f"{outdir}/tutorial_2a.png")

# now, for the transients, add a SplineSet of ISplines to the collection
from disstans.models import ISpline, SplineSet  # noqa: E402
mdl_coll["Transient"] = SplineSet(degree=2,
                                  t_center_start=t_start_str,
                                  t_center_end=t_end_str,
                                  list_num_knots=[4, 8, 16, 32, 64, 128],
                                  splineclass=ISpline)

# create Network and Station
from disstans import Network, Station, Timeseries  # noqa: E402
net_name = "TutorialLand"
stat_name = "TUT"
caltech_lla = (34.1375, -118.125, 263)
net = Network(name=net_name)
stat = Station(name=stat_name,
               location=caltech_lla)
net[stat_name] = stat

# add truth, data timeseries and data models to Station
ts = Timeseries.from_array(timevector=timevector,
                           data=synth_data,
                           src="synthetic",
                           data_unit="mm",
                           data_cols=["Total"])
truth = Timeseries.from_array(timevector=timevector,
                              data=sum_all_models,
                              src="synthetic",
                              data_unit="mm",
                              data_cols=["Total"])
stat["Displacement"] = ts
stat["Truth"] = truth
stat.add_local_model_dict(ts_description="Displacement",
                          model_dict=mdl_coll)

# fit and evaluate without any regularization
net.fit(ts_description="Displacement", solver="linear_regression")
net.evaluate(ts_description="Displacement", output_description="Fit_noreg")

# create residual and error timeseries
stat["Res_noreg"] = stat["Displacement"] - stat["Fit_noreg"]
stat["Err_noreg"] = stat["Fit_noreg"] - stat["Truth"]
_ = stat.analyze_residuals(ts_description="Res_noreg",
                           mean=True, std=True, verbose=True)
"""
TUT: Res_noreg                          Mean  Standard Deviation
Total-Displacement_Model_Total  1.610461e-08            2.046004
"""

# plot fit and residual
from matplotlib.lines import Line2D  # noqa: E402
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(stat["Displacement"].data, label="Synthetic")
ax[0].plot(stat["Fit_noreg"].data, label="Fit")
ax[0].set_ylim(-50, 250)
ax[0].set_ylabel("Displacement [mm]")
ax[0].legend(loc="upper left")
ax[0].set_title("No Regularization")
ax[1].plot(stat["Res_noreg"].data, c='0.3', ls='none',
           marker='.', markersize=0.5)
ax[1].plot(stat["Err_noreg"].time, sum_noise, c='C1', ls='none',
           marker='.', markersize=0.5)
ax[1].plot(stat["Err_noreg"].data, c="C0")
ax[1].set_ylim(-15, 15)
ax[1].set_ylabel("Error [mm]")
custom_lines = [Line2D([0], [0], c="0.3", marker=".", linestyle='none'),
                Line2D([0], [0], c="C1", marker=".", linestyle='none'),
                Line2D([0], [0], c="C0")]
ax[1].legend(custom_lines, ["Residual", "True Noise", "Error"],
             loc="lower left", ncol=3)
fig.savefig(f"{outdir}/tutorial_2b.png")

# plot scalogram
fig, ax = stat.models["Displacement"]["Transient"].make_scalogram(t_left=t_start_str,
                                                                  t_right=t_end_str,
                                                                  cmaprange=20)
ax[0].set_title("No Regularization")
fig.savefig(f"{outdir}/tutorial_2c.png")

# repeat everything with L2 regularization
net.fit(ts_description="Displacement", solver="ridge_regression", penalty=10)
net.evaluate(ts_description="Displacement", output_description="Fit_L2")
stat["Res_L2"] = stat["Displacement"] - stat["Fit_L2"]
stat["Err_L2"] = stat["Fit_L2"] - stat["Truth"]
_ = stat.analyze_residuals(ts_description="Res_L2",
                           mean=True, std=True, verbose=True)
"""
TUT: Res_L2                             Mean  Standard Deviation
Total-Displacement_Model_Total  1.667917e-09            2.087589
"""

# plot fit and residual
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(stat["Displacement"].data, label="Synthetic")
ax[0].plot(stat["Fit_L2"].data, label="Fit")
ax[0].set_ylabel("Displacement [mm]")
ax[0].legend(loc="upper left")
ax[0].set_title("L2 Regularization")
ax[1].plot(stat["Res_L2"].data, c='0.3', ls='none',
           marker='.', markersize=0.5)
ax[1].plot(stat["Err_L2"].time, sum_noise, c='C1', ls='none',
           marker='.', markersize=0.5)
ax[1].plot(stat["Err_L2"].data, c="C0")
ax[1].set_ylim(-15, 15)
ax[1].set_ylabel("Error [mm]")
custom_lines = [Line2D([0], [0], c="0.3", marker=".", linestyle='none'),
                Line2D([0], [0], c="C1", marker=".", linestyle='none'),
                Line2D([0], [0], c="C0")]
ax[1].legend(custom_lines, ["Residual", "True Noise", "Error"],
             loc="lower left", ncol=3)
fig.savefig(f"{outdir}/tutorial_2d.png")

# plot scalogram
fig, ax = stat.models["Displacement"]["Transient"].make_scalogram(t_left=t_start_str,
                                                                  t_right=t_end_str,
                                                                  cmaprange=20)
ax[0].set_title("L2 Regularization")
fig.savefig(f"{outdir}/tutorial_2e.png")

# repeat everything with L1 regularization
net.fit(ts_description="Displacement", solver="lasso_regression", penalty=10)
net.evaluate(ts_description="Displacement", output_description="Fit_L1")
stat["Res_L1"] = stat["Displacement"] - stat["Fit_L1"]
stat["Err_L1"] = stat["Fit_L1"] - stat["Truth"]
_ = stat.analyze_residuals(ts_description="Res_L1",
                           mean=True, std=True, verbose=True)
"""
TUT: Res_L1                         Mean  Standard Deviation
Total-Displacement_Model_Total  0.000003            2.121952
"""

# plot fit and residual
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(stat["Displacement"].data, label="Synthetic")
ax[0].plot(stat["Fit_L1"].data, label="Fit")
ax[0].set_ylabel("Displacement [mm]")
ax[0].legend(loc="upper left")
ax[0].set_title("L1 Regularization")
ax[1].plot(stat["Res_L1"].data, c='0.3', ls='none',
           marker='.', markersize=0.5)
ax[1].plot(stat["Err_L1"].time, sum_noise, c='C1', ls='none',
           marker='.', markersize=0.5)
ax[1].plot(stat["Err_L1"].data, c="C0")
ax[1].set_ylim(-15, 15)
ax[1].set_ylabel("Error [mm]")
custom_lines = [Line2D([0], [0], c="0.3", marker=".", linestyle='none'),
                Line2D([0], [0], c="C1", marker=".", linestyle='none'),
                Line2D([0], [0], c="C0")]
ax[1].legend(custom_lines, ["Residual", "True Noise", "Error"],
             loc="lower left", ncol=3)
fig.savefig(f"{outdir}/tutorial_2f.png")

# plot scalogram
fig, ax = stat.models["Displacement"]["Transient"].make_scalogram(t_left=t_start_str,
                                                                  t_right=t_end_str,
                                                                  cmaprange=20)
ax[0].set_title("L1 Regularization")
fig.savefig(f"{outdir}/tutorial_2g.png")

# repeat everything with reweighted L1 regularization
net.fit(ts_description="Displacement", solver="lasso_regression",
        penalty=10, reweight_max_iters=10)
net.evaluate(ts_description="Displacement", output_description="Fit_L1R")
stat["Res_L1R"] = stat["Displacement"] - stat["Fit_L1R"]
stat["Err_L1R"] = stat["Fit_L1R"] - stat["Truth"]
_ = stat.analyze_residuals(ts_description="Res_L1R",
                           mean=True, std=True, verbose=True)
"""
TUT: Res_L1R                        Mean  Standard Deviation
Total-Displacement_Model_Total -0.000001            2.119665
"""

# plot fit and residual
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(stat["Displacement"].data, label="Synthetic")
ax[0].plot(stat["Fit_L1R"].data, label="Fit")
ax[0].set_ylabel("Displacement [mm]")
ax[0].legend(loc="upper left")
ax[0].set_title("Reweighted L1 Regularization")
ax[1].plot(stat["Res_L1R"].data, c='0.3', ls='none',
           marker='.', markersize=0.5)
ax[1].plot(stat["Err_L1R"].time, sum_noise, c='C1', ls='none',
           marker='.', markersize=0.5)
ax[1].plot(stat["Err_L1R"].data, c="C0")
ax[1].set_ylim(-15, 15)
ax[1].set_ylabel("Error [mm]")
custom_lines = [Line2D([0], [0], c="0.3", marker=".", linestyle='none'),
                Line2D([0], [0], c="C1", marker=".", linestyle='none'),
                Line2D([0], [0], c="C0")]
ax[1].legend(custom_lines, ["Residual", "True Noise", "Error"],
             loc="lower left", ncol=3)
fig.savefig(f"{outdir}/tutorial_2h.png")

# plot scalogram
fig, ax = stat.models["Displacement"]["Transient"].make_scalogram(t_left=t_start_str,
                                                                  t_right=t_end_str,
                                                                  cmaprange=20)
ax[0].set_title("Reweighted L1 Regularization")
fig.savefig(f"{outdir}/tutorial_2i.png")

# compare the secular and seasonal rates to the truth
reldiff_sec = (mdl_coll_synth["Secular"].parameters
               / stat.models["Displacement"]["Secular"].parameters).ravel() - 1
reldiff_ann_amp = mdl_coll_synth["Annual"].amplitude \
                  / stat.models["Displacement"]["Annual"].amplitude - 1
reldiff_sem_amp = mdl_coll_synth["Semi-Annual"].amplitude \
                  / stat.models["Displacement"]["Semi-Annual"].amplitude - 1
absdiff_ann_ph = mdl_coll_synth["Annual"].phase \
                  - stat.models["Displacement"]["Annual"].phase
absdiff_sem_ph = mdl_coll_synth["Semi-Annual"].phase \
                  - stat.models["Displacement"]["Semi-Annual"].phase
print(f"Percent Error Constant:              {reldiff_sec[0]: %}\n"
      f"Percent Error Linear:                {reldiff_sec[1]: %}\n"
      f"Percent Error Annual Amplitude:      {reldiff_ann_amp: %}\n"
      f"Percent Error Semi-Annual Amplitude: {reldiff_sem_amp: %}\n"
      f"Absolute Error Annual Phase:         {absdiff_ann_ph: f} rad\n"
      f"Absolute Error Semi-Annual Phase:    {absdiff_sem_ph: f} rad")
"""
Percent Error Constant:              -34.870393%
Percent Error Linear:                 14.251579%
Percent Error Annual Amplitude:      -1.046252%
Percent Error Semi-Annual Amplitude: -0.037603%
Absolute Error Annual Phase:          0.019479 rad
Absolute Error Semi-Annual Phase:    -0.017371 rad
"""

# get the trend for the time of rapid transient deformation
trend, _ = stat.get_trend("Displacement", fit_list=["Transient"],
                          t_start="2002-06-01", t_end="2002-08-01")
print(f"Transient Velocity: {trend[0]:f} {ts.data_unit}/D")
"""
Transient Velocity: 0.120261 mm/D
"""
