"""
This is a very simple first introduction to disstans.
"""

# make output folder
import os
outdir = "out/tutorial_1"
os.makedirs(outdir, exist_ok=True)

# build a station
from disstans import Station  # noqa: E402
synth_stat = Station(name="TUT", location=(34.05, -118.25, 93))

# make a model collection
import numpy as np  # noqa: E402
from disstans.models import Polynomial, Sinusoid, Step  # noqa: E402
mdl_secular = Polynomial(order=1, time_unit="D", t_reference="2000-01-01")
mdl_secular.read_parameters(np.array([-1, 5e-3]))
mdl_annual = Sinusoid(period=365.25, time_unit="D", t_reference="2000-01-01")
mdl_annual.read_parameters(np.array([0.3, 0]))
mdl_steps = Step(steptimes=["2000-03-01", "2000-10-10", "2000-10-15"])
mdl_steps.read_parameters(np.array([0.5, 0.2, -0.01]))
collection = {"Secular": mdl_secular,
              "Annual": mdl_annual,
              "Steps": mdl_steps}

# evaluate the models
import pandas as pd  # noqa: E402
timevector = pd.date_range(start="2000-01-01", end="2000-12-31", freq="1D")
sum_models = np.zeros((timevector.size, 1))
for model_description, model in collection.items():
    evaluated = model.evaluate(timevector)
    sum_models += evaluated["fit"]

# create a synthetic timeseries
from disstans import Timeseries  # noqa: E402
synth_ts = Timeseries.from_array(timevector=timevector,
                                 data=sum_models,
                                 src="synthetic",
                                 data_unit="m",
                                 data_cols=["total"])
synth_stat["Data"] = synth_ts

# add noise to synthetic timeseries
np.random.seed(1)  # make this example reproducible
noise = np.random.randn(*synth_stat["Data"].shape)*0.01
synth_stat["Data"].data += noise

# plot the synthetic data
import matplotlib.pyplot as plt  # noqa: E402
plt.plot(synth_stat["Data"].data)
plt.savefig(f"{outdir}/tutorial_1a.png")
plt.close()

# add the models to fit to the timeseries
for model_description, model in collection.items():
    synth_stat.add_local_model("Data", model_description, model)

# run the fitting
from disstans.solvers import linear_regression  # noqa: E402
result = linear_regression(ts=synth_stat["Data"],
                           models=synth_stat.models["Data"])

# quick access to the model collection
stat_coll = synth_stat.models["Data"]
# give the model collection the best-fit parameters and covariances
stat_coll.read_parameters(result.parameters, result.covariances)
# evaluate each individual model and add as a fit
for model_description in stat_coll.model_names:
    modeled = stat_coll[model_description].evaluate(timevector)
    fit_ts = synth_stat.add_fit(ts_description="Data",
                                fit=modeled,
                                model_description=model_description)
# evaluate the entire model collection at once
modeled = stat_coll.evaluate(timevector)
fit_ts = synth_stat.add_fit(ts_description="Data", fit=modeled)

synth_stat.add_timeseries(ts_description="Modeled",
                          timeseries=synth_stat.fits["Data"].allfits,
                          override_src="model", override_data_cols=synth_ts.data_cols)

# plot the data and fit
plt.plot(synth_stat["Data"].data, label="Data")
plt.plot(synth_stat["Modeled"].data, label="Modeled")
plt.legend()
plt.savefig(f"{outdir}/tutorial_1b.png")
plt.close()

# calculate the residuals
synth_stat["Residual"] = synth_stat["Data"] - synth_stat["Modeled"]
stats_dict = synth_stat.analyze_residuals(ts_description="Residual",
                                          mean=True, std=True, verbose=True)
"""
TUT: Residual          Mean  Standard Deviation
total-total   -6.784447e-15            0.009595
"""

# plot the residuals
plt.plot(synth_stat["Residual"].data)
plt.savefig(f"{outdir}/tutorial_1c.png")
plt.close()
