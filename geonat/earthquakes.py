"""
This module contains functions relating to the processing and representation
of earthquakes.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from okada_wrapper import dc3d0wrapper as dc3d0
from warnings import warn

from .config import defaults
from .tools import parallelize
from .models import Step


def _okada_get_displacements(station_and_parameters):
    """
    Parallelizable sub-function of okada_prior that for a single earthquake
    source calculates the displacement response for many locations.
    """
    # unpack inputs
    stations, eq = station_and_parameters
    # rotate from relative lat, lon, alt to xyz
    strike_rad = eq['strike']*np.pi/180
    R = np.array([[np.cos(strike_rad),  np.sin(strike_rad), 0],
                  [np.sin(strike_rad), -np.cos(strike_rad), 0],
                  [0, 0, 1]])
    stations = stations @ R
    # get displacements in xyz-frame
    disp = np.zeros_like(stations)
    for i in range(stations.shape[0]):
        success, u, grad_u = dc3d0(eq['alpha'], stations[i, :],
                                   eq['depth'], eq['dip'], eq['potency'])
        # unit for u is [unit of potency] / [unit of station location & depth]^2
        # unit for grad_u is [unit of potency] / [unit of station location & depth]^3
        # assume potency is Nm/GPa = 1e-9 m^3 and locations are in km,
        # then u is in [1e-15 m] and grad_u in [1e-18 m]
        if success == 0:
            disp[i, :] = u / 10**12  # output is now in mm
        else:
            warn(f"Success = {success} for station {i}!", category=RuntimeWarning)
    # transform back to lat, lon, alt
    # yes this is the same matrix
    disp = disp @ R
    return disp


def _okada_get_cumdisp(time_station_settings):
    """
    Parallelizable sub-function of okada_prior that for a single station
    calculates the cumulative displacements at the timestamps contained
    in the timeseries of that station.
    """
    eq_times, stat_time, station_disp, threshold = time_station_settings
    steptimes = []
    for itime in range(len(stat_time) - 1):
        disp = station_disp[(eq_times > stat_time[itime])
                            & (eq_times <= stat_time[itime + 1]), :]
        cumdisp = np.sum(np.linalg.norm(disp, axis=1), axis=None)
        if cumdisp >= threshold:
            steptimes.append(str(stat_time[itime]))
    return steptimes


def okada_prior(network, catalog_path, target_timeseries, target_model,
                target_model_regularize=False, catalog_prior_kw_args={}):
    r"""
    Given a catalog of earthquakes (including moment tensors), calculate an approximate
    displacement for each of the stations in the network, and add a step model to
    the target timeseries.

    This function operates on the network instance directly.

    Parameters
    ---------
    network : Network
        Network instance whose stations should be used.
    catalog_path : str
        File name of the earthquake catalog to load. Currently, only the Japanese
        NIED's F-net earthquake mechanism catalog format is supported.
    target_timeseries : str
        Name of the timeseries to add the model to.
    target_model : str
        Name of the earthquake model added to ``target_timeseries``.
    target_model_regularize : bool, optional
        Whether to mark the model for regularization or not.
    catalog_prior_kw_args : dict, optional
        A dictionary fine-tuning the displacement calculation and modeling, see
        :attr:`~geonat.config.defaults` for explanations and defaults.

    Notes
    -----
    This function uses Okada's [okada92]_ dislocation calculation subroutines coded in
    Fortran and wrapped by the Python package `okada_wrapper`_.

    The catalog format needs to have the following named columns:
    *Date,Origin_Time(JST),Latitude(°),Longitude(°),JMA_Depth(km),JMA_Magnitude(Mj),*
    *Region_Name,Strike,Dip,Rake,Mo(Nm),MT_Depth(km),MT_Magnitude(Mw),Var._Red.,*
    *mxx,mxy,mxz,myy,myz,mzz,Unit(Nm),Number_of_Stations*

    The keywords in ``catalog_prior_kw_args`` are:

    ============= ====== ==============================================================
    Keyword       Units  Description
    ============= ====== ==============================================================
    ``mu``        GPa    Shear modulus μ of the elastic half space
    ``alpha``     \-     Medium constant α=(λ+μ)/(λ+2μ), where λ is the first Lamé
                         parameter and μ the second one (shear modulus)
    ``threshold`` mm     Minimum amount of calculated displacement that a station needs
                         to surpass in order for a step to be added to the model.
    ============= ====== ==============================================================

    References
    ----------

    .. _`okada_wrapper`: https://github.com/tbenthompson/okada_wrapper
    .. [okada92] Yoshimitsu Okada (1992),
       *Internal deformation due to shear and tensile faults in a half-space*.
       Bulletin of the Seismological Society of America, 82 (2): 1018–1040.
    """
    catalog_prior_settings = defaults["prior"].copy()
    catalog_prior_settings.update(catalog_prior_kw_args)
    stations_lla = np.array([station.location for station in network])
    # convert height from m to km
    stations_lla[:, 2] /= 1000
    # load earthquake catalog
    catalog = pd.read_csv(catalog_path, header=0, parse_dates=[[0, 1]])
    eq_times = catalog['Date_Origin_Time(JST)']
    eq_lla = catalog[['Latitude(°)', 'Longitude(°)',  'MT_Depth(km)']].values
    eq_lla[:, 2] *= -1
    n_eq = eq_lla.shape[0]
    # relative position in lla
    stations_rel = [np.array(stations_lla - eq_lla[i, :].reshape(1, -1)) for i in range(n_eq)]
    # transform to xyz space, coarse approximation, ignores z-component of stations
    for i in range(n_eq):
        stations_rel[i][:, 0] *= 111.13292 - 0.55982*np.cos(2*eq_lla[i, 0]*np.pi/180)
        stations_rel[i][:, 1] *= 111.41284*np.cos(eq_lla[i, 0]*np.pi/180)
        stations_rel[i][:, 2] = 0
    # stations_rel is now in km, just like depth

    # compute station displacements
    parameters = ((stations_rel[i], {'alpha': catalog_prior_settings["alpha"],
                                     'lat': eq_lla[i, 0], 'lon': eq_lla[i, 1],
                                     'depth': -eq_lla[i, 2],
                                     'strike': float(catalog['Strike'][i].split(';')[0]),
                                     'dip': float(catalog['Dip'][i].split(';')[0]),
                                     'potency': [catalog['Mo(Nm)'][i]
                                                 / catalog_prior_settings["mu"], 0, 0, 0]})
                  for i in range(n_eq))
    station_disp = np.zeros((n_eq, stations_lla.shape[0], 3))
    for i, result in enumerate(tqdm(parallelize(_okada_get_displacements,
                                                parameters, chunksize=100),
                                    ascii=True, total=n_eq, unit="eq",
                                    desc="Simulating Earthquake Displacements")):
        station_disp[i, :, :] = result

    # add steps to station timeseries if they exceed the threshold
    station_names = list(network.stations.keys())
    cumdisp_parameters = ((eq_times,
                           network.stations[stat_name].timeseries[target_timeseries].time.values,
                           station_disp[:, istat, :], catalog_prior_settings["threshold"])
                          for istat, stat_name in enumerate(station_names))
    for i, result in enumerate(tqdm(parallelize(_okada_get_cumdisp, cumdisp_parameters),
                                    ascii=True, total=len(network.stations),
                                    desc="Adding steps where necessary", unit="station")):
        stepmodel = Step(steptimes=result, regularize=target_model_regularize)
        network.stations[station_names[i]].add_local_model(target_timeseries, target_model,
                                                           stepmodel)
