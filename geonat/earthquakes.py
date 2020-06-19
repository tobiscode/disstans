import numpy as np
import pandas as pd
from tqdm import tqdm
from okada_wrapper import dc3d0wrapper as dc3d0
from warnings import warn

from . import defaults
from .tools import parallelize
from .model import Step


def _okada_get_displacements(station_and_parameters):
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
        success, u, grad_u = dc3d0(eq['alpha'], stations[i, :], eq['depth'], eq['dip'], eq['potency'])
        # unit for u is [unit of potency] / [unit of station location & depth]^2
        # unit for grad_u is [unit of potency] / [unit of station location & depth]^3
        # assume potency is Nm/GPa = 1e-9 m^3 and locations are in km,
        # then u is in [1e-15 m] and grad_u in [1e-18 m]
        if success == 0:
            disp[i, :] = u * 10**12  # output is now in mm
        else:
            warn(f"Success = {success} for station {i}!", category=RuntimeWarning)
    # transform back to lat, lon, alt
    # yes this is the same matrix
    disp = disp @ R
    return disp


def _okada_get_cumdisp(time_station_settings):
    eq_times, stat_time, station_disp, threshold = time_station_settings
    steptimes = []
    for itime in range(len(stat_time) - 1):
        disp = station_disp[(eq_times > stat_time[itime]) & (eq_times <= stat_time[itime + 1]), :]
        cumdisp = np.sum(np.linalg.norm(disp, axis=1), axis=None)
        if cumdisp >= threshold:
            steptimes.append(str(stat_time[itime]))
    return steptimes


def okada_prior(network, catalog_path, target_timeseries, target_model, target_model_regularize=False, catalog_prior_kw_args={}):
    catalog_prior_settings = defaults["catalog_prior"].copy()
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
    parameters = ((stations_rel[i], {'alpha': catalog_prior_settings["alpha"], 'lat': eq_lla[i, 0], 'lon': eq_lla[i, 1],
                                     'depth': -eq_lla[i, 2], 'strike': float(catalog['Strike'][i].split(';')[0]),
                                     'dip': float(catalog['Dip'][i].split(';')[0]),
                                     'potency': [catalog['Mo(Nm)'][i]/catalog_prior_settings["mu"], 0, 0, 0]})
                  for i in range(n_eq))
    station_disp = np.zeros((n_eq, stations_lla.shape[0], 3))
    for i, result in enumerate(tqdm(parallelize(_okada_get_displacements, parameters, chunksize=100),
                                    ascii=True, total=n_eq, desc="Simulating Earthquake Displacements", unit="eq")):
        station_disp[i, :, :] = result

    # add steps to station timeseries if they exceed the threshold
    station_names = list(network.stations.keys())
    cumdisp_parameters = ((eq_times, network.stations[stat_name].timeseries[target_timeseries].time.values,
                           station_disp[:, istat, :], catalog_prior_settings["threshold"])
                          for istat, stat_name in enumerate(station_names))
    for i, result in enumerate(tqdm(parallelize(_okada_get_cumdisp, cumdisp_parameters),
                                    ascii=True, total=len(network.stations), desc="Adding steps where necessary", unit="station")):
        network.stations[station_names[i]].add_local_model(target_timeseries, target_model,
                                                           Step(steptimes=result, regularize=target_model_regularize))
