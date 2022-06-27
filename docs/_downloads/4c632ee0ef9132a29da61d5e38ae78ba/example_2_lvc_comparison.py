"""
Second example of the DISSTANS documentation:
Long Valley Caldera - Secular Velocity Comparison

This example demonstrates how to load non-DISSTANS models in order to
use DISSTANS' functionality using UNR's MIDAS and GAGE's secular
velocity solutions, showing the difference of DISSTANS' fitting approach.
The differences will become more visible after removing an average background
velocity field, which will be calculated in two different ways.

Note: This example only contains simple plotting, but saves all the data
necessary for final comparisons in a GMT-compatible format.
"""

# imports
import os
import pickle
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
from urllib import request
from disstans.tools import (R_ecef2enu, strain_rotation_invariants,
                            estimate_euler_pole, rotvec2eulerpole)

# make plots look better
from matplotlib import rcParams
rcParams['font.sans-serif'] = ["NewComputerModernSans10"]
rcParams['font.size'] = "14"

# start script
if __name__ == "__main__":

    # preparation
    main_dir = Path("proj_dir").resolve()
    data_dir = main_dir / "data/gnss"
    gnss_dir = data_dir / "longvalley"
    out_dir_ex1 = main_dir / "out/example_1"
    out_dir_ex2 = main_dir / "out/example_2"
    os.makedirs(out_dir_ex2, exist_ok=True)
    os.chdir(out_dir_ex2)

    # load network object of example 1
    print("Loading example 1 network objects... ", end="", flush=True)
    with gzip.open(f"{out_dir_ex1}/example_1_net.pkl.gz", "rb") as f:
        net = pickle.load(f)

    # load stations dataset
    stations_df = pd.read_pickle(f"{gnss_dir}/downloaded.pkl.gz")

    # load secular velocities from example 1
    all_poly_df = pd.read_csv(f"{out_dir_ex1}/example_1_secular_velocities.csv", index_col=0)
    poly_stat_names = all_poly_df.index.tolist()
    print("Done")

    # get predicted velocities from uniform motion & deformation
    v_pred_hom, v_O, epsilon, omega = \
        net.hom_velocity_field(ts_description="final", mdl_description="Linear")

    # get invariants
    dilatation, strain, shear, rotation = strain_rotation_invariants(epsilon, omega)
    print("Homogenous transation, rotation, and strain field results:\n"
          f"Dilatation rate: {dilatation:.2e} 1/a\n"
          f"Strain rate:     {strain:.2e} 1/a\n"
          f"Shearing rate:   {shear:.2e} 1/a\n"
          f"Rotation rate:   {rotation:.2e} rad/a")

    # get predicted velocity from Euler pole estimation
    v_pred_euler, rotation_vector, rotation_covariance = \
        net.euler_rot_field(ts_description="final", mdl_description="Linear")

    # get Euler pole notation
    euler_pole, euler_pole_covariance = \
        rotvec2eulerpole(rotation_vector, rotation_covariance)
    # convert to degrees for printing
    euler_pole = np.rad2deg(euler_pole)
    euler_pole_sd = np.rad2deg(np.sqrt(np.diag(euler_pole_covariance)))
    print("Euler pole results (with one s.d.):\n"
          f"Longitude: {euler_pole[0]:.3f} +/- {euler_pole_sd[0]:.3f} °\n"
          f"Latitude:  {euler_pole[1]:.3f} +/- {euler_pole_sd[1]:.3f} °\n"
          f"Rate:      {euler_pole[2]:.2g} +/- {euler_pole_sd[2]:.2g} °/a")

    # get locations of the stations that have solutions
    station_locs = net.station_locations.loc[v_pred_euler.index.tolist(), :].to_numpy()

    # preparation
    proj_gui = ccrs.Mercator()
    proj_lla = ccrs.PlateCarree()

    # plot comparison between the modeled and the two predicted velocity fields
    fig_map = plt.figure(figsize=(8, 5))
    ax_map = fig_map.add_subplot(projection=proj_gui)
    ax_map.gridlines(draw_labels=True, zorder=-1)
    ax_map.set_extent([-120.3, -117.75, 36.9, 38.3])
    q = ax_map.quiver(station_locs[:, 1], station_locs[:, 0],
                      v_pred_hom["east"], v_pred_hom["north"],
                      units="xy", scale=5e-4, width=1e3, color="C2",
                      transform=proj_lla, label="Homogenous")
    q = ax_map.quiver(station_locs[:, 1], station_locs[:, 0],
                      v_pred_euler["east"], v_pred_euler["north"],
                      units="xy", scale=5e-4, width=1e3, color="C5",
                      transform=proj_lla, label="Euler")
    q = ax_map.quiver(station_locs[:, 1], station_locs[:, 0],
                      all_poly_df["vel_e"], all_poly_df["vel_n"],
                      units="xy", scale=5e-4, width=1e3, color="k",
                      transform=proj_lla, label="Modeled")
    ax_map.quiverkey(q, 0.85, 0.85, 10, "10 mm/a", color="0.3")
    ax_map.legend(loc="lower center", ncol=3, fontsize="small")
    ax_map.set_title("Secular Velocity (IGS)")
    fig_map.savefig(out_dir_ex2 / "example_2_single.png", dpi=300)
    plt.close(fig_map)

    # download MIDAS velocity file if not present
    fname_midas = Path(f"{data_dir}/midas.IGS14.txt")
    if not fname_midas.is_file():
        url_midas = "http://geodesy.unr.edu/velocities/midas.IGS14.txt"
        print(f"Downloading {url_midas} to {fname_midas}... ", end="", flush=True)
        with request.urlopen(url_midas) as u, open(fname_midas, mode="wb") as f:
            f.write(u.read())
        print("Done")
    else:
        print(f"Found {fname_midas}")

    # repeat for GAGE
    fname_gage = Path(f"{data_dir}/pbo.final_igs14.vel")
    if not fname_gage.is_file():
        url_gage = "https://data.unavco.org/archive/gnss/products/velocity/pbo.final_igs14.vel"
        print(f"Downloading {url_gage} to {fname_gage}... ", end="", flush=True)
        with request.urlopen(url_gage) as u, open(fname_gage, mode="wb") as f:
            f.write(u.read())
        print("Done")
    else:
        print(f"Found {fname_gage}")

    # load MIDAS velocities
    print("Reading MIDAS and GAGE velocities... ", end="", flush=True)
    v_mdl_midas = pd.read_csv(fname_midas,
                              header=0, delim_whitespace=True,
                              names=["sta", "label", "t(1)", "t(m)", "delt", "m", "mgood",
                                     "n", "ve50", "vn50", "vu50", "sve", "svn", "svu",
                                     "xe50", "xn50", "xu50", "fe", "fn", "fu",
                                     "sde", "sdn", "sdu", "nstep", "lat", "lon", "alt"])
    v_mdl_midas.set_index("sta", inplace=True, verify_integrity=True)
    v_mdl_midas.sort_index(inplace=True)

    # load GAGE velocities
    v_mdl_gage = pd.read_fwf(fname_gage, header=35,
                             widths=[5, 17, 15, 11, 16, 15, 15, 16, 16, 11, 11, 9, 9, 8,
                                     8, 8, 7, 7, 7, 11, 9, 9, 8, 8, 8, 7, 7, 7, 15, 15],
                             names=["Dot#", "Name", "Ref_epoch", "Ref_jday",
                                    "Ref_X", "Ref_Y", "Ref_Z",
                                    "Ref_Nlat", "Ref_Elong", "Ref_Up",
                                    "dX/dt", "dY/dt", "dZ/dt", "SXd", "SYd", "SZd",
                                    "Rxy", "Rxz", "Ryz", "dN/dt", "dE/dt", "dU/dt",
                                    "SNd", "SEd", "SUd", "Rne", "Rnu", "Reu",
                                    "first_epoch", "last_epoch"])
    # manually checked that the last occurrence of repeated station velocities
    # is the median of the reported velocities
    v_mdl_gage.drop_duplicates(subset=["Dot#"], inplace=True, keep="last")
    v_mdl_gage.set_index("Dot#", inplace=True, verify_integrity=True)
    v_mdl_gage.sort_index(inplace=True)
    print("Done")

    # copy into a DataFrame with the same column names as DISSTANS
    # make a list of stations that are common to all three datasets (GAGE is smaller)
    common_stations = sorted(list(set(poly_stat_names)
                                  & set(v_mdl_midas.index.tolist())
                                  & set(v_mdl_gage.index.tolist())))

    # get sublists of stations inside a bounding box, approximating caldera extent
    station_lolaal = net.station_locations.loc[
        common_stations, ["Longitude [°]", "Latitude [°]", "Altitude [m]"]].to_numpy()
    lons, lats, alts = station_lolaal[:, 0], station_lolaal[:, 1], station_lolaal[:, 2]
    inner_bbox = [-119-12/60, -118-33/60, 37+30/60, 37+50/60]
    inner_shell = ((lons > 360 + inner_bbox[0]) &
                   (lons < 360 + inner_bbox[1]) &
                   (lats > inner_bbox[2]) &
                   (lats < inner_bbox[3]))
    outer_shell = ~inner_shell
    inner_stations = np.array(common_stations)[inner_shell].tolist()
    outer_stations = np.array(common_stations)[outer_shell].tolist()

    # reduce DataFrame to what we need for the velocities comparison
    v_mdl = {}
    v_mdl["DISSTANS"] = all_poly_df.loc[
        common_stations, ["vel_e", "vel_n", "sig_vel_e", "sig_vel_n"]].copy() / 1000
    v_mdl["DISSTANS"]["corr_vel_en"] = all_poly_df.loc[common_stations, ["corr_vel_en"]].copy()

    # also convert the MIDAS and GAGE velocity dataframes into the same format
    v_mdl["MIDAS"] = pd.DataFrame({"vel_e": v_mdl_midas.loc[common_stations, "ve50"],
                                   "vel_n": v_mdl_midas.loc[common_stations, "vn50"],
                                   "sig_vel_e": v_mdl_midas.loc[common_stations, "sve"],
                                   "sig_vel_n": v_mdl_midas.loc[common_stations, "svn"],
                                   "corr_vel_en": np.zeros(len(common_stations))},
                                  index=common_stations)
    v_mdl["GAGE"] = pd.DataFrame({"vel_e": v_mdl_gage.loc[common_stations, "dE/dt"],
                                  "vel_n": v_mdl_gage.loc[common_stations, "dN/dt"],
                                  "sig_vel_e": v_mdl_gage.loc[common_stations, "SEd"],
                                  "sig_vel_n": v_mdl_gage.loc[common_stations, "SNd"],
                                  "corr_vel_en": v_mdl_gage.loc[common_stations, "Rne"]},
                                 index=common_stations)

    # we don't need to rotate into the NA frame, but will help other plots
    # get predicted NA rotation (from what GAGE uses)
    # ITRF2014 plate motion model: https://doi.org/10.1093/gji/ggx136
    rot_NA_masperyear = np.array([0.024, -0.694, -0.063])
    rot_NA = rot_NA_masperyear / 1000 * np.pi / 648000  # [rad/a]
    crs_lla = ccrs.Geodetic()
    crs_xyz = ccrs.Geocentric()
    stat_xyz = crs_xyz.transform_points(crs_lla, lons, lats, alts)
    v_NA_xyz = np.cross(rot_NA, stat_xyz)  # [m/a]
    v_NA_enu = pd.DataFrame(
        data=np.stack([(R_ecef2enu(lo, la) @ v_NA_xyz[i, :]) for i, (lo, la)
                       in enumerate(zip(lons, lats))]),
        index=common_stations, columns=["vel_e", "vel_n", "vel_u"])

    # remove NA plate velocity
    v_mdl_na = {case: (-v_NA_enu.iloc[:, :2]) + v_mdl[case][["vel_e", "vel_n"]].values
                for case in v_mdl.keys()}

    # to use estimate_euler_pole, we need the (co)variances instead of the correlations
    v_mdl_covs = {case: np.stack(
        [v_mdl[case]["sig_vel_e"].values**2, v_mdl[case]["sig_vel_n"].values**2,
         np.prod(v_mdl[case][["sig_vel_e", "sig_vel_n", "corr_vel_en"]].values, axis=1)
         ], axis=1)
        for case in v_mdl.keys()}

    # estimate Euler pole for all stations, and get residuals
    v_pred, v_res = {}, {}
    for case in ["DISSTANS", "MIDAS", "GAGE"]:
        # get solution for all stations
        rotation_vector = estimate_euler_pole(station_lolaal[:, :2],
                                              v_mdl_na[case].values,
                                              v_mdl_covs[case])[0]
        # calculate the surface motion at each station
        v_temp = np.cross(rotation_vector, stat_xyz)
        # rotate the motion into the local ENU frame
        v_pred[case] = pd.DataFrame(
            data=np.stack([(R_ecef2enu(lo, la) @ v_temp[i, :]) for i, (lo, la)
                           in enumerate(zip(lons, lats))])[:, :2],
            index=common_stations, columns=["vel_e", "vel_n"])
        # get residual
        v_res[case] = v_mdl_na[case] - v_pred[case].values

    # repeat with just the outer stations being used in the estimation
    v_pred_os, v_res_os = {}, {}
    for case in ["DISSTANS", "MIDAS", "GAGE"]:
        rotation_vector = estimate_euler_pole(station_lolaal[outer_shell, :2],
                                              v_mdl_na[case].values[outer_shell, :],
                                              v_mdl_covs[case][outer_shell, :])[0]
        v_temp = np.cross(rotation_vector, stat_xyz)
        v_pred_os[case] = pd.DataFrame(
            data=np.stack([(R_ecef2enu(lo, la) @ v_temp[i, :]) for i, (lo, la)
                           in enumerate(zip(lons, lats))])[:, :2],
            index=common_stations, columns=["vel_e", "vel_n"])
        v_res_os[case] = v_mdl_na[case] - v_pred_os[case].values

    # saving is all going to be the same columns and rows, so can initialize an
    # empty DataFrame to change contents
    print("Saving GMT-compatible velocity files... ", end="", flush=True)
    df_gmt = pd.DataFrame({"lon[°]": lons, "lat[°]": lats} |
                          {c: np.zeros(len(common_stations)) for c in v_mdl[case].columns} |
                          {"station": common_stations})
    # save into different files in GMT-readable format
    for case in ["DISSTANS", "MIDAS", "GAGE"]:
        # copy over the uncertainties which are always the same
        df_gmt.iloc[:, 4:-1] = v_mdl[case].iloc[:, -3:].values
        for vname, vdf in zip(["mdl", "pred", "res"], [v_mdl_na, v_pred, v_res]):
            # fill in only the velocities themselves
            df_gmt.iloc[:, 2:4] = vdf[case].iloc[:, :2].values
            # save
            df_gmt.to_csv(f"example_2_v_{vname}_{case}.csv", index=False)
    print("Done")

    # compare the predictions among velocity models
    # RMS of magnitude of the difference vector
    rmses_predmdls = pd.DataFrame(
        {"D2M": [(np.linalg.norm(vp["DISSTANS"].values - vp["MIDAS"].values,
                 axis=1)**2).mean()**0.5 for vp in [v_pred, v_pred_os]],
         "D2G": [(np.linalg.norm(vp["DISSTANS"].values - vp["GAGE"].values,
                 axis=1)**2).mean()**0.5 for vp in [v_pred, v_pred_os]],
         "M2G": [(np.linalg.norm(vp["MIDAS"].values - vp["GAGE"].values,
                 axis=1)**2).mean()**0.5 for vp in [v_pred, v_pred_os]]},
        index=["common", "outer"])

    # now we want the norm of the residuals for all station, just the outer stations,
    # and just the inner stations, for both estimation subsets
    rmses_res = pd.DataFrame(np.zeros((3, 3)),
                             columns=list(v_res.keys()), index=["all", "outer", "inner"])
    rmses_res_os, rmses_resdiff = rmses_res.copy(), rmses_res.copy()
    for sub_name, sub_ix in zip(["all", "outer", "inner"],
                                [[True] * outer_shell.size, outer_shell, ~outer_shell]):
        rmses_res.loc[sub_name, :] = \
            [(np.linalg.norm(v_res[case].values[sub_ix, :], axis=1)**2).mean()**0.5
             for case in v_res.keys()]
        rmses_res_os.loc[sub_name, :] = \
            [(np.linalg.norm(v_res_os[case].values[sub_ix, :], axis=1)**2).mean()**0.5
             for case in v_res_os.keys()]
        # compare the residuals between the common and outer stations cases
        # this is the same as comparing the predictions, as the plate velocity cancels out
        rmses_resdiff.loc[sub_name, :] = \
            [(np.linalg.norm(v_res[case].values[sub_ix, :]
                             - v_res_os[case].values[sub_ix, :], axis=1
                             )**2).mean(axis=0)**0.5 for case in v_res_os.keys()]

    # print
    print("\nDifference between model predictions\n",
          rmses_predmdls,
          "\n\nResiduals common_stations\n",
          rmses_res,
          "\n\nResiduals outer_stations\n",
          rmses_res_os,
          "\n\nDifference of residuals between station subsets\n",
          rmses_resdiff,
          "\n\nRelative improvement from other models to DISSTANS\n",
          (rmses_res.iloc[:, 1:] - rmses_res.iloc[:, 0].values.reshape(-1, 1)
           ) / rmses_res.iloc[:, 1:].values, "\n", sep="")

    # start cartopy plots
    print("Plotting with Cartopy... ", end="", flush=True)

    # IGS velocities
    fig_map = plt.figure(figsize=(8, 5))
    ax_map = fig_map.add_subplot(projection=proj_gui)
    ax_map.gridlines(draw_labels=True, zorder=-1)
    ax_map.set_extent([-120.25, -117.75, 36.9, 38.3])
    for case, col in zip(reversed(v_mdl.keys()), ["C0", "C1", "k"]):
        q = ax_map.quiver(lons, lats,
                          v_mdl[case]["vel_e"], v_mdl[case]["vel_n"],
                          units="xy", scale=5e-7, width=1e3, color=col,
                          transform=proj_lla, label=case)
    ax_map.quiverkey(q, 0.85, 0.85, 0.01, "10 mm/a", color="0.3")
    ax_map.legend(loc="lower center", ncol=3, fontsize="small")
    ax_map.set_title("Modeled (IGS)")
    fig_map.savefig(out_dir_ex2 / "example_2_modeled_igs.png", dpi=300)
    plt.close(fig_map)

    # NA velocities
    fig_map = plt.figure(figsize=(8, 5))
    ax_map = fig_map.add_subplot(projection=proj_gui)
    ax_map.gridlines(draw_labels=True, zorder=-1)
    ax_map.set_extent([-120.25, -117.75, 36.9, 38.3])
    for case, col in zip(reversed(v_mdl_na.keys()), ["C0", "C1", "k"]):
        q = ax_map.quiver(lons, lats,
                          v_mdl_na[case]["vel_e"], v_mdl_na[case]["vel_n"],
                          units="xy", scale=5e-7, width=1e3, color=col,
                          transform=proj_lla, label=case)
    ax_map.quiverkey(q, 0.85, 0.85, 0.01, "10 mm/a", color="0.3")
    ax_map.legend(loc="lower center", ncol=3, fontsize="small")
    ax_map.set_title("Modeled (NA)")
    fig_map.savefig(out_dir_ex2 / "example_2_modeled_na.png", dpi=300)
    plt.close(fig_map)

    # predicted velocities
    fig_map = plt.figure(figsize=(8, 5))
    ax_map = fig_map.add_subplot(projection=proj_gui)
    ax_map.gridlines(draw_labels=True, zorder=-1)
    ax_map.set_extent([-120.25, -117.75, 36.9, 38.3])
    for case, col in zip(reversed(v_pred.keys()), ["C0", "C1", "k"]):
        q = ax_map.quiver(lons, lats,
                          v_pred[case]["east"], v_pred[case]["north"],
                          units="xy", scale=5e-7, width=1e3, color=col,
                          transform=proj_lla, label=case)
    ax_map.quiverkey(q, 0.85, 0.85, 0.01, "10 mm/a", color="0.3")
    ax_map.legend(loc="lower center", ncol=3, fontsize="small")
    ax_map.set_title("Background velocity field (NA)")
    fig_map.savefig(out_dir_ex2 / "example_2_predicted.png", dpi=300)
    plt.close(fig_map)

    # residuals
    fig_map = plt.figure(figsize=(8, 5))
    ax_map = fig_map.add_subplot(projection=proj_gui)
    ax_map.gridlines(draw_labels=True, zorder=-1)
    ax_map.set_extent([-120.25, -117.75, 36.9, 38.3])
    for case, col in zip(reversed(v_res.keys()), ["C0", "C1", "k"]):
        q = ax_map.quiver(lons, lats,
                          v_res[case]["vel_e"], v_res[case]["vel_n"],
                          units="xy", scale=2e-7, width=1e3, color=col,
                          transform=proj_lla, label=case)
    ax_map.quiverkey(q, 0.85, 0.85, 0.005, "5 mm/a", color="0.3")
    ax_map.legend(loc="lower center", ncol=3, fontsize="small")
    ax_map.set_title("Residuals (NA)")
    fig_map.savefig(out_dir_ex2 / "example_2_residuals.png", dpi=300)
    plt.close(fig_map)

    # done
    print("Done")
