"""
This is a tutorial creating a synthetic network to test
the Euler Pole and strain estimation methods.
"""

if __name__ == "__main__":

    # imports
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib import rcParams
    from scipy.spatial.distance import cdist
    from cmcrameri import cm
    from disstans.tools import (get_hom_vel_strain_rot, get_field_vel_strain_rot,
                                strain_rotation_invariants,
                                estimate_euler_pole, rotvec2eulerpole)

    # decide output quality, looks & folder
    outdir = Path("out/tutorial_6/")
    rcParams['font.sans-serif'] = ["NewComputerModernSans10"]
    rcParams['font.size'] = "14"
    fmt = "png"
    os.makedirs(outdir, exist_ok=True)

    # initialize RNG
    rng = np.random.default_rng(0)
    # make stations
    locs = rng.uniform(low=[-121, 33],
                       high=[-116, 37],
                       size=(100, 2))
    n_locs = locs.shape[0]
    # get indices west and east of plate boundary
    i_west = locs[:, 0] < -119
    i_east = ~i_west
    locs_west = locs[i_west, :]
    locs_east = locs[i_east, :]
    # make velocities
    # create noisy zero background velocity field
    vels = rng.normal(scale=1e-3, size=(n_locs, 2))
    # add far away euler pole motion to the western plate
    vels[i_west, :] += np.array([[-0.055, 0.045]])
    # add inflation motion from volcano based on distance and direction
    center_vol = np.array([-117.1, 35.4])
    dist_vol = cdist(locs, center_vol[None, :]).ravel()
    vecs_vol = locs - center_vol[None, :]
    azims_vol = np.arctan2(vecs_vol[:, 1], vecs_vol[:, 0])
    vels_volcano = (vecs_vol / np.linalg.norm(vecs_vol, axis=1, keepdims=True)
                    / dist_vol[:, None]**2 / 70)
    vels += vels_volcano
    # make for uncertainties
    vels_var = (rng.uniform(low=9e-3, high=11e-3, size=(n_locs, 2))) ** 2
    vels_cov = rng.normal(scale=1e-6, size=(n_locs, 1))
    vels_varcov = np.concatenate([vels_var, vels_cov], axis=1)

    # network plot
    plt.figure(constrained_layout=True)
    plt.scatter(locs_west[:, 0], locs_west[:, 1], label="West")
    plt.scatter(locs_east[:, 0], locs_east[:, 1], label="East")
    q = plt.quiver(locs[:, 0], locs[:, 1],
                   vels[:, 0], vels[:, 1], scale=1)
    plt.quiverkey(q, 0.36, 0.02, 0.1, "100 mm/a")
    plt.gca().set_aspect("equal")
    plt.xlim(-121, -116)
    plt.ylim(33, 37)
    plt.xlabel("Longitude °E")
    plt.ylabel("Latitude °N")
    plt.legend(loc="lower right")
    plt.savefig(outdir / f"tutorial_6a.{fmt}", dpi=300, bbox_inches="tight")

    # get global strain somewhere in the west
    v_hom, eps_hom, om_hom = get_hom_vel_strain_rot(locs,
                                                    vels,
                                                    covariances=vels_varcov,
                                                    reference=[-120.5, 35])
    # convert to scalars
    dil_hom, strain_hom, shear_hom, rot_hom = strain_rotation_invariants(eps_hom, om_hom)
    print(f"Average motion = [{v_hom[0] * 1000:.2f}, {v_hom[1] * 1000:.2f}] mm/a\n"
          f"Average rotation = {rot_hom * 1e6:.4f} rad/Ma")

    # get euler pole for the western plate
    rotvec_west, rotcov_west = estimate_euler_pole(locs[i_west, :],
                                                vels[i_west, :],
                                                covariances=vels_varcov[i_west, :])
    # convert to more readable format
    ep_west, ep_cov_west = rotvec2eulerpole(rotvec_west, rotcov_west)
    ep_west_deg = np.rad2deg(ep_west)
    print(f"Longitude = {ep_west_deg[0]:.1f} °E\n"
          f"Latitude = {ep_west_deg[1]:.1f} °N\n"
          f"Rotation = {ep_west_deg[2] * 1e6:.1f} °/Ma")

    # define field coordinates
    x_range = np.linspace(-121, -116, num=50)
    y_range = np.linspace(33, 37, num=50)[::-1]
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    xy_mesh = np.stack([x_mesh.ravel(), y_mesh.ravel()], axis=1)
    # get strain fields
    v_field, eps_field, om_field = \
        get_field_vel_strain_rot(locs,
                                 vels,
                                 xy_mesh,
                                 2,
                                 covariances=vels_varcov,
                                 estimate_within=1e6,
                                 distance_method="quadratic",
                                 coverage_method="voronoi")
    # convert to scalar fields
    dil_field, strain_field, shear_field, rot_field = \
        strain_rotation_invariants(eps_field, om_field)

    # test checks
    print(f"Velocity check: {v_field[1275, 0]:.4e}")
    print(f"Dilatation check: {dil_field[1275]:.4e}")
    print(f"Strain check: {strain_field[1275]:.4e}")
    print(f"Shear check: {shear_field[1275]:.4e}")
    print(f"Rotation check: {rot_field[1275]:.4e}")

    # velocity field
    v_north_field = v_field.reshape(list(x_mesh.shape) + [2])[:, :, 1]
    plt.figure(constrained_layout=True)
    plt.pcolormesh(x_range, y_range,
                   v_north_field * 1000,
                   cmap=cm.vik,
                   vmin=-50,
                   vmax=50,
                   shading="gouraud")
    cb = plt.colorbar(shrink=0.7)
    cb.set_label("Velocity [mm/a]", labelpad=10)
    q = plt.quiver(locs[:, 0], locs[:, 1],
                   vels[:, 0], vels[:, 1], scale=1)
    plt.quiverkey(q, 0.36, 0.02, 0.1, "100 mm/a")
    plt.xlim(-121, -116)
    plt.ylim(33, 37)
    plt.xlabel("Longitude °E")
    plt.ylabel("Latitude °N")
    plt.gca().set_aspect("equal")
    plt.savefig(outdir / f"tutorial_6b.{fmt}", dpi=300, bbox_inches="tight")

    # shear strain field
    plt.figure(constrained_layout=True)
    plt.pcolormesh(x_range, y_range,
                   shear_field.reshape(x_mesh.shape) * 1e9,
                   cmap=cm.lajolla,
                   vmin=40,
                   vmax=150,
                   shading="gouraud")
    cb = plt.colorbar(shrink=0.7)
    cb.set_label("Max Shear Strain [nanostrain/a]", labelpad=10)
    q = plt.quiver(locs[:, 0], locs[:, 1],
                   vels[:, 0], vels[:, 1], scale=1)
    plt.quiverkey(q, 0.36, 0.02, 0.1, "100 mm/a")
    plt.xlim(-121, -116)
    plt.ylim(33, 37)
    plt.xlabel("Longitude °E")
    plt.ylabel("Latitude °N")
    plt.gca().set_aspect("equal")
    plt.savefig(outdir / f"tutorial_6c.{fmt}", dpi=300, bbox_inches="tight")
