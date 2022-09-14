import os
import numpy as np
import healpy as hp
from astropy.io import fits

from rubin_sim.scheduler.utils import Sky_area_generator


def read_galplane_footprint(aggregation_level):
    #  https://github.com/LSST-TVSSC/software_tools is original source for the files here
    # specifically, tvs_software_tools/GalPlaneSurvey/HighCadenceZone
    rootdir = "."
    if isinstance(aggregation_level, float):
        aggregation_level = f"{aggregation_level :.1f}"
    if aggregation_level == "1.5":
        filename = "aggregated_priority_map_p1.5_a1.5_sum.fits"
    elif aggregation_level == "2.0":
        filename = "aggregated_priority_map_p2.0_a2.0_sum.fits"
    else:
        raise ValueError(
            f"Looking for either 1.5 or 2.0 (str) for aggregation_level, "
            f"not {aggregation_level}"
        )

    with fits.open(os.path.join(rootdir, filename)) as hdu1:
        gp = hdu1[1].data["pixelPriority"]
    return gp


def set_filter_ratios(**kwargs):
    # kwargs -- can override keys in dictionary by providing "magellenic_clouds_ratios = {dict}" for example
    filter_ratios = {}
    # Previous values from sky_area_generator default kwargs (in return_maps)
    filter_ratios["scp_ratios"] = {
        "u": 0.1,
        "g": 0.1,
        "r": 0.1,
        "i": 0.1,
        "z": 0.1,
        "y": 0.1,
    }
    filter_ratios["nes_ratios"] = {"g": 0.28, "r": 0.4, "i": 0.4, "z": 0.28}

    filter_ratios["low_dust_ratios"] = {
        "u": 0.32,
        "g": 0.4,
        "r": 1.0,
        "i": 1.0,
        "z": 0.9,
        "y": 0.9,
    }
    filter_ratios["virgo_ratios"] = {
        "u": 0.32,
        "g": 0.4,
        "r": 1.0,
        "i": 1.0,
        "z": 0.9,
        "y": 0.9,
    }
    # Previous values for these TVS/SLVMW priority fields
    # filter_ratios['magellenic_clouds_ratios'] = {"u": 0.32, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9}
    # filter_ratios['dusty_plane_ratios'] = {"u": 0.1, "g": 0.28, "r": 0.28, "i": 0.28, "z": 0.28, "y": 0.1}
    # filter_ratios['bulge_ratios'] = {"u": 0.18, "g": 1.0, "r": 1.05, "i": 1.05, "z": 1.0, "y": 0.23}

    # New values from aggregation maps
    gp_ratios = {"u": 0.74, "g": 1.24, "r": 0.55, "i": 0.55, "z": 0.61, "y": 0.84}

    filter_ratios["bulge_ratios"] = gp_ratios
    filter_ratios["magellenic_clouds_ratios"] = gp_ratios
    filter_ratios["dusty_plane_ratios"] = {f: gp_ratios[f] / 3.5 for f in "ugrizy"}

    if kwargs is not None:
        for k in kwargs:
            filter_ratios[k] = kwargs[k]

    return filter_ratios


def return_maps_galplane_sky(nside, gp, filter_ratios=None):

    if filter_ratios is None:
        filter_ratios = set_filter_ratios()

    # Turn gp map into mask for locations of high priority fields, and resample
    gp_mask = np.where(gp > 0, 1, 0)
    gp_mask = hp.ud_grade(gp_mask, nside)

    # Set up baseline sky coverage.
    # Sky_area_generator - from rubin_sim/scheduler/utils
    # Note smaller coverage over LMC/SMC .. not sure it's significant, but
    # is closer to requested from TVS/SLVMW
    sky = Sky_area_generator(nside=nside, smc_radius=4, lmc_radius=6)

    sky.pix_labels = np.zeros(hp.nside2npix(sky.nside), dtype="U20")
    sky.healmaps = np.zeros(
        hp.nside2npix(sky.nside),
        dtype=list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7)),
    )
    sky.add_magellanic_clouds(
        filter_ratios["magellenic_clouds_ratios"], lmc_ra=89, lmc_dec=-70
    )
    sky.add_lowdust_wfd(filter_ratios["low_dust_ratios"])
    sky.add_virgo_cluster(filter_ratios["virgo_ratios"])

    # Add galplane high priority regions
    label = "bulge"
    indx = np.where((gp_mask > 0) & (sky.pix_labels == ""))
    sky.pix_labels[indx] = label
    for filtername in filter_ratios["bulge_ratios"]:
        sky.healmaps[filtername][indx] = filter_ratios["bulge_ratios"][filtername]

    # sky.add_bulge(bulge_ratios)
    sky.add_nes(filter_ratios["nes_ratios"])
    sky.add_dusty_plane(filter_ratios["dusty_plane_ratios"])
    sky.add_scp(filter_ratios["scp_ratios"])

    return sky.healmaps, sky.pix_labels
