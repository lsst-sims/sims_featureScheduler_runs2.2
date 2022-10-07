#!/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt
import healpy as hp
from rubin_sim.scheduler.modelObservatory import Model_observatory
from rubin_sim.scheduler.schedulers import Core_scheduler, simple_filter_sched
from rubin_sim.scheduler.utils import (Sky_area_generator, TargetoO, Sim_targetoO_server,
                                       make_rolling_footprints, read_fields,
                                       comcamTessellate, gnomonic_project_toxy, tsp_convex,
                                       scheduled_observation)
import rubin_sim.scheduler.basis_functions as bf
from rubin_sim.scheduler.surveys import (Greedy_survey, Blob_survey, Scripted_survey,
                                         ToO_survey, ToO_master, BaseMarkovDF_survey)
from rubin_sim.scheduler import sim_runner
import rubin_sim.scheduler.detailers as detailers
from rubin_sim.utils import _hpid2RaDec, _angularSeparation, _approx_RaDec2AltAz, _raDec2Hpid
import sys
import subprocess
import os
import argparse
from make_ddf_survey import generate_ddf_scheduled_obs
import rubin_sim
# So things don't fail on hyak
from astropy.utils import iers
iers.conf.auto_download = False


def mean_longitude(longitude):
    """Compute a mean longitude, accounting for wrap around.
    """
    x = np.cos(longitude)
    y = np.sin(longitude)
    meanx = np.mean(x)
    meany = np.mean(y)
    angle = np.arctan2(meany, meanx)
    radius = np.sqrt(meanx**2 + meany**2)
    mid_longitude = angle % (2.*np.pi)
    if radius < 0.1:
        mid_longitude = np.pi
    return mid_longitude


# XXX--should move this to utils generally and refactor it out of base MarkovSurvey
def order_observations(RA, dec):
    """
    Take a list of ra,dec positions and compute a traveling salesman solution through them
    """
    # Let's find a good spot to project the points to a plane
    mid_dec = (np.max(dec) - np.min(dec))/2.
    mid_ra = mean_longitude(RA)
    # Project the coordinates to a plane. Could consider scaling things to represent
    # time between points rather than angular distance.
    pointing_x, pointing_y = gnomonic_project_toxy(RA, dec, mid_ra, mid_dec)
    # Round off positions so that we ensure identical cross-platform performance
    scale = 1e6
    pointing_x = np.round(pointing_x*scale).astype(int)
    pointing_y = np.round(pointing_y*scale).astype(int)
    # Now I have a bunch of x,y pointings. Drop into TSP solver to get an effiencent route
    towns = np.vstack((pointing_x, pointing_y)).T
    # Leaving optimize=False for speed. The optimization step doesn't usually improve much.
    better_order = tsp_convex(towns, optimize=False)
    return better_order


class ToO_scripted_survey(Scripted_survey, BaseMarkovDF_survey):
    """If there is a new ToO event, generate a set of scripted observations to try and follow it up.

    Parameters
    ----------
    times : list of floats
        The times after the detection that observations should be attempted (hours)

    """
    def __init__(self, basis_functions, followup_footprint=None, nside=32, reward_val=1e6, times=[0, 1, 2, 4, 24],
                 filters_at_times=['gz', 'gz', 'gz', 'gz', 'gz'], 
                 nvis=[1, 1, 1, 1, 6],
                 exptime=30., camera='LSST',
                 survey_name='ToO', flushtime=2., mjd_tol=1./24., dist_tol=0.5,
                 alt_min=20., alt_max=85., HA_min=5, HA_max=19, ignore_obs='dummy', dither=True,
                 seed=42, npositions=7305, n_snaps=2, n_usnaps=1, id_start=1,
                 detailers=None):
        # Figure out what else I need to super here

        self.basis_functions = basis_functions
        self.survey_name = survey_name
        self.followup_footprint = followup_footprint
        self.last_event_id = -1
        self.night = -1
        self.reward_val = reward_val
        self.times = np.array(times)/24.  # to days
        self.filters_at_times = filters_at_times
        self.exptime = exptime
        self.nvis = nvis
        self.n_snaps = n_snaps
        self.n_usnaps = n_usnaps
        self.nside = nside
        self.flushtime = flushtime/24.
        self.mjd_tol = mjd_tol
        self.dist_tol = np.radians(dist_tol)
        self.alt_min = np.radians(alt_min)
        self.alt_max = np.radians(alt_max)
        self.HA_min = HA_min
        self.HA_max = HA_max
        self.ignore_obs = ignore_obs
        self.extra_features = {}
        self.extra_basis_functions = {}
        self.detailers = []
        self.dither = dither
        self.id_start = id_start
        self.detailers = detailers

        self.camera = camera
        # Load the OpSim field tesselation and map healpix to fields
        if self.camera == 'LSST':
            self.fields_init = read_fields()
        elif self.camera == 'comcam':
            self.fields_init = comcamTessellate()
        else:
            ValueError('camera %s unknown, should be "LSST" or "comcam"' % camera)
        self.fields = self.fields_init.copy()
        self.hp2fields = np.array([])
        self._hp2fieldsetup(self.fields['RA'], self.fields['dec'])

        # Initialize the list of scripted observations
        self.clear_script()

        # Generate and store rotation positions to use.
        # This way, if different survey objects are seeded the same, they will
        # use the same dither positions each night
        rng = np.random.default_rng(seed)
        self.lon = rng.random(npositions)*np.pi*2
        # Make sure latitude points spread correctly
        # http://mathworld.wolfram.com/SpherePointPicking.html
        self.lat = np.arccos(2.*rng.random(npositions) - 1.)
        self.lon2 = rng.random(npositions)*np.pi*2

    def _check_list(self, conditions):
        """Check to see if the current mjd is good
        """
        observation = None
        if self.obs_wanted is not None:
            # Scheduled observations that are in the right time window and have not been executed
            in_time_window = np.where((self.mjd_start < conditions.mjd) &
                                      (self.obs_wanted['flush_by_mjd'] > conditions.mjd) &
                                      (~self.obs_wanted['observed']))[0]

            if np.size(in_time_window) > 0:
                pass_checks = self._check_alts_HA(self.obs_wanted[in_time_window], conditions)
                matches = in_time_window[pass_checks]
            else:
                matches = []

            if np.size(matches) > 0:
                # If we have something in the current filter, do that, otherwise whatever is first
                #in_filt = np.where(self.obs_wanted[matches]['filter'] == conditions.current_filter)[0]
                #if np.size(in_filt) > 0:
                #    indx = matches[in_filt[0]]
                #else:
                #    indx = matches[0]
                #observation = self._slice2obs(self.obs_wanted[indx])
                observation = self._slice2obs(self.obs_wanted[matches[0]])
                
        return observation

    def flush_script(self, conditions):
        """Remove things from the script that aren't needed anymore
        """
        if self.obs_wanted is not None:
            still_relevant = np.where((self.obs_wanted['observed'] == False) 
                                      & (self.obs_wanted['flush_by_mjd'] < conditions.mjd))[0]
            if np.size(still_relevant) > 0:
                observations = self.obs_wanted[still_relevant]
                self.set_script(observations)
            else:
                self.clear_script()

    def _new_event(self, target_o_o, conditions):
        """A new ToO event, generate any observations for followup
        """
        # flush out any old observations or ones that have been completed
        self.flush_script(conditions)
        # Check that the event center is in the footprint we want to observe
        hpid_center = _raDec2Hpid(self.nside, target_o_o.ra_rad_center, target_o_o.dec_rad_center)
        if self.followup_footprint[hpid_center] > 0:
            target_area = self.followup_footprint * target_o_o.footprint
            # generate a list of pointings for that area
            hpid_to_observe = np.where(target_area > 0)[0]

            # Check if we should spin the tesselation for the night.
            if self.dither & (conditions.night != self.night):
                self._spin_fields(conditions)
                self.night = conditions.night.copy()

            field_ids = np.unique(self.hp2fields[hpid_to_observe])

            # Put the fields in a good order. Skipping dither positions for now.
            better_order = order_observations(self.fields['RA'][field_ids],
                                              self.fields['dec'][field_ids])
            ras = self.fields['RA'][field_ids[better_order]]
            decs = self.fields['dec'][field_ids[better_order]]

            # Figure out an MJD start time for the object if it is still rising and low.
            alt, az = _approx_RaDec2AltAz(target_o_o.ra_rad_center, target_o_o.dec_rad_center,
                                          conditions.site.latitude_rad, None,
                                          conditions.mjd,
                                          lmst=conditions.lmst)
            HA = conditions.lmst - target_o_o.ra_rad_center*12./np.pi

            if (HA < self.HA_max) & (HA > self.HA_min):
                t_to_rise = (self.HA_max - HA)/24.
                mjd0 = conditions.mjd + t_to_rise
            else:
                mjd0 = conditions.mjd + 0.

            obs_list = []
            for time, filternames, nv in zip(self.times, self.filters_at_times, self.nvis):
                for filtername in filternames:
                    # Subsitute y for z if needed
                    if (filtername == 'z') & (filtername not in conditions.mounted_filters):
                        filtername = 'y'
                    for i in range(nv):
                        if filtername in conditions.mounted_filters:

                            if filtername == 'u':
                                nexp = self.n_usnaps
                            else:
                                nexp = self.n_snaps

                            obs = scheduled_observation(ras.size)
                            obs['RA'] = ras
                            obs['dec'] = decs
                            obs['mjd'] = mjd0 + time
                            obs['flush_by_mjd'] = mjd0 + time + self.flushtime
                            obs['exptime'] = self.exptime
                            obs['nexp'] = nexp
                            obs['filter'] = filtername
                            obs['rotSkyPos'] = 0  # XXX--maybe throw a rotation detailer in here
                            obs['mjd_tol'] = self.mjd_tol
                            obs['dist_tol'] = self.dist_tol
                            obs['alt_min'] = self.alt_min
                            obs['alt_max'] = self.alt_max
                            obs['HA_max'] = self.HA_max
                            obs['HA_min'] = self.HA_min

                            obs['note'] = self.survey_name + ', %i_t%i' % (target_o_o.id, time*24)
                            obs_list.append(obs)             
            observations = np.concatenate(obs_list)
            if self.obs_wanted is not None:
                if np.size(self.obs_wanted) > 0:
                    observations = np.concatenate([self.obs_wanted, observations])
            self.set_script(observations)

    def calc_reward_function(self, conditions):
        """If there is an observation ready to go, execute it, otherwise, -inf
        """
        # check if any new event has come in

        if conditions.targets_of_opportunity is not None:
            for target_o_o in conditions.targets_of_opportunity:
                if target_o_o.id > self.last_event_id:
                    self._new_event(target_o_o, conditions)
                    self.last_event_id = target_o_o.id

        observation = self._check_list(conditions)
        if observation is None:
            self.reward = -np.inf
        else:
            self.reward = self.reward_val
        return self.reward


def gen_greedy_surveys(nside=32, nexp=2, exptime=30., filters=['r', 'i', 'z', 'y'],
                       camera_rot_limits=[-80., 80.],
                       shadow_minutes=60., max_alt=76., moon_distance=30., ignore_obs='DD',
                       m5_weight=3., footprint_weight=0.75, slewtime_weight=3.,
                       stayfilter_weight=3., repeat_weight=-1., footprints=None):
    """
    Make a quick set of greedy surveys

    This is a convienence function to generate a list of survey objects that can be used with
    rubin_sim.scheduler.schedulers.Core_scheduler.
    To ensure we are robust against changes in the sims_featureScheduler codebase, all kwargs are
    explicitly set.

    Parameters
    ----------
    nside : int (32)
        The HEALpix nside to use
    nexp : int (1)
        The number of exposures to use in a visit.
    exptime : float (30.)
        The exposure time to use per visit (seconds)
    filters : list of str (['r', 'i', 'z', 'y'])
        Which filters to generate surveys for.
    camera_rot_limits : list of float ([-80., 80.])
        The limits to impose when rotationally dithering the camera (degrees).
    shadow_minutes : float (60.)
        Used to mask regions around zenith (minutes)
    max_alt : float (76.
        The maximium altitude to use when masking zenith (degrees)
    moon_distance : float (30.)
        The mask radius to apply around the moon (degrees)
    ignore_obs : str or list of str ('DD')
        Ignore observations by surveys that include the given substring(s).
    m5_weight : float (3.)
        The weight for the 5-sigma depth difference basis function
    footprint_weight : float (0.3)
        The weight on the survey footprint basis function.
    slewtime_weight : float (3.)
        The weight on the slewtime basis function
    stayfilter_weight : float (3.)
        The weight on basis function that tries to stay avoid filter changes.
    """
    # Define the extra parameters that are used in the greedy survey. I
    # think these are fairly set, so no need to promote to utility func kwargs
    greed_survey_params = {'block_size': 1, 'smoothing_kernel': None,
                           'seed': 42, 'camera': 'LSST', 'dither': True,
                           'survey_name': 'greedy'}

    surveys = []
    detailer_list = [detailers.Camera_rot_detailer(min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits))]
    detailer_list.append(detailers.Rottep2Rotsp_desired_detailer())

    for filtername in filters:
        bfs = []
        bfs.append((bf.M5_diff_basis_function(filtername=filtername, nside=nside), m5_weight))
        bfs.append((bf.Footprint_basis_function(filtername=filtername,
                                                footprint=footprints,
                                                out_of_bounds_val=np.nan, nside=nside), footprint_weight))
        bfs.append((bf.Slewtime_basis_function(filtername=filtername, nside=nside), slewtime_weight))
        bfs.append((bf.Strict_filter_basis_function(filtername=filtername), stayfilter_weight))
        bfs.append((bf.Visit_repeat_basis_function(gap_min=0, gap_max=18*60., filtername=None,
                                                   nside=nside, npairs=20), repeat_weight))
        # Masks, give these 0 weight
        bfs.append((bf.Zenith_shadow_mask_basis_function(nside=nside, shadow_minutes=shadow_minutes,
                                                         max_alt=max_alt), 0))
        bfs.append((bf.Moon_avoidance_basis_function(nside=nside, moon_distance=moon_distance), 0))

        bfs.append((bf.Filter_loaded_basis_function(filternames=filtername), 0))
        bfs.append((bf.Planet_mask_basis_function(nside=nside), 0))

        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        surveys.append(Greedy_survey(basis_functions, weights, exptime=exptime, filtername=filtername,
                                     nside=nside, ignore_obs=ignore_obs, nexp=nexp,
                                     detailers=detailer_list, **greed_survey_params))

    return surveys


def generate_blobs(nside, nexp=2, exptime=30., filter1s=['u', 'u', 'g', 'r', 'i', 'z', 'y'],
                   filter2s=['g', 'r', 'r', 'i', 'z', 'y', 'y'], pair_time=33.,
                   camera_rot_limits=[-80., 80.], n_obs_template=3,
                   season=300., season_start_hour=-4., season_end_hour=2.,
                   shadow_minutes=60., max_alt=76., moon_distance=30., ignore_obs='DD',
                   m5_weight=6., footprint_weight=1.5, slewtime_weight=3.,
                   stayfilter_weight=3., template_weight=12., u_template_weight=24., footprints=None, u_nexp1=True,
                   scheduled_respect=45., good_seeing={'g': 3, 'r': 3, 'i': 3}, good_seeing_weight=3.,
                   mjd_start=1, repeat_weight=-1):
    """
    Generate surveys that take observations in blobs.

    Parameters
    ----------
    nside : int (32)
        The HEALpix nside to use
    nexp : int (1)
        The number of exposures to use in a visit.
    exptime : float (30.)
        The exposure time to use per visit (seconds)
    filter1s : list of str
        The filternames for the first set
    filter2s : list of str
        The filter names for the second in the pair (None if unpaired)
    pair_time : float (33)
        The ideal time between pairs (minutes)
    camera_rot_limits : list of float ([-80., 80.])
        The limits to impose when rotationally dithering the camera (degrees).
    n_obs_template : int (3)
        The number of observations to take every season in each filter
    season : float (300)
        The length of season (i.e., how long before templates expire) (days)
    season_start_hour : float (-4.)
        For weighting how strongly a template image needs to be observed (hours)
    sesason_end_hour : float (2.)
        For weighting how strongly a template image needs to be observed (hours)
    shadow_minutes : float (60.)
        Used to mask regions around zenith (minutes)
    max_alt : float (76.
        The maximium altitude to use when masking zenith (degrees)
    moon_distance : float (30.)
        The mask radius to apply around the moon (degrees)
    ignore_obs : str or list of str ('DD')
        Ignore observations by surveys that include the given substring(s).
    m5_weight : float (3.)
        The weight for the 5-sigma depth difference basis function
    footprint_weight : float (0.3)
        The weight on the survey footprint basis function.
    slewtime_weight : float (3.)
        The weight on the slewtime basis function
    stayfilter_weight : float (3.)
        The weight on basis function that tries to stay avoid filter changes.
    template_weight : float (12.)
        The weight to place on getting image templates every season
    u_template_weight : float (24.)
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a little higher than
        the standard template_weight kwarg.
    u_nexp1 : bool (True)
        Add a detailer to make sure the number of expossures in a visit is always 1 for u observations.
    scheduled_respect : float (45)
        How much time to require there be before a pre-scheduled observation (minutes)
    """

    template_weights = {'u': u_template_weight, 'g': template_weight,
                        'r': template_weight, 'i': template_weight,
                        'z': template_weight, 'y': template_weight}

    blob_survey_params = {'slew_approx': 7.5, 'filter_change_approx': 140.,
                          'read_approx': 2., 'min_pair_time': 15., 'search_radius': 30.,
                          'alt_max': 85., 'az_range': 90., 'flush_time': 30.,
                          'smoothing_kernel': None, 'nside': nside, 'seed': 42, 'dither': True,
                          'twilight_scale': False}

    surveys = []

    times_needed = [pair_time, pair_time*2]
    for filtername, filtername2 in zip(filter1s, filter2s):
        detailer_list = []
        detailer_list.append(detailers.Camera_rot_detailer(min_rot=np.min(camera_rot_limits),
                                                           max_rot=np.max(camera_rot_limits)))
        detailer_list.append(detailers.Rottep2Rotsp_desired_detailer())
        detailer_list.append(detailers.Close_alt_detailer())
        detailer_list.append(detailers.Flush_for_sched_detailer())
        # List to hold tuples of (basis_function_object, weight)
        bfs = []

        if filtername2 is not None:
            bfs.append((bf.M5_diff_basis_function(filtername=filtername, nside=nside), m5_weight/2.))
            bfs.append((bf.M5_diff_basis_function(filtername=filtername2, nside=nside), m5_weight/2.))

        else:
            bfs.append((bf.M5_diff_basis_function(filtername=filtername, nside=nside), m5_weight))

        if filtername2 is not None:
            bfs.append((bf.Footprint_basis_function(filtername=filtername,
                                                    footprint=footprints,
                                                    out_of_bounds_val=np.nan, nside=nside), footprint_weight/2.))
            bfs.append((bf.Footprint_basis_function(filtername=filtername2,
                                                    footprint=footprints,
                                                    out_of_bounds_val=np.nan, nside=nside), footprint_weight/2.))
        else:
            bfs.append((bf.Footprint_basis_function(filtername=filtername,
                                                    footprint=footprints,
                                                    out_of_bounds_val=np.nan, nside=nside), footprint_weight))

        bfs.append((bf.Slewtime_basis_function(filtername=filtername, nside=nside), slewtime_weight))
        bfs.append((bf.Strict_filter_basis_function(filtername=filtername), stayfilter_weight))
        bfs.append((bf.Visit_repeat_basis_function(gap_min=0, gap_max=18*60., filtername=None,
                                                   nside=nside, npairs=20), repeat_weight))

        if filtername2 is not None:
            bfs.append((bf.N_obs_per_year_basis_function(filtername=filtername, nside=nside,
                                                         footprint=footprints.get_footprint(filtername),
                                                         n_obs=n_obs_template, season=season,
                                                         season_start_hour=season_start_hour,
                                                         season_end_hour=season_end_hour), template_weights[filtername]/2.))
            bfs.append((bf.N_obs_per_year_basis_function(filtername=filtername2, nside=nside,
                                                         footprint=footprints.get_footprint(filtername2),
                                                         n_obs=n_obs_template, season=season,
                                                         season_start_hour=season_start_hour,
                                                         season_end_hour=season_end_hour), template_weights[filtername2]/2.))
        else:
            bfs.append((bf.N_obs_per_year_basis_function(filtername=filtername, nside=nside,
                                                         footprint=footprints.get_footprint(filtername),
                                                         n_obs=n_obs_template, season=season,
                                                         season_start_hour=season_start_hour,
                                                         season_end_hour=season_end_hour), template_weight))

        # Insert things for getting good seeing templates
        if filtername2 is not None:
            if filtername in list(good_seeing.keys()):
                bfs.append((bf.N_good_seeing_basis_function(filtername=filtername, nside=nside, mjd_start=mjd_start,
                                                            footprint=footprints.get_footprint(filtername),
                                                            n_obs_desired=good_seeing[filtername]), good_seeing_weight))
            if filtername2 in list(good_seeing.keys()):
                bfs.append((bf.N_good_seeing_basis_function(filtername=filtername2, nside=nside, mjd_start=mjd_start,
                                                            footprint=footprints.get_footprint(filtername2),
                                                            n_obs_desired=good_seeing[filtername2]), good_seeing_weight))
        else:
            if filtername in list(good_seeing.keys()):
                bfs.append((bf.N_good_seeing_basis_function(filtername=filtername, nside=nside, mjd_start=mjd_start,
                                                            footprint=footprints.get_footprint(filtername),
                                                            n_obs_desired=good_seeing[filtername]), good_seeing_weight))
        # Make sure we respect scheduled observations
        bfs.append((bf.Time_to_scheduled_basis_function(time_needed=scheduled_respect), 0))
        # Masks, give these 0 weight
        bfs.append((bf.Zenith_shadow_mask_basis_function(nside=nside, shadow_minutes=shadow_minutes, max_alt=max_alt,
                                                         penalty=np.nan, site='LSST'), 0.))
        bfs.append((bf.Moon_avoidance_basis_function(nside=nside, moon_distance=moon_distance), 0.))
        filternames = [fn for fn in [filtername, filtername2] if fn is not None]
        bfs.append((bf.Filter_loaded_basis_function(filternames=filternames), 0))
        if filtername2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append((bf.Time_to_twilight_basis_function(time_needed=time_needed), 0.))
        bfs.append((bf.Not_twilight_basis_function(), 0.))
        bfs.append((bf.Planet_mask_basis_function(nside=nside), 0.))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        if filtername2 is None:
            survey_name = 'blob, %s' % filtername
        else:
            survey_name = 'blob, %s%s' % (filtername, filtername2)
        if filtername2 is not None:
            detailer_list.append(detailers.Take_as_pairs_detailer(filtername=filtername2))

        if u_nexp1:
            detailer_list.append(detailers.Filter_nexp(filtername='u', nexp=1))
        surveys.append(Blob_survey(basis_functions, weights, filtername1=filtername, filtername2=filtername2,
                                   exptime=exptime,
                                   ideal_pair_time=pair_time,
                                   survey_note=survey_name, ignore_obs=ignore_obs,
                                   nexp=nexp, detailers=detailer_list, **blob_survey_params))

    return surveys


def generate_twi_blobs(nside, nexp=2, exptime=30., filter1s=['r', 'i', 'z', 'y'],
                       filter2s=['i', 'z', 'y', 'y'], pair_time=15.,
                       camera_rot_limits=[-80., 80.], n_obs_template=3,
                       season=300., season_start_hour=-4., season_end_hour=2.,
                       shadow_minutes=60., max_alt=76., moon_distance=30., ignore_obs='DD',
                       m5_weight=6., footprint_weight=1.5, slewtime_weight=3.,
                       stayfilter_weight=3., template_weight=12., footprints=None, repeat_night_weight=None,
                       wfd_footprint=None, scheduled_respect=15., repeat_weight=-1.):
    """
    Generate surveys that take observations in blobs.

    Parameters
    ----------
    nside : int (32)
        The HEALpix nside to use
    nexp : int (1)
        The number of exposures to use in a visit.
    exptime : float (30.)
        The exposure time to use per visit (seconds)
    filter1s : list of str
        The filternames for the first set
    filter2s : list of str
        The filter names for the second in the pair (None if unpaired)
    pair_time : float (22)
        The ideal time between pairs (minutes)
    camera_rot_limits : list of float ([-80., 80.])
        The limits to impose when rotationally dithering the camera (degrees).
    n_obs_template : int (3)
        The number of observations to take every season in each filter
    season : float (300)
        The length of season (i.e., how long before templates expire) (days)
    season_start_hour : float (-4.)
        For weighting how strongly a template image needs to be observed (hours)
    sesason_end_hour : float (2.)
        For weighting how strongly a template image needs to be observed (hours)
    shadow_minutes : float (60.)
        Used to mask regions around zenith (minutes)
    max_alt : float (76.
        The maximium altitude to use when masking zenith (degrees)
    moon_distance : float (30.)
        The mask radius to apply around the moon (degrees)
    ignore_obs : str or list of str ('DD')
        Ignore observations by surveys that include the given substring(s).
    m5_weight : float (3.)
        The weight for the 5-sigma depth difference basis function
    footprint_weight : float (0.3)
        The weight on the survey footprint basis function.
    slewtime_weight : float (3.)
        The weight on the slewtime basis function
    stayfilter_weight : float (3.)
        The weight on basis function that tries to stay avoid filter changes.
    template_weight : float (12.)
        The weight to place on getting image templates every season
    u_template_weight : float (24.)
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a little higher than
        the standard template_weight kwarg.
    """

    blob_survey_params = {'slew_approx': 7.5, 'filter_change_approx': 140.,
                          'read_approx': 2., 'min_pair_time': 10., 'search_radius': 30.,
                          'alt_max': 85., 'az_range': 90., 'flush_time': 30.,
                          'smoothing_kernel': None, 'nside': nside, 'seed': 42, 'dither': True,
                          'twilight_scale': False, 'in_twilight': True}

    surveys = []

    times_needed = [pair_time, pair_time*2]
    for filtername, filtername2 in zip(filter1s, filter2s):
        detailer_list = []
        detailer_list.append(detailers.Camera_rot_detailer(min_rot=np.min(camera_rot_limits),
                                                           max_rot=np.max(camera_rot_limits)))
        detailer_list.append(detailers.Rottep2Rotsp_desired_detailer())
        detailer_list.append(detailers.Close_alt_detailer())
        detailer_list.append(detailers.Flush_for_sched_detailer())
        # List to hold tuples of (basis_function_object, weight)
        bfs = []

        if filtername2 is not None:
            bfs.append((bf.M5_diff_basis_function(filtername=filtername, nside=nside), m5_weight/2.))
            bfs.append((bf.M5_diff_basis_function(filtername=filtername2, nside=nside), m5_weight/2.))

        else:
            bfs.append((bf.M5_diff_basis_function(filtername=filtername, nside=nside), m5_weight))

        if filtername2 is not None:
            bfs.append((bf.Footprint_basis_function(filtername=filtername,
                                                    footprint=footprints,
                                                    out_of_bounds_val=np.nan, nside=nside), footprint_weight/2.))
            bfs.append((bf.Footprint_basis_function(filtername=filtername2,
                                                    footprint=footprints,
                                                    out_of_bounds_val=np.nan, nside=nside), footprint_weight/2.))
        else:
            bfs.append((bf.Footprint_basis_function(filtername=filtername,
                                                    footprint=footprints,
                                                    out_of_bounds_val=np.nan, nside=nside), footprint_weight))

        bfs.append((bf.Slewtime_basis_function(filtername=filtername, nside=nside), slewtime_weight))
        bfs.append((bf.Strict_filter_basis_function(filtername=filtername), stayfilter_weight))
        bfs.append((bf.Visit_repeat_basis_function(gap_min=0, gap_max=18*60., filtername=None,
                                                   nside=nside, npairs=20), repeat_weight))

        if filtername2 is not None:
            bfs.append((bf.N_obs_per_year_basis_function(filtername=filtername, nside=nside,
                                                         footprint=footprints.get_footprint(filtername),
                                                         n_obs=n_obs_template, season=season,
                                                         season_start_hour=season_start_hour,
                                                         season_end_hour=season_end_hour), template_weight/2.))
            bfs.append((bf.N_obs_per_year_basis_function(filtername=filtername2, nside=nside,
                                                         footprint=footprints.get_footprint(filtername2),
                                                         n_obs=n_obs_template, season=season,
                                                         season_start_hour=season_start_hour,
                                                         season_end_hour=season_end_hour), template_weight/2.))
        else:
            bfs.append((bf.N_obs_per_year_basis_function(filtername=filtername, nside=nside,
                                                         footprint=footprints.get_footprint(filtername),
                                                         n_obs=n_obs_template, season=season,
                                                         season_start_hour=season_start_hour,
                                                         season_end_hour=season_end_hour), template_weight))
        if repeat_night_weight is not None:
            bfs.append((bf.Avoid_long_gaps_basis_function(nside=nside, filtername=None,
                                                          min_gap=0., max_gap=10./24., ha_limit=3.5,
                                                          footprint=wfd_footprint), repeat_night_weight))
        # Make sure we respect scheduled observations
        bfs.append((bf.Time_to_scheduled_basis_function(time_needed=scheduled_respect), 0))
        # Masks, give these 0 weight
        bfs.append((bf.Zenith_shadow_mask_basis_function(nside=nside, shadow_minutes=shadow_minutes, max_alt=max_alt,
                                                         penalty=np.nan, site='LSST'), 0.))
        bfs.append((bf.Moon_avoidance_basis_function(nside=nside, moon_distance=moon_distance), 0.))
        filternames = [fn for fn in [filtername, filtername2] if fn is not None]
        bfs.append((bf.Filter_loaded_basis_function(filternames=filternames), 0))
        if filtername2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append((bf.Time_to_twilight_basis_function(time_needed=time_needed, alt_limit=12), 0.))
        bfs.append((bf.Planet_mask_basis_function(nside=nside), 0.))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        if filtername2 is None:
            survey_name = 'blob_twi, %s' % filtername
        else:
            survey_name = 'blob_twi, %s%s' % (filtername, filtername2)
        if filtername2 is not None:
            detailer_list.append(detailers.Take_as_pairs_detailer(filtername=filtername2))
        surveys.append(Blob_survey(basis_functions, weights, filtername1=filtername, filtername2=filtername2,
                                   exptime=exptime,
                                   ideal_pair_time=pair_time,
                                   survey_note=survey_name, ignore_obs=ignore_obs,
                                   nexp=nexp, detailers=detailer_list, **blob_survey_params))

    return surveys


def ddf_surveys(detailers=None, season_frac=0.2, euclid_detailers=None):
    obs_array = generate_ddf_scheduled_obs(season_frac=season_frac)

    euclid_obs = np.where((obs_array['note'] == 'DD:EDFS_b') | (obs_array['note'] == 'DD:EDFS_a'))[0]
    all_other = np.where((obs_array['note'] != 'DD:EDFS_b') & (obs_array['note'] != 'DD:EDFS_a'))[0]

    survey1 = Scripted_survey([], detailers=detailers)
    survey1.set_script(obs_array[all_other])

    survey2 = Scripted_survey([], detailers=euclid_detailers)
    survey2.set_script(obs_array[euclid_obs])

    return [survey1, survey2]


def run_sched(surveys, observatory, survey_length=365.25, nside=32, fileroot='baseline_', verbose=False,
              extra_info=None, illum_limit=40., event_table=None):
    years = np.round(survey_length/365.25)
    scheduler = Core_scheduler(surveys, nside=nside)
    n_visit_limit = None
    filter_sched = simple_filter_sched(illum_limit=illum_limit)
    observatory, scheduler, observations = sim_runner(observatory, scheduler,
                                                      survey_length=survey_length,
                                                      filename=fileroot+'%iyrs.db' % years,
                                                      delete_past=True, n_visit_limit=n_visit_limit,
                                                      verbose=verbose, extra_info=extra_info,
                                                      filter_scheduler=filter_sched,
                                                      event_table=event_table)


def generate_events(nside=32, mjd_start=59853.5, radius=6.5, survey_length=365.25*10,
                    rate=10., expires=3., seed=42):
    """Generate a bunch of ToO events

    Parameters
    ----------
    rate : float (10)
        The number of events per year.
    expires : float (3)
        How long to keep broadcasting events as relevant (days)
    """

    np.random.seed(seed=seed)
    ra, dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
    radius = np.radians(radius)
    # Use a ceil here so we get at least 1 event even if doing a short run.
    n_events = int(np.ceil(survey_length/365.25*rate))
    names = ['mjd_start', 'ra', 'dec', 'expires']
    types = [float]*4
    event_table = np.zeros(n_events, dtype=list(zip(names, types)))

    event_table['mjd_start'] = np.sort(np.random.random(n_events))*survey_length + mjd_start
    event_table['expires'] = event_table['mjd_start']+expires
    # Make sure latitude points spread correctly
    # http://mathworld.wolfram.com/SpherePointPicking.html
    event_table['ra'] = np.random.rand(n_events)*np.pi*2
    event_table['dec'] = np.arccos(2.*np.random.rand(n_events) - 1.) - np.pi/2.

    events = []
    for i, event_time in enumerate(event_table['mjd_start']):
        dist = _angularSeparation(ra, dec, event_table['ra'][i], event_table['dec'][i])
        good = np.where(dist <= radius)
        footprint = np.zeros(ra.size, dtype=float)
        footprint[good] = 1
        events.append(TargetoO(i, footprint, event_time, expires,
                               ra_rad_center=event_table['ra'][i], dec_rad_center=event_table['dec'][i]))
    events = Sim_targetoO_server(events)
    return events, event_table


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument("--survey_length", type=float, default=365.25*10)
    parser.add_argument("--outDir", type=str, default="")
    parser.add_argument("--maxDither", type=float, default=0.7, help="Dither size for DDFs (deg)")
    parser.add_argument("--moon_illum_limit", type=float, default=40., help="illumination limit to remove u-band")
    parser.add_argument("--nexp", type=int, default=2)
    parser.add_argument("--rolling_nslice", type=int, default=2)
    parser.add_argument("--rolling_strength", type=float, default=0.9)
    parser.add_argument("--dbroot", type=str)
    parser.add_argument("--gsw", type=float, default=3.0)
    parser.add_argument("--ddf_season_frac", type=float, default=0.2)
    parser.add_argument("--too_rate", type=float, default=10, help="N events per year")
    parser.add_argument("--filters", type=str, default='gz')
    parser.add_argument("--nfollow", type=int, default=1)

    args = parser.parse_args()
    survey_length = args.survey_length  # Days
    outDir = args.outDir
    verbose = args.verbose
    max_dither = args.maxDither
    illum_limit = args.moon_illum_limit
    nexp = args.nexp
    nslice = args.rolling_nslice
    scale = args.rolling_strength
    dbroot = args.dbroot
    gsw = args.gsw
    too_rate = args.too_rate
    too_filters = args.filters
    too_nfollow = args.nfollow

    ddf_season_frac = args.ddf_season_frac

    nside = 32
    per_night = True  # Dither DDF per night

    camera_ddf_rot_limit = 75.

    extra_info = {}
    exec_command = ''
    for arg in sys.argv:
        exec_command += ' ' + arg
    extra_info['exec command'] = exec_command
    try:
        extra_info['git hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    except subprocess.CalledProcessError:
        extra_info['git hash'] = 'Not in git repo'

    extra_info['file executed'] = os.path.realpath(__file__)
    try:
        rs_path = rubin_sim.__path__[0]
        hash_file = os.path.join(rs_path, '../', '.git/refs/heads/main')
        extra_info['rubin_sim git hash'] = subprocess.check_output(['cat', hash_file])
    except subprocess.CalledProcessError:
        pass

    # Use the filename of the script to name the output database
    if dbroot is None:
        fileroot = os.path.basename(sys.argv[0]).replace('.py', '') + '_'
    else:
        fileroot = dbroot + '_'
    file_end = 'rate%i_%s_nf%i_' % (too_rate, too_filters, too_nfollow) + 'v2.2_'

    sm = Sky_area_generator(nside=nside)

    footprints_hp_array, labels = sm.return_maps()
    wfd_indx = np.where((labels == 'lowdust') | (labels == 'LMC_SMC') | (labels == 'virgo'))[0]
    wfd_footprint = footprints_hp_array['r']*0
    wfd_footprint[wfd_indx] = 1

    footprints_hp = {}
    for key in footprints_hp_array.dtype.names:
        footprints_hp[key] = footprints_hp_array[key]

    repeat_night_weight = None

    observatory = Model_observatory(nside=nside)
    conditions = observatory.return_conditions()
    sim_ToOs, event_table = generate_events(nside=nside, survey_length=survey_length,
                                            rate=too_rate, mjd_start=conditions.mjd_start)
    observatory = Model_observatory(nside=nside, sim_ToO=sim_ToOs)

    footprints = make_rolling_footprints(fp_hp=footprints_hp, mjd_start=conditions.mjd_start,
                                         sun_RA_start=conditions.sun_RA_start, nslice=nslice, scale=scale,
                                         nside=nside, wfd_indx=wfd_indx)

    # Let's make a footprint to follow up ToO events
    too_footprint = footprints_hp['r'] * 0 + np.nan
    too_footprint[np.where(footprints_hp['r'] > 0)[0]] = 1.

    # Set up the DDF surveys to dither
    u_detailer = detailers.Filter_nexp(filtername='u', nexp=1)
    dither_detailer = detailers.Dither_detailer(per_night=per_night, max_dither=max_dither)
    details = [detailers.Camera_rot_detailer(min_rot=-camera_ddf_rot_limit, max_rot=camera_ddf_rot_limit),
               dither_detailer, u_detailer, detailers.Rottep2Rotsp_desired_detailer()]
    euclid_detailers = [detailers.Camera_rot_detailer(min_rot=-camera_ddf_rot_limit, max_rot=camera_ddf_rot_limit),
                        detailers.Euclid_dither_detailer(), u_detailer, detailers.Rottep2Rotsp_desired_detailer()]
    ddfs = ddf_surveys(detailers=details, season_frac=ddf_season_frac, euclid_detailers=euclid_detailers)

    greedy = gen_greedy_surveys(nside, nexp=nexp, footprints=footprints)

    # Set up the damn ToO kwargs
    times = [0, 1, 2, 4, 24]
    filters_at_times = [too_filters]*4 + ['gz']
    nvis = [1, 1, 1, 1, 6]

    camera_rot_limits = [-80., 80.]
    detailer_list = []
    detailer_list.append(detailers.Camera_rot_detailer(min_rot=np.min(camera_rot_limits),
                                                       max_rot=np.max(camera_rot_limits)))
    detailer_list.append(detailers.Rottep2Rotsp_desired_detailer())
    toos = [ToO_scripted_survey([], nside=nside, followup_footprint=too_footprint,
                                times=times[0:too_nfollow],
                                filters_at_times=filters_at_times[0:too_nfollow],
                                nvis=nvis[0:too_nfollow], detailers=detailer_list)]

    blobs = generate_blobs(nside, nexp=nexp, footprints=footprints, mjd_start=conditions.mjd_start,
                           good_seeing_weight=gsw)
    twi_blobs = generate_twi_blobs(nside, nexp=nexp,
                                   footprints=footprints,
                                   wfd_footprint=wfd_footprint,
                                   repeat_night_weight=repeat_night_weight)
    surveys = [toos, ddfs, blobs, twi_blobs, greedy]
    run_sched(surveys, observatory, survey_length=survey_length, verbose=verbose,
              fileroot=os.path.join(outDir, fileroot+file_end), extra_info=extra_info,
              nside=nside, illum_limit=illum_limit, event_table=event_table)
