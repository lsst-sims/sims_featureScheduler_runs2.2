import numpy as np
import matplotlib.pylab as plt
from rubin_sim.utils import calcSeason, ddf_locations
from rubin_sim.scheduler.utils import scheduled_observation
import os
import argparse


def match_cumulative(cumulative_desired, mask=None, no_duplicate=True):
    """Generate a schedule that tries to match the desired cumulative distribution given a mask

    Parameters
    ----------
    cumulative_desired : `np.array`, float
        An array with the cumulative number of desired observations. Elements
        are assumed to be evenly spaced. 
    mask : `np.array`, bool or int (None)
        Set to zero for indices that cannot be scheduled
    no_duplicate : bool (True)
        If True, only 1 event can be scheduled per element

    Returns
    -------
    schedule : `np.array`
        The resulting schedule, with values marking number of events in that cell.
    """

    rounded_desired = np.round(cumulative_desired)
    sched = cumulative_desired*0
    if mask is None:
        mask = np.ones(sched.size)

    valid = np.where(mask > 0)[0].tolist()
    x = np.arange(sched.size)

    drd = np.diff(rounded_desired)
    step_points = np.where(drd > 0)[0] + 1

    # would be nice to eliminate this loop, but it's not too bad.
    # can't just use searchsorted on the whole array, because then there
    # can be duplicate values, and array[[n,n]] = 1 means that extra match gets lost.
    for indx in step_points:
        left = np.searchsorted(x[valid], indx)
        right = np.searchsorted(x[valid], indx, side='right')
        d1 = indx - left
        d2 = right - indx
        if d1 < d2:
            sched_at = left
        else:
            sched_at = right

        # If we are off the end
        if sched_at >= len(valid):
            sched_at -= 1
        
        sched[valid[sched_at]] += 1
        if no_duplicate:
            valid.pop(sched_at)

    return sched


def optimize_ddf_times(ddf_name, ddf_RA, ddf_grid,
                       sun_limit=-18, airmass_limit=2.5, sky_limit=None,
                       g_depth_limit=23.5, sequence_limit=258, season_frac=0.1):
    """Run gyrobi to optimize the times of a ddf

    Parameters
    ----------
    ddf : `str`
        The name of the DDF
    ddf_grid : `np.array`
        An array with info for the DDFs. Generated by the `generate_grid.py` script
    season_frac : `float`
        7.2 month observing season if season_frac = 0.2 (shaves 20% off each end of the full year)
    """
    sun_limit = np.radians(sun_limit)

    # XXX-- double check that I got this right
    ack = ddf_grid['sun_alt'][0:-1] * ddf_grid['sun_alt'][1:]
    night = np.zeros(ddf_grid.size, dtype=int)
    night[np.where((ddf_grid['sun_alt'][1:] >= 0) & (ack < 0))] += 1
    night = np.cumsum(night)
    ngrid = ddf_grid['mjd'].size

    # set a sun, airmass, sky masks
    sun_mask = np.ones(ngrid, dtype=int)
    sun_mask[np.where(ddf_grid['sun_alt'] >= sun_limit)] = 0

    airmass_mask = np.ones(ngrid, dtype=int)
    airmass_mask[np.where(ddf_grid['%s_airmass' % ddf_name] >= airmass_limit)] = 0

    sky_mask = np.ones(ngrid, dtype=int)
    if sky_limit is not None:
        sky_mask[np.where(ddf_grid['%s_sky_g' % ddf_name] <= sky_limit)] = 0
        sky_mask[np.where(np.isnan(ddf_grid['%s_sky_g' % ddf_name]) == True)] = 0

    m5_mask = np.zeros(ngrid, dtype=bool)
    m5_mask[np.isfinite(ddf_grid['%s_m5_g' % ddf_name])] = 1

    if g_depth_limit is not None:
        m5_mask[np.where(ddf_grid['%s_m5_g' % ddf_name] < g_depth_limit)] = 0

    big_mask = sun_mask * airmass_mask * sky_mask * m5_mask

    potential_nights = np.unique(night[np.where(big_mask > 0)])

    # prevent a repeat sequence in a night
    unights, indx = np.unique(night, return_index=True)
    night_mjd = ddf_grid['mjd'][indx]
    # The season of each night
    night_season = calcSeason(ddf_RA, night_mjd)

    raw_obs = np.ones(unights.size)
    # take out the ones that are out of season
    season_mod = night_season % 1

    out_season = np.where((season_mod < season_frac) | (season_mod > (1.-season_frac)))
    raw_obs[out_season] = 0
    cumulative_desired = np.cumsum(raw_obs)
    cumulative_desired = cumulative_desired/cumulative_desired.max()*sequence_limit

    night_mask = unights*0
    night_mask[potential_nights] = 1

    unight_sched = match_cumulative(cumulative_desired, mask=night_mask)
    cumulative_sched = np.cumsum(unight_sched)

    nights_to_use = unights[np.where(unight_sched == 1)]
    
    # For each night, find the best time in the night. 
    # XXX--probably need to expand this part to resolve the times when multiple things get scheduled
    mjds = []
    for night_check in nights_to_use:
        in_night = np.where((night == night_check) & (np.isfinite(ddf_grid['%s_m5_g' % ddf_name])))[0]
        m5s = ddf_grid['%s_m5_g' % ddf_name][in_night]
        # we could intorpolate this to get even better than 15 min resolution on when to observe
        max_indx = np.where(m5s == m5s.max())[0].min()
        mjds.append(ddf_grid['mjd'][in_night[max_indx]])

    return mjds, night_mjd, cumulative_desired, cumulative_sched


def generate_ddf_scheduled_obs(data_file='ddf_grid.npz', flush_length=2, mjd_tol=15, expt=30.,
                               alt_min=25, alt_max=85, HA_min=21., HA_max=3.,
                               dist_tol=3., season_frac=0.1,
                               low_season_frac=0.4, low_season_rate=0.3,
                               nvis_master=[8, 10, 20, 20, 26, 20], filters='ugrizy',
                               nsnaps=[1, 2, 2, 2, 2, 2], sequence_limit=258):

    flush_length = flush_length  # days
    mjd_tol = mjd_tol/60/24.  # minutes to days
    expt = expt
    alt_min = np.radians(alt_min)
    alt_max = np.radians(alt_max)
    dist_tol = np.radians(dist_tol)

    ddfs = ddf_locations()
    ddf_data = np.load(data_file)
    ddf_grid = ddf_data['ddf_grid'].copy()
    
    all_scheduled_obs = []
    for ddf_name in ['XMM_LSS']:
        print('Optimizing %s' % ddf_name)

        # 'ID', 'RA', 'dec', 'mjd', 'flush_by_mjd', 'exptime', 'filter', 'rotSkyPos', 'nexp',
        #         'note'
        # 'mjd_tol', 'dist_tol', 'alt_min', 'alt_max', 'HA_max', 'HA_min', 'observed'
        mjds = optimize_ddf_times(ddf_name, ddfs[ddf_name][0], ddf_grid,
                                  season_frac=season_frac,
                                  sequence_limit=sequence_limit)[0]
        for mjd in mjds:
            for filtername, nvis, nexp in zip(filters, nvis_master, nsnaps):
                if 'EDFS' in ddf_name:
                    obs = scheduled_observation(n=int(nvis/2))
                    obs['RA'] = np.radians(ddfs[ddf_name][0])
                    obs['dec'] = np.radians(ddfs[ddf_name][1])
                    obs['mjd'] = mjd
                    obs['flush_by_mjd'] = mjd + flush_length
                    obs['exptime'] = expt
                    obs['filter'] = filtername
                    obs['nexp'] = nexp
                    obs['note'] = 'DD:%s' % ddf_name

                    obs['mjd_tol'] = mjd_tol
                    obs['dist_tol'] = dist_tol
                    # Need to set something for HA limits
                    obs['HA_min'] = HA_min
                    obs['HA_max'] = HA_max
                    obs['alt_min'] = alt_min
                    obs['alt_max'] = alt_max
                    all_scheduled_obs.append(obs)

                    obs = scheduled_observation(n=int(nvis/2))
                    obs['RA'] = np.radians(ddfs[ddf_name.replace('_a', '_b')][0])
                    obs['dec'] = np.radians(ddfs[ddf_name.replace('_a', '_b')][1])
                    obs['mjd'] = mjd
                    obs['flush_by_mjd'] = mjd + flush_length
                    obs['exptime'] = expt
                    obs['filter'] = filtername
                    obs['nexp'] = nexp
                    obs['note'] = 'DD:%s' % ddf_name.replace('_a', '_b')

                    obs['mjd_tol'] = mjd_tol
                    obs['dist_tol'] = dist_tol
                    # Need to set something for HA limits
                    obs['HA_min'] = HA_min
                    obs['HA_max'] = HA_max
                    obs['alt_min'] = alt_min
                    obs['alt_max'] = alt_max
                    all_scheduled_obs.append(obs)

                else:

                    obs = scheduled_observation(n=nvis)
                    obs['RA'] = np.radians(ddfs[ddf_name][0])
                    obs['dec'] = np.radians(ddfs[ddf_name][1])
                    obs['mjd'] = mjd
                    obs['flush_by_mjd'] = mjd + flush_length
                    obs['exptime'] = expt
                    obs['filter'] = filtername
                    obs['nexp'] = nexp
                    obs['note'] = 'DD:%s' % ddf_name

                    obs['mjd_tol'] = mjd_tol
                    obs['dist_tol'] = dist_tol
                    # Need to set something for HA limits
                    obs['HA_min'] = HA_min
                    obs['HA_max'] = HA_max
                    obs['alt_min'] = alt_min
                    obs['alt_max'] = alt_max
                    all_scheduled_obs.append(obs)

    result = np.concatenate(all_scheduled_obs)
    result['scripted_id'] = np.arange(result.size)
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", type=str, default='ddf.npz')
    parser.add_argument("--season_frac", type=float, default=0.1)
    parser.add_argument("--plot_dir", type=str, default='ddf_plots')
    args = parser.parse_args()
    filename = args.out_file
    season_frac = args.season_frac
    plot_dir = args.plot_dir

    if (plot_dir is not None) & (plot_dir != 'None'):
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

    obs_array = generate_ddf_scheduled_obs(plot_dir=plot_dir, season_frac=season_frac)
    np.savez(filename, obs_array=obs_array)