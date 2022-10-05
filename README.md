[![DOI](https://zenodo.org/badge/488745027.svg)](https://zenodo.org/badge/latestdoi/488745027)


# sims_featureScheduler_runs2.2
Even more survey strategy simulations


Baseline:  Yet another baseline

carina: re-run of the carina sim because it seemed to drop ~half the observations last time. Might have just been a bug not updating survey start date.

ddf_season_length: Try varying the season length of DDFs. Values are set to extremes, so expect DDF sequences to break down in some eimulations. Note the final DDF scheduling should probably have different season lengths for different fields.

noroll: Like the baseline, but no rolling cadence.

twilight_neo:  Running an NEO survey in twilight time. Also have twi_neo_brightest where only the brightest part of twilight is used for the NEO survey. (there are some minor differences between twilight_neo and the baseline, so compare to no_twilight_neo).

no_twilight_neo: Like the twilight_neo, but regular scheduling. 

