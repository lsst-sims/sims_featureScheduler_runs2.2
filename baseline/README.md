
The version 2.2 baseline

* Now using pre-computed DDF scheduling
* Including basis function to supress additional repeat observations in a night
* Added detailer to be sure we flush the queue for scheduled observations
* Turn off the ability of pair times to scale (was causing run-over into twilight)
* (probably) updating to newer sky brightness files (might cause small drop in u-band coverage)
* added increased weight on the u-band template basis function to ensure coverage in first year.


Note:  Need to run `generate_grid.py` to pre-compute airmasses and depths for each DDF. Takes a while, but only needs to be done once.

