
The version 2.2 baseline

* Now using pre-computed DDF scheduling.
* including basis function to supress additional repeat observations in a night
* Added detailer to be sure we flush the queue for scheduled observations

* XXX--need to fix bug where long blobs are not respecting morning twilight.


Note:  Need to run `generate_grid.py` to pre-compute airmasses and depths for each DDF. Takes a while, but only needs to be done once.

