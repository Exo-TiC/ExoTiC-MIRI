Custom step descriptions
========================

Here we list each of the available custom steps, and include a
description of the algorithms and arguments.

Drop integrations
-----------------

This steps allows the user to drop integrations from a data chunk,
most likely because these groups are too severely affected by
systematics to be worth processing. This step may also be useful
if the user wants to test pipelines on only a small subset of data.

drop_integrations = int_list(default=None)  # integrations to drop, zero-indexed.

Drop groups
-----------

This steps allows the user to drop groups from each integration which
may adversely affect the ramps.

drop_groups = int_list(default=None)  # groups to drop, zero-indexed.

Regroup
-------

This steps allows the user to regroup your integrations, comprised
of n groups, into several smaller integrations, comprised of m
groups, where n is a multiple of m.

n_groups = integer(default=10)  # new number of groups per integration

Reference pixel
---------------

This steps allows the user to apply corrections to their group-
level images, using the reference pixels, for the MIRI LRS
subarray. The corrections can be made with a variety of options
for smoothing the values and/or separating odd and even rows.

smoothing_length = integer(default=None)  # median smooth values over pixel length
odd_even_rows = boolean(default=True)  # treat and odd and even rows separately

Extract 1d
----------

A step that will sky subtract, trace find, and extract 1d spectra
from the 2d rate images using various algorithms. This step assumes
the photom step has not been run, or at least input data units are
DN/s.

bkg_algo = option("constant", "polynomial", default="polynomial")  # background algorithm
bkg_region = int_list(default=None)  # background region, start, stop, start, stop
bkg_poly_order = integer(default=1)  # order of polynomial for background fitting
bkg_smoothing_length = integer(default=None)  # median smooth values over pixel length
extract_algo = option("box", "optimal", "anchor", default="box")  # extraction algorithm
extract_region_width = integer(default=20)  # full width of extraction region
extract_poly_order = integer(default=1)  # order of polynomial for optimal extraction
max_iter = integer(default=10)  # max iterations of anchor algorithm
