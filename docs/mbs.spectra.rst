The mbs.spectra module
======================

.. testsetup:: *

   from mbs.spectra import Spectrum

The mbs module simplifies dealing with data from MB Scientific photoelectron analyzers.

.. doctest::

    To load a file, you can do for example
    >>> s = Spectrum.from_filename("../example_data/190731-MBS-00116_00000.txt")

    Metadata is accessible as a `dict` under the `.metadata` attribute:
    >>> s.metadata['Pass Energy']
    'PE200'