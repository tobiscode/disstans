# The DISSTANS Python package

Welcome to the package repository for the **D**ecomposition and **I**nference of
**S**ources through **S**patio**t**emporal **A**nalysis of **N**etwork **S**ignals
(DISSTANS) toolbox.

With DISSTANS, you can:

- Fit GNSS displacement (or any type of) timeseries using a variety of functional
  models. These range from common ones such as a polynomial, a sinusoid, or a
  logarithm, to more complex ones as modulated sinusoids or dictionary of
  splines.
- Solve for model parameters using least squares with no regularization or using
  any of the L2, L1, and L0 norms. Spatial L0 allows to use expected spatial coherence
  in the data to improve local fits.
- Take advantage of multiprocessor systems using multiple threads for the most
  computationally heavy steps.
- Perform PCA/ICA-based common mode estimation and timeseries basis decomposition.
- Clean data from outliers.
- Visualize timeseries, network maps, dictionary scalograms, station motions,
  model parameter correlations, and more.
- Manage databases of raw RINEX files, including availability plots.
- Download GNSS timeseries from public sites.
- Use catalogued seismic and maintenance events to inform the model setup.
- Detect jumps in the data using a simple step detector.
- Run the [MIDAS](https://doi.org/10.1002/2015JB012552) algorithm.
- Generate synthetic timeseries.
- Load timeseries in JPL's `.tseries` or UNR's `.tenv3` formats natively,
  or load standard NumPy and pandas data.

All from within your Python shell, and everything in standard Python object-oriented
programming style, allowing you to easily subclass existing code to suit your individual
needs.

## Documentation

DISSTANS contains full code annotation, an API documentation, as well as tutorials
and real-data examples that show the usage of the package.

The documentation can be found in the `docs/` folder. It is hosted on Github publicly
at [tobiscode.github.io/disstans](https://tobiscode.github.io/disstans), but you can
also read it locally, e.g., by running `python -m http.server 8080 --bind 127.0.0.1`
from with the documentation folder and then opening a browser.

## Installation

The full installation instructions, including necessary prerequisites, can be found
[in the documentation](https://tobiscode.github.io/disstans/installation.html).
The short answer is:

``` bash
# clone repository to disstans/
git clone https://github.com/tobiscode/disstans.git
# change into folder
cd disstans/
# create the conda environment using the specification file,
# installing all dependencies along the way
conda env create -f environment.yml
# activate the environment
conda activate disstans
# install the package into the environment
pip install .
```

## Using and citing this work

If you're using this code or any parts of it, please cite the following study:

  T. Köhne, B. Riel, M. Simons (in preparation).
  *Decomposition and Inference of Sources through Spatiotemporal Analysis of*
  *Network Signals: The DISSTANS Python Package*

Please also note that this work is under a GPL-3.0 License.

## Acknowledgments

This code would not be possible without the work of others, such as:

- The inspiration for this code, [pygeodesy](https://github.com/bryanvriel/pygeodesy)
  by Bryan V. Riel
- The [MIDAS code](http://geodesy.unr.edu/MIDAS_release.tar) by Geoff Blewitt
- The powerlaw noise generation code
  [colorednoise](https://github.com/felixpatzelt/colorednoise) by Felix Patzelt
- The wrapper for the Okada elastic dislocation model
  [okada_wrapper](https://github.com/tbenthompson/okada_wrapper/) by Thomas Ben Thompson

## Reporting bugs and getting involved

If you find a bug or have a question about the code, please raise an issue on Github.
If you have any other comment, feedback, or suggestion, feel free to send me an email
to [tkoehne@caltech.edu](mailto:tkoehne@caltech.edu).
Similarly, if you want to contribute to any part of the code (functions, classes,
documentation, examples, etc.), please send me an email - contributions of all kinds
are always welcome!
