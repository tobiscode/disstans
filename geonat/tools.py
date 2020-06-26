import os
import pandas as pd
from multiprocessing import Pool

# set default number of threads to use
from . import defaults
defaults["general"]["num_threads"] = int(len(os.sched_getaffinity(0)) // 2)


def tvec_to_numpycol(timevector, t_reference=None, time_unit='D'):
    # get reference time
    if t_reference is None:
        t_reference = timevector[0]
    else:
        t_reference = pd.Timestamp(t_reference)
    assert isinstance(t_reference, pd.Timestamp), f"'t_reference' must be a pandas.Timestamp object, got {type(t_reference)}."
    # return Numpy array
    return ((timevector - t_reference) / pd.to_timedelta(1, time_unit)).values


def parallelize(func, iterable, num_threads=None, chunksize=1):
    if num_threads is None:
        num_threads = defaults["general"]["num_threads"]
    if num_threads > 0:
        with Pool(num_threads) as p:
            for result in p.imap(func, iterable, chunksize):
                yield result
    else:
        for parameter in iterable:
            yield func(parameter)
