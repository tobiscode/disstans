"""
This module contains the :class:`~Timeseries` base class and other
formats included by default in GeoNAT.
"""

import numpy as np
import pandas as pd
from warnings import warn
from copy import deepcopy


class Timeseries():
    """
    Object that expands the functionality of a :class:`~pandas.DataFrame` object
    for better integration into GeoNAT. Apart from the data itself, it contains
    information about the source and units of the data. It also performs input
    checks and uses property setters/getters to ensure consistency.

    Also enables the ability to perform math on timeseries directly.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The timeseries' data as a DataFrame. The index should be time, whereas
        data columns can be both data and their uncertainties.
    src : str
        Source description.
    data_unit : str
        Data unit.
    data_cols : list
        List of strings with the names of the columns of ``dataframe`` that
        contain the data.
        The length cooresponds to the number of components :attr:`~num_components`.
    var_cols : list, optional
        List of strings with the names of the columns of ``dataframe`` that contain the
        data's variance.
        Must have the same length as ``data_cols``.
        Defaults to no data variance columns.
    cov_cols : list, optional
        List of strings with the names of the columns of ``dataframe`` that contain
        the data's covariance.
        Must have length ``(num_components * (num_components - 1)) / 2``, where
        the order of the elements is determined by their row-by-row, sequential
        position in the covariance matrix (see Notes).
        Defaults to no covariance columns.

    Notes
    -----

    In terms of mapping the covariance matrix of observations into the format
    for the :class:`~Timeseries` class, consider this example for observations with
    three components:

    +-------------------+-------------------+-------------------+
    | ``var_cols[0]``   | ``cov_cols[0]``   | ``cov_cols[1]``   |
    +-------------------+-------------------+-------------------+
    | (symmetric)       | ``var_cols[1]``   | ``cov_cols[2]``   |
    +-------------------+-------------------+-------------------+
    | (symmetric)       | (symmetric)       | ``var_cols[2]``   |
    +-------------------+-------------------+-------------------+
    """
    def __init__(self, dataframe, src, data_unit, data_cols,
                 var_cols=None, cov_cols=None):
        assert isinstance(dataframe, pd.DataFrame)
        assert isinstance(src, str)
        assert isinstance(data_unit, str)
        assert isinstance(data_cols, list) and all([isinstance(dcol, str) for dcol in data_cols])
        assert all([dcol in dataframe.columns for dcol in data_cols])
        self._df = dataframe
        self._src = src
        self._data_unit = data_unit
        self._data_cols = data_cols
        if var_cols is not None:
            assert isinstance(var_cols, list) and \
                all([isinstance(vcol, str) for vcol in var_cols]), \
                f"If not None, 'var_cols' must be a list of strings, got {var_cols}."
            assert all([vcol in dataframe.columns for vcol in var_cols]), \
                "All entries in 'var_cols' must be present in the dataframe."
            assert len(self._data_cols) == len(var_cols), \
                "If passing variance columns, " \
                "the list needs to have the same length as the data columns one."
        self._var_cols = var_cols
        if cov_cols is not None:
            assert self.num_components > 1, \
                "In order to set covariances, the timeseries needs at least 2 components."
            assert cov_cols is not None, \
                "If setting covariances, the standard deviations must also be set."
            assert isinstance(cov_cols, list) and \
                all([isinstance(ccol, str) for ccol in cov_cols]), \
                f"If not None, 'cov_cols' must be a list of strings, got {cov_cols}."
            assert all([ccol in dataframe.columns for ccol in cov_cols]), \
                "All entries in 'cov_cols' must be present in the dataframe."
            cov_dims = int((self.num_components * (self.num_components - 1)) / 2)
            assert len(cov_cols) == cov_dims, \
                "If passing covariance columns, the list needs to have the appropriate " \
                "length given the data columns length. " \
                f"Expected {cov_dims}, got {len(cov_cols)}."
            self._make_index_map()
        self._cov_cols = cov_cols

    def __repr__(self):
        """
        Special function that returns a readable summary of the timeseries.
        Accessed, for example, by Python's ``print()`` built-in function.

        Returns
        -------
        info : str
            Timeseries summary.
        """
        info = f"Timeseries\n - Source: {self.src}\n - Units: {self.data_unit}" \
               f"\n - Shape: {self.shape}\n - Data: {[key for key in self.data_cols]}"
        if self.var_cols is not None:
            info += f"\n - Variances: {[key for key in self.var_cols]}"
        if self.cov_cols is not None:
            info += f"\n - Covariances: {[key for key in self.cov_cols]}"
        return info

    def __getitem__(self, columns):
        """
        Convenience special function that provides a shorthand notation
        to access the timeseries' columns.

        Parameters
        ----------
        columns : str, list
            String or list of strings of the columns to return.

        Returns
        -------
        pandas.Series, pandas.DataFrame
            Returns the requested data as a Series (if a single column) or
            DataFrame (if multiple columns).

        Example
        -------
        If ``ts`` is a :class:`~Timeseries` instance and ``columns`` a list
        of column names, the following two are equivalent::

            ts.df[columns]
            ts[ts_description]
        """
        if not isinstance(columns, str) and \
           (not isinstance(columns, list) or
           not all([isinstance(col, str) for col in columns])):
            raise KeyError("Error when accessing data in timeseries: 'column' must be a "
                           f"string or list of strings, given was {columns}.")
        return self.df[columns]

    @property
    def src(self):
        """ Source information. """
        return self._src

    @src.setter
    def src(self, new_src):
        if not isinstance(new_src, str):
            raise TypeError(f"New 'src' attribute has to be a string, got {type(new_src)}.")
        self._src = new_src

    @property
    def data_unit(self):
        """ Data unit. """
        return self._data_unit

    @data_unit.setter
    def data_unit(self, new_data_unit):
        if not isinstance(new_data_unit, str):
            raise TypeError("New 'data_unit' attribute has to be a string, "
                            f"got {type(new_data_unit)}.")
        self._data_unit = new_data_unit

    @property
    def data_cols(self):
        """ List of the column names in :attr:`~df` that contain data. """
        return self._data_cols

    @data_cols.setter
    def data_cols(self, new_data_cols):
        assert isinstance(new_data_cols, list) and \
            all([isinstance(dcol, str) for dcol in new_data_cols]), \
            "New 'data_cols' attribute must be a list of strings of the same length as the " \
            f"current 'data_cols' ({len(self._data_cols)}), got {new_data_cols}."
        self._df.rename(columns={old_col: new_col for old_col, new_col
                                 in zip(self._data_cols, new_data_cols)},
                        errors='raise', inplace=True)
        self._data_cols = new_data_cols

    @property
    def var_cols(self):
        """ List of the column names in :attr:`~df` that contain data variance. """
        return self._var_cols

    @var_cols.setter
    def var_cols(self, new_var_cols):
        assert self._var_cols is not None, \
            "No variance columns found that can be renamed."
        assert isinstance(new_var_cols, list) and \
            all([(vcol is None) or isinstance(vcol, str) for vcol in new_var_cols]), \
            "New 'var_cols' attribute must be a list of strings or Nones of the same length " \
            f"as the current 'var_cols' ({len(self._var_cols)}), got {new_var_cols}."
        for ivcol, vcol in enumerate(new_var_cols):
            if vcol is None:
                new_var_cols[ivcol] = self._var_cols[ivcol]
        self._df.rename(columns={old_col: new_col for old_col, new_col
                                 in zip(self._var_cols, new_var_cols)},
                        errors='raise')
        self._var_cols = new_var_cols

    @property
    def cov_cols(self):
        """ List of the column names in :attr:`~df` that contain data covariances. """
        return self._cov_cols

    @cov_cols.setter
    def cov_cols(self, new_cov_cols):
        assert self._cov_cols is not None, \
            "No covariance columns found that can be renamed."
        assert isinstance(new_cov_cols, list) and \
            all([(ccol is None) or isinstance(ccol, str) for ccol in new_cov_cols]), \
            "New 'cov_cols' attribute must be a list of strings or Nones of the same length " \
            f"as the current 'cov_cols' ({len(self._cov_cols)}), got {new_cov_cols}."
        for iccol, ccol in enumerate(new_cov_cols):
            if ccol is None:
                new_cov_cols[iccol] = self._cov_cols[iccol]
        self._df.rename(columns={old_col: new_col for old_col, new_col
                                 in zip(self._cov_cols, new_cov_cols)},
                        errors='raise')
        self._cov_cols = new_cov_cols

    @property
    def num_components(self):
        """ Number of data columns. """
        return len(self._data_cols)

    @property
    def df(self):
        """ The entire timeseries' DataFrame. """
        return self._df

    @property
    def num_observations(self):
        """ Number of observations (rows in :attr:`~df`). """
        return self._df.shape[0]

    @property
    def data(self):
        """ View of only the data columns in :attr:`~df`. """
        return self._df.loc[:, self._data_cols]

    @data.setter
    def data(self, new_data):
        self._df.loc[:, self._data_cols] = new_data

    @property
    def sigmas(self):
        """ View of only the data standard deviation columns in :attr:`~df`. """
        return self.vars ** (1/2)

    @sigmas.setter
    def sigmas(self, new_sigma):
        self.vars = new_sigma ** 2

    @property
    def vars(self):
        """ Returns the variances from :attr:`~df`. """
        if self._var_cols is None:
            raise ValueError("No variance columns present to return.")
        return self._df.loc[:, self._var_cols]

    @vars.setter
    def vars(self, new_var):
        if self._var_cols is None:
            raise ValueError("No variance columns present to set.")
        self.df.loc[:, self._var_cols] = new_var

    @property
    def covs(self):
        """ Returns the covariances from :attr:`~df`. """
        if self._cov_cols is None:
            raise ValueError("No covariance columns present to return.")
        return self._df.loc[:, self._cov_cols]

    @covs.setter
    def covs(self, new_cov):
        if self._cov_cols is None:
            raise ValueError("No covariance columns present to set.")
        self.df.loc[:, self._cov_cols] = new_cov

    @property
    def var_cov(self):
        """ Retuns the variance as well as covariance columns from :attr:`~df`. """
        if self._var_cols is None:
            raise ValueError("No variance columns present to return.")
        if self._cov_cols is None:
            raise ValueError("No covariance columns present to set.")
        return self._df.loc[:, self._var_cols + self._cov_cols]

    @var_cov.setter
    def var_cov(self, new_var_cov):
        assert new_var_cov.shape[1] == self.num_components * 2, \
            "Setting 'var_cov' requires a column for each variance (first half of array) " \
            "and covariance (second half)."
        self.vars = new_var_cov[:, :self.num_components]
        self.covs = new_var_cov[:, self.num_components:]

    @property
    def time(self):
        """ Timestamps of the timeseries (index of :attr:`~df`). """
        return self._df.index

    @property
    def shape(self):
        r"""
        Returns the shape tuple (similar to NumPy) of the timeseries, which is of shape
        :math:`(\text{n_observations},\text{n_components})`.
        """
        return (self.num_observations, self.num_components)

    def add_uncertainties(self, timeseries=None,
                          var_data=None, var_cols=None, cov_data=None, cov_cols=None):
        """
        Add variance and covariance data and column names to the timeseries.

        Parameters
        ----------
        timeseries : geonat.timeseries.Timeseries
            Another timeseries object that contains uncertainty information.
            If set, the function will ignore the rest of the arguments.
        var_data : numpy.ndarray, optional
            New data variance.
        var_cols : list, optional
            List of variance column names.
        cov_data : numpy.ndarray, optional
            New data covariance.
            Setting this but not ``var_data`` requires there to already be data variance.
        cov_cols : list, optional
            List of covariance column names.

        Notes
        -----
        If ``ts`` is a :class:`~geonat.timeseries.Timeseries` instance, just using::

            ts.vars = new_variance
            ts.covs = new_covariance

        will only work when the respective columns already exist in the dataframe.
        (This is them same behavior for renaming variance columns that do not exist.)
        If they do not exist, the calls will results in an error because no column names exist,
        in an effort to make the inner workings more transparent and rigorous.

        This function allows to override the default behavior, and can also generate column
        names by itself if none are specified.
        """
        # check if input is a Timeseries, override rest of input arguments
        if timeseries is not None:
            if not isinstance(timeseries, Timeseries):
                raise TypeError("Cannot use uncertainty data: "
                                "'timeseries' is not a Timeseries object.")
            # get variance data and description
            if timeseries.var_cols is not None:
                var_data = timeseries.vars[timeseries.time.isin(self.time)].values
                var_cols = timeseries.var_cols
            else:
                var_data = None
                var_cols = None
            # get covariance data and description
            if timeseries.cov_cols is not None:
                cov_data = timeseries.covs[timeseries.time.isin(self.time)].values
                cov_cols = timeseries.cov_cols
            else:
                cov_data = None
                cov_cols = None
        # add variance
        if var_data is not None:
            assert isinstance(var_data, np.ndarray) and var_data.shape == self.shape, \
                   "'var_data' must be a NumPy array of the same shape as the data " \
                   f"{self.shape}, got " \
                   f"{var_data.shape if isinstance(var_data, np.ndarray) else var_data}."
            if var_cols is None:
                var_cols = [dcol + "_sigma" for dcol in self.data_cols]
            else:
                assert (isinstance(var_cols, list) and len(var_cols) == self.num_components and
                        all([isinstance(vcol, str) for vcol in var_cols])), \
                       "New 'var_cols' attribute must be a list of strings of the same length " \
                       f"as the current 'data_cols' ({len(self._data_cols)}), got {var_cols}."
            for ivcol, vcol in enumerate(var_cols):
                self._df[vcol] = var_data[:, ivcol]
            self._var_cols = var_cols
        # add covariance, check for variance
        if cov_data is not None:
            assert self._var_cols is not None, \
                "Cannot set covariance data without first adding variance data."
            cov_dims = int((self.num_components * (self.num_components - 1)) / 2)
            assert (isinstance(cov_data, np.ndarray)
                    and cov_data.shape[0] == self.num_observations
                    and cov_data.shape[1] == cov_dims), \
                "'cov_data' must be a NumPy array of the same length as the data " \
                "and width corresponding to all possible covariances. Expected shape " \
                f"({self.num_observations}, {cov_dims}), got " \
                f"{cov_data.shape if isinstance(cov_data, np.ndarray) else cov_data}."
            if cov_cols is None:
                cov_cols = []
                for i1 in range(self.num_components):
                    for i2 in range(i1 + 1, self.num_components):
                        cov_cols.append(f"{self.data_cols[i1]}_{self.data_cols[i2]}_cov")
            else:
                assert (isinstance(cov_cols, list) and len(cov_cols) == cov_dims and
                        all([isinstance(vcol, str) for vcol in cov_cols])), \
                    "New 'cov_cols' attribute must be a list of strings of length " \
                    "corresponding to all possible covariances " \
                    f"( expected {cov_dims}, got {len(cov_cols)}."
            for iccol, ccol in enumerate(cov_cols):
                self._df[ccol] = cov_data[:, iccol]
            self._make_index_map()
            self._cov_cols = cov_cols

    def copy(self, only_data=False, src=None):
        """
        Return a deep copy of the timeseries instance.

        Parameters
        ----------
        only_data : bool, optional
            If ``True``, only copy the data columns and ignore any uncertainty information.
            Defaults to ``False``.
        src : str, optional
            Set a new source information attribute for the copy.
            Uses the current one by default.

        Returns
        -------
        Timeseries
            The copy of the timeseries instance.
        """
        new_name = deepcopy(self._src) if src is None else src
        if not only_data:
            return Timeseries(self._df.copy(), new_name, deepcopy(self._data_unit),
                              deepcopy(self._data_cols), deepcopy(self._var_cols),
                              deepcopy(self._cov_cols))
        else:
            return Timeseries(self._df[self._data_cols].copy(), new_name,
                              deepcopy(self._data_unit), deepcopy(self._data_cols), None, None)

    def mask_out(self, dcol):
        """
        Mask out an entire data column (and if present, its uncertainty column) by setting
        the entire column to ``NaN``. Converts it to a sparse representation to save memory.

        Parameters
        ----------
        dcol : str
            Name of the data column to mask out.
        """
        icomp = self._data_cols.index(dcol)
        self._df[dcol] = np.NaN
        self._df[dcol] = self._df[dcol].astype(pd.SparseDtype(dtype=float))
        if self._var_cols is not None:
            vcol = self._var_cols[icomp]
            self._df[vcol] = np.NaN
            self._df[vcol] = self._df[vcol].astype(pd.SparseDtype(dtype=float))
            if self._cov_cols is not None:
                for iccol in self._get_cov_indices(icomp):
                    ccol = self._cov_cols[iccol]
                    self._df[ccol] = np.NaN
                    self._df[ccol] = self._df[ccol].astype(pd.SparseDtype(dtype=float))

    def _make_index_map(self):
        index_map = np.zeros((self.num_components, self.num_components))
        index_map[:] = np.NaN
        seq_ix = 0
        for irow in range(self.num_components):
            for icol in range(irow + 1, self.num_components):
                index_map[irow, icol] = seq_ix
                seq_ix += 1
        var_cov_map = (np.triu(index_map + self.num_components, 1) +
                       np.triu(index_map + self.num_components, 1).T)
        self.index_map = index_map
        self.var_cov_map = (var_cov_map + np.diag(np.arange(self.num_components))
                            ).astype(int).ravel()

    def _get_cov_indices(self, icomp):
        from_row = self.index_map[icomp, :]
        from_col = self.index_map[:, icomp]
        return [i for i in from_row if i != np.NaN] + [i for i in from_col if i != np.NaN]

    @staticmethod
    def prepare_math(left, right, operation):
        r"""
        Tests two timeseries' ability to be cast together in a mathematical operation,
        and returns output characteristics.
        Currently, only addition, subtraction, multiplication, and division are supported.

        All uncertainty information is lost during mathematical operations.

        One of the objects can be a NumPy array. In this case, the array has to have
        the exact same shape as the data in the Timeseries instance.
        Furthermore, the resulting Timeseries object will have the same :attr:`~src`,
        :attr:`~data_unit` and :attr:`~data_cols` attributes (instead of a combination
        of both).

        Warning
        -------
        This method is called under-the-hood whenever a mathematical operation is
        performed, and should not need to be used by normal users.

        Parameters
        ----------
        left : Timeseries or numpy.ndarray
            Left term of the operation.
        right : Timeseries or numpy.ndarray
            Right term of the operation.
        operation : str
            Operation to perform.
            Possible values are ``'+'``, ``'-'``, ``'*'`` and ``'/'``.

        Returns
        -------
        left_data : numpy.ndarray
            View of the 2D left data array of the operation
            with shape (``len(out_time)``, :attr:`~num_components`).
        right_data : numpy.ndarray
            View of the 2D right data array of the operation
            with shape (``len(out_time)``, :attr:`~num_components`).
        out_src : str
            Combines the sources of each object to a new string.
        out_data_unit : str
            Combines the data units of each object into a new unit.
        out_data_cols : list
            List of strings containing the new data column names.
        out_time : pandas.Index
            Index object containing the indices of all timestamps common to both.

        Raises
        ------
        TypeError
            If one of the operands is not a :class:`~Timeseries` or :class:`~numpy.ndarray`,
            or if both are :class:`~numpy.ndarray` (since then this function would never be
            called anyway).
        ValueError
            If the number of data columns is not equal between the two operands,
            or if the data units are not the same adding or subtracting.
        AssertionError
            If one of the operands is a NumPy array but does not have the same number of
            rows as the other operand.

        See Also
        --------
        __add__ : Addition for two Timeseries or a Timeseries and a NumPy array
        __radd__ : Addition for a NumPy array and a Timeseries.
        __sub__ : Subtraction for two Timeseries or a Timeseries and a NumPy array
        __rsub__ : Subtraction for a NumPy array and a Timeseries.
        __mul__ : Multiplication for two Timeseries or a Timeseries and a NumPy array
        __rmul__ : Multiplication for a NumPy array and a Timeseries.
        __truediv__ : Division for two Timeseries or a Timeseries and a NumPy array
        __rtruediv__ : Division for a NumPy array and a Timeseries.
        """
        # check for valid operator
        if operation not in ["+", "-", "*", "/"]:
            raise NotImplementedError("Timeseries math problem: "
                                      f"unknown operation '{operation}'.")
        # check for compatible types
        for operand, position in zip([left, right], ["Left", "Right"]):
            if (not isinstance(operand, Timeseries)) and (not isinstance(operand, np.ndarray)):
                raise TypeError(f"{position} operand has to be either a '{Timeseries}' or "
                                f"'{np.ndarray}' object, got {type(operand)}")
        if all([isinstance(operand, np.ndarray) for operand in [left, right]]):
            raise TypeError(f"At least one of the operands has to be a '{Timeseries}' object, "
                            f"got two '{np.ndarray}' objects.")
        # check for same dimensions
        if isinstance(left, Timeseries):
            len_left_data_cols = left.num_components
        else:
            len_left_data_cols = left.shape[1]
        if isinstance(right, Timeseries):
            len_right_data_cols = right.num_components
        else:
            len_right_data_cols = right.shape[1]
        if len_left_data_cols != len_right_data_cols:
            raise ValueError("Timeseries math problem: conflicting number of data columns "
                             f"({len_left_data_cols} and {len_right_data_cols}).")
        # get output attributes
        if isinstance(left, np.ndarray):
            assert left.shape[0] == right.num_observations, \
                "Timeseries math problem: conflicting number of observation " + \
                f"({left.shape[0]} and {right.num_observations})."
            out_time = right.time
            out_src = right.src
            out_data_unit = right.data_unit
            out_data_cols = right.data_cols
            left_data = left
            right_data = right.data.loc[out_time, :].values
        elif isinstance(right, np.ndarray):
            assert left.num_observations == right.shape[0], \
                "Timeseries math problem: conflicting number of observation " + \
                f"({left.num_observations} and {right.shape[0]})."
            out_time = left.time
            out_src = left.src
            out_data_unit = left.data_unit
            out_data_cols = left.data_cols
            left_data = left.data.loc[out_time, :].values
            right_data = right
        else:
            # time index is the intersection
            out_time = left.time.intersection(right.time, sort=None)
            # check compatible units (+-) or define new one (*/)
            # define new src and data_cols
            if operation in ["+", "-"]:
                if left.data_unit != right.data_unit:
                    raise ValueError("Timeseries math problem: conflicting data units "
                                     f"'{left.data_unit}' and '{right.data_unit}'.")
                else:
                    out_data_unit = left.data_unit
                    out_src = f"{left.src}{operation}{right.src}"
                    out_data_cols = [f"{lcol}{operation}{rcol}" for lcol, rcol
                                     in zip(left.data_cols, right.data_cols)]
            elif operation in ["*", "/"]:
                out_data_unit = f"({left.data_unit}){operation}({right.data_unit})"
                out_src = f"({left.src}){operation}({right.src})"
                out_data_cols = [f"({lcol}){operation}({rcol})" for lcol, rcol
                                 in zip(left.data_cols, right.data_cols)]
            left_data = left.data.loc[out_time, :].values
            right_data = right.data.loc[out_time, :].values
        # return data unit and column names
        return left_data, right_data, out_src, out_data_unit, out_data_cols, out_time

    def __add__(self, other):
        """
        Special function that allows two timeseries instances (or a timeseries and
        an equivalently shaped NumPy array) to be added together element-wise.

        Parameters
        ----------
        other : Timeseries
            Timeseries to add to instance.

        Returns
        -------
        Timeseries
            New timeseries object containing the sum of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it
            for more details about how the two objects are cast together.

        Example
        -------
        Add two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 + ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(self, other, '+')
        return Timeseries(pd.DataFrame(left_data + right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __radd__(self, other):
        """
        Reflected operation of :meth:`~__add__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(other, self, '+')
        return Timeseries(pd.DataFrame(left_data + right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __sub__(self, other):
        """
        Special function that allows a timeseries instance (or a timeseries and
        an equivalently shaped NumPy array) to be subtracted from another element-wise.

        Parameters
        ----------
        other : Timeseries
            Timeseries to subtract from instance.

        Returns
        -------
        Timeseries
            New timeseries object containing the difference of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it
            for more details about how the two objects are cast together.

        Example
        -------
        Subtract two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 - ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(self, other, '-')
        return Timeseries(pd.DataFrame(left_data - right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __rsub__(self, other):
        """
        Reflected operation of :meth:`~__sub__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(other, self, '-')
        return Timeseries(pd.DataFrame(left_data - right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __mul__(self, other):
        """
        Special function that allows two timeseries instances (or a timeseries and
        an equivalently shaped NumPy array) to be multiplied together element-wise.

        Parameters
        ----------
        other : Timeseries
            Timeseries to multiply to instance.

        Returns
        -------
        Timeseries
            New timeseries object containing the product of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it
            for more details about how the two objects are cast together.

        Example
        -------
        Multiply two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 * ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(self, other, '*')
        return Timeseries(pd.DataFrame(left_data * right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __rmul__(self, other):
        """
        Reflected operation of :meth:`~__mul__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(other, self, '*')
        return Timeseries(pd.DataFrame(left_data * right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __truediv__(self, other):
        """
        Special function that allows a timeseries instances (or a timeseries and
        an equivalently shaped NumPy array) to be divided by another element-wise.

        Parameters
        ----------
        other : Timeseries
            Timeseries to divide instance by.

        Returns
        -------
        Timeseries
            New timeseries object containing the quotient of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it
            for more details about how the two objects are cast together.

        Example
        -------
        Divide two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 / ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(self, other, '/')
        return Timeseries(pd.DataFrame(left_data / right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __rtruediv__(self, other):
        """
        Reflected operation of :meth:`~__truediv__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(other, self, '/')
        return Timeseries(pd.DataFrame(left_data / right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def get_arch(self):
        """
        Build a dictionary describing the architecture of this timeseries,
        to be used when creating a network JSON configuration file.

        Without subclassing :class:`~Timeseries`, this function will return an
        empty dictionary by default, since it is unknown how to recreate
        a general Timeseries object from just a JSON-compatible dictionary.

        See Also
        --------
        geonat.network.Network.to_json : Export the Network configuration as a JSON file.
        geonat.timeseries.Timeseries.get_arch
            Get the architecture dictionary of a :class:`~geonat.timeseries.Timeseries` instance.
        geonat.models.Model.get_arch
            Get the architecture dictionary of a :class:`~geonat.models.Model` instance.
        """
        return {}

    @classmethod
    def from_fit(cls, data_unit, data_cols, fit):
        """
        Import a fit dictionary and create a Timeseries instance.

        Parameters
        ----------
        data_unit : str
            Data unit.
        data_cols : list
            List of strings containing the data column names.
            Uncertainty column names are generated by adding a '_sigma'.
        fit : dict
            Dictionary with the keys ``'time'``, ``'fit'`` and ``'var'``.

        Returns
        -------
        Timeseries
            Timeseries instance created from ``fit``.

        See Also
        --------
        geonat.models.Model.evaluate
            Evaluating a model produces the fit dictionary.
        """
        df_data = {dcol: fit["fit"][:, icol] for icol, dcol in enumerate(data_cols)}
        var_cols = None
        if fit["var"] is not None:
            var_cols = [dcol + "_var" for dcol in data_cols]
            df_data.update({vcol: fit["var"][:, icol] for icol, vcol in enumerate(var_cols)})
        df = pd.DataFrame(data=df_data, index=fit["time"])
        return cls(df, "fitted", data_unit, data_cols, var_cols)

    @classmethod
    def from_array(cls, timevector, data, src, data_unit, data_cols,
                   var=None, var_cols=None, cov=None, cov_cols=None):
        r"""
        Constructor method to create a :class:`~Timeseries` instance from a NumPy
        :class:`~numpy.ndarray`.

        Parameters
        ----------
        timevector : pandas.Series, pandas.DatetimeIndex
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.
        data : numpy.ndarray
            2D NumPy array of shape :math:`(\text{n_observations},\text{n_components})`
            containing the data.
        src : str
            Source description.
        data_unit : str
            Data unit.
        data_cols : list
            List of strings with the names of the columns of ``data``.
        var : numpy.ndarray, optional
            2D NumPy array of shape :math:`(\text{n_observations},\text{n_components})`
            containing the data variances.
            Defaults to no data uncertainty.
        var_cols : list, optional
            List of strings with the names of the columns of ``data`` that contain the
            data's variance.
            Must have the same length as ``data_cols``.
            If ``var`` is given but ``var_cols`` is not, it defaults to appending
            ``'_var'`` to ``data_cols``.
        cov : numpy.ndarray, optional
            2D NumPy array of shape :math:`(\text{n_observations},\text{n_components})`
            containing the data covariances (as defined in :class:`~Timeseries`).
            Defaults to no data uncertainty.
        cov_cols : list, optional
            List of strings with the names of the columns of ``data`` that contain the
            data's covariance.
            Must have the same length as ``data_cols``.
            If ``cov`` is given but ``cov_cols`` is not, it defaults to appending
            ``'_cov'`` to the two respective entries of ``data_cols``.

        See Also
        --------
        :func:`~pandas.date_range` : Quick function to generate a timevector.
        """
        assert len(timevector) == data.shape[0], \
            "Length of 'timevector' has to match the number of rows in 'data', " \
            f"got {len(timevector)} and {data.shape}."
        assert isinstance(data_cols, list) and \
            all([isinstance(dcol, str) for dcol in data_cols]), \
            f"'data_cols' has to be a list of strings, got {data_cols}."
        # create dataframe with data
        df_data = {dcol: data[:, icol] for icol, dcol in enumerate(data_cols)}
        # add variances
        if var is None:
            var_cols = None
        else:
            assert data.shape == var.shape, \
                "'data' and 'var' need to have the same shape, got " \
                f"{data.shape} and {var.shape}."
            if var_cols is None:
                var_cols = [dcol + "_var" for dcol in data_cols]
            else:
                assert len(var_cols) == var.shape[1]
            df_data.update({vcol: var[:, icol] for icol, vcol in enumerate(var_cols)})
            # if variances are given, can also check for covariances
            if cov is None:
                cov_cols = None
            else:
                assert data.shape == cov.shape, \
                    "'data' and 'cov' need to have the same shape, got " \
                    f"{data.shape} and {cov.shape}."
                if cov_cols is None:
                    cov_cols = []
                    num_components = len(data_cols)
                    for i1 in range(num_components):
                        for i2 in range(i1 + 1, num_components):
                            cov_cols.append(f"{data_cols[i1]}_{data_cols[i2]}_cov")
                else:
                    assert len(cov_cols) == cov.shape[1]
                df_data.update({ccol: cov[:, icol] for icol, ccol in enumerate(cov_cols)})
        df = pd.DataFrame(data=df_data, index=timevector)
        return cls(df, src, data_unit, data_cols, var_cols, cov_cols)


class GipsyTimeseries(Timeseries):
    """
    Subclasses :class:`~Timeseries`.

    Timeseries subclass for GNSS measurements in JPL's Gipsy ``.tseries`` file format.
    The data and (co)variances are converted into millimeters (squared).

    Parameters
    ----------
    path : str
        Path to the timeseries file.
    show_warnings : bool, optional
        If ``True``, warn if there are data inconsistencies encountered while loading.
        Defaults to ``True``.

    Notes
    -----

    The column format is described on `JPL's website`_:

    +---------------+-------------------------------------------------+
    | Columns       | Description                                     |
    +===============+=================================================+
    | Column 1      | Decimal year computed with 365.25 days/yr       |
    +---------------+-------------------------------------------------+
    | Columns 2-4   | East, North and Vertical [m]                    |
    +---------------+-------------------------------------------------+
    | Columns 5-7   | East, North and Vertical standard deviation [m] |
    +---------------+-------------------------------------------------+
    | Columns 8-10  | East, North and Vertical covariance [m^2]       |
    +---------------+-------------------------------------------------+
    | Column 11     | Time in Seconds past J2000                      |
    +---------------+-------------------------------------------------+
    | Columns 12-17 | Time in YEAR MM DD HR MN SS                     |
    +---------------+-------------------------------------------------+

    Time is GPS time, and the time series are relative to each station's first epoch.

    .. _JPL's website: https://sideshow.jpl.nasa.gov/pub/JPL_GPS_Timeseries/\
repro2018a/raw/position/envseries/0000_README.format

    """
    def __init__(self, path, show_warnings=True):
        self._path = str(path)
        # load data
        data_cols = ["east", "north", "up"]
        var_cols = ["east_var", "north_var", "up_var"]
        cov_cols = ["east_north_cov", "east_up_cov", "north_up_cov"]
        all_cols = data_cols + var_cols + cov_cols
        time = pd.read_csv(self._path, delim_whitespace=True, header=None,
                           usecols=[11, 12, 13, 14, 15, 16],
                           names=["year", "month", "day", "hour", "minute", "second"])
        time = pd.to_datetime(time).to_frame(name="time")
        data = pd.read_csv(self._path, delim_whitespace=True, header=None,
                           usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                           names=all_cols)
        # convert to standard deviations and millimeters
        data.loc[:, var_cols] *= data.loc[:, var_cols]  # original data is in s.d.
        data.loc[:, data_cols] *= 1e3  # now in mm
        data.loc[:, var_cols + cov_cols] *= 1e6  # now in mm^2
        df = time.join(data)
        # check for duplicate timestamps and create time index
        num_duplicates = int(df.duplicated(subset="time").sum())
        if show_warnings and (num_duplicates > 0):
            warn(f"Timeseries file {path} contains data for {num_duplicates} duplicate dates. "
                 "Keeping first occurrences.")
        df = df.drop_duplicates(subset="time").set_index("time")
        # construct Timeseries object
        super().__init__(dataframe=df, src=".tseries", data_unit="mm",
                         data_cols=data_cols, var_cols=var_cols, cov_cols=cov_cols)

    def get_arch(self):
        """
        Returns a JSON-compatible dictionary with all the information necessary to recreate
        the Timeseries instance (provided the data file is available).

        Returns
        -------
        dict
            JSON-compatible dictionary sufficient to recreate the GipsyTimeseries instance.

        See Also
        --------
        Timeseries.get_arch : For further information.
        """
        return {"type": "GipsyTimeseries",
                "kw_args": {"path": self._path}}


class UNRTimeseries(Timeseries):
    """
    Subclasses :class:`~Timeseries`.

    Timeseries subclass for GNSS measurements in UNR's ``.tenv3`` file format.
    The data and (co)variances are converted into millimeters (squared).

    Parameters
    ----------
    path : str
        Path to the timeseries file.
    show_warnings : bool, optional
        If ``True``, warn if there are data inconsistencies encountered while loading.
        Defaults to ``True``.

    Notes
    -----

    The column format is described on `UNR's website`_:

    +---------------+---------------------------------------------------+
    | Columns       | Description                                       |
    +===============+===================================================+
    | Column 1      | Station name                                      |
    +---------------+---------------------------------------------------+
    | Column 2      | Date                                              |
    +---------------+---------------------------------------------------+
    | Column 3      | Decimal year                                      |
    +---------------+---------------------------------------------------+
    | Column 4      | Modified Julian day                               |
    +---------------+---------------------------------------------------+
    | Columns 5-6   | GPS week and day                                  |
    +---------------+---------------------------------------------------+
    | Column 7      | Longitude [°] of reference meridian               |
    +---------------+---------------------------------------------------+
    | Columns 8-9   | Easting [m] from ref. mer., integer and fraction  |
    +---------------+---------------------------------------------------+
    | Columns 10-11 | Northing [m] from equator, integer and fraction   |
    +---------------+---------------------------------------------------+
    | Columns 12-13 | Vertical [m], integer and fraction                |
    +---------------+---------------------------------------------------+
    | Column 14     | Antenna height [m]                                |
    +---------------+---------------------------------------------------+
    | Column 15-17  | East, North, Vertical standard deviation [m]      |
    +---------------+---------------------------------------------------+
    | Column 18     | East-North correlation coefficient [-]            |
    +---------------+---------------------------------------------------+
    | Column 19     | East-Vertical correlation coefficient [-]         |
    +---------------+---------------------------------------------------+
    | Column 20     | North-Vertical correlation coefficient [-]        |
    +---------------+---------------------------------------------------+

    The time series are relative to each station's first integer epoch.

    .. _UNR's website: http://geodesy.unr.edu/gps_timeseries/README_tenv3.txt

    """
    def __init__(self, path, show_warnings=True):
        self._path = str(path)
        # load data and check for some warnings
        df = pd.read_csv(self._path, delim_whitespace=True,
                         usecols=[0, 3] + list(range(6, 13)) + list(range(14, 20)))
        if show_warnings and int(df.duplicated(subset="site").sum()) > 1:
            warn(f"Timeseries file {path} contains multiple site codes: "
                 f"{df['site'].unique()}")
        if int(df.duplicated(subset="reflon").sum()) > 1:
            raise NotImplementedError(f"Timeseries file {path} contains "
                                      "multiple reference longitudes: "
                                      f"{df['reflon'].unique()}")
        if int(df.duplicated(subset="_e0(m)").sum()) > 1:
            if show_warnings:
                warn(f"Timeseries file {path} contains multiple integer "
                     f"Eastings: {df['_e0(m)'].unique()}")
            offsets_east = df["_e0(m)"].values - df["_e0(m)"].values[0]
        else:
            offsets_east = 0
        if int(df.duplicated(subset="____n0(m)").sum()) > 1:
            if show_warnings:
                warn(f"Timeseries file {path} contains multiple integer "
                     f"Northings: {df['____n0(m)'].unique()}")
            offsets_north = df["____n0(m)"].values - df["____n0(m)"].values[0]
        else:
            offsets_north = 0
        if int(df.duplicated(subset="u0(m)").sum()) > 1:
            if show_warnings:
                warn(f"Timeseries file {path} contains multiple integer "
                     f"Verticals: {df['u0(m)'].unique()}")
            offsets_up = df["u0(m)"].values - df["u0(m)"].values[0]
        else:
            offsets_up = 0
        # remove columns that are no longer needed
        df.drop(columns=["site", "reflon"], inplace=True)
        # make the data
        df["east"] = (df["__east(m)"] + offsets_east) * 1e3
        df["north"] = (df["_north(m)"] + offsets_north) * 1e3
        df["up"] = (df["____up(m)"] + offsets_up) * 1e3
        df.drop(columns=["_e0(m)", "__east(m)", "____n0(m)", "_north(m)",
                         "u0(m)", "____up(m)"], inplace=True)
        # make the covariance
        df["__corr_en"] *= df["sig_e(m)"] * df["sig_n(m)"] * 1e6
        df["__corr_eu"] *= df["sig_e(m)"] * df["sig_u(m)"] * 1e6
        df["__corr_nu"] *= df["sig_n(m)"] * df["sig_u(m)"] * 1e6
        df.rename(columns={"__corr_en": "east_north_cov", "__corr_eu": "east_up_cov",
                           "__corr_nu": "north_up_cov"}, inplace=True)
        # make the variance
        old_sig_cols = ["sig_e(m)", "sig_n(m)", "sig_u(m)"]
        df.loc[:, old_sig_cols] = (df.loc[:, old_sig_cols] * 1e3)**2
        df.rename(columns={"sig_e(m)": "east_var", "sig_n(m)": "north_var",
                           "sig_u(m)": "up_var"}, inplace=True)
        # check for duplicate timestamps and create time index
        num_duplicates = int(df.duplicated(subset="__MJD").sum())
        if show_warnings and (num_duplicates > 0):
            warn(f"Timeseries file {path} contains data for {num_duplicates} duplicate dates. "
                 "Keeping first occurrences.")
        df.drop_duplicates(subset="__MJD", inplace=True)
        df["__MJD"] = pd.to_datetime(df["__MJD"] + 2400000.5, unit="D", origin='julian')
        df.rename(columns={"__MJD": "time"}, inplace=True)
        df.set_index("time", inplace=True)
        # construct Timeseries object
        super().__init__(dataframe=df, src=".tenv3", data_unit="mm",
                         data_cols=["east", "north", "up"],
                         var_cols=["east_var", "north_var", "up_var"],
                         cov_cols=["east_north_cov", "east_up_cov", "north_up_cov"])

    def get_arch(self):
        """
        Returns a JSON-compatible dictionary with all the information necessary to recreate
        the Timeseries instance (provided the data file is available).

        Returns
        -------
        dict
            JSON-compatible dictionary sufficient to recreate the UNRTimeseries instance.

        See Also
        --------
        Timeseries.get_arch : For further information.
        """
        return {"type": "UNRTimeseries",
                "kw_args": {"path": self._path}}
