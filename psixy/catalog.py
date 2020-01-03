# -*- coding: utf-8 -*-
# Copyright 2020 The PsiXy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for formatting stimulus data into a standardized catalog.

Classes:
    Catalog: An object for keeping track of stimulus IDs and stimulus
        filepaths.

Functions:
    load_catalog: Load a hdf5 file as a psixy.models.Catalog
        object.

"""

import os
from pathlib import Path

import h5py
import numpy as np


class Catalog(object):
    """Class to manage stimuli information.

    Attributes:
        n_stimuli:  Integer indicating number of unique stimuli.
        filepath: A list of pathlib.Path objects indicating the
            corresponding filepath for each stimulus. Note that this is
            a list of paths relative to an (optional) common filepath
            shared by all stimuli.
            len=n_stimuli
        common_path: A pathlib.Path object indicating the common
            filepath (if any).
        stimulus_id: A non-negative integer array indicating the
            unique stimulus IDs.
            shape=(n_stimuli,)

    Methods:
        fullpath: Return the full filepath for the stimuli.
        save: Save the catalog to disk using hdf5 format.
        subset: Return a subset of the catalog.

    """

    def __init__(
            self, filepath, common_path=None, stimulus_id=None):
        """Initialize.

        Arguments:
            filepath: A list of strings or pathlib.Path objects.
                len=n_stimuli
            stimulus_id (optional): A 1D non-negative integer NumPy
                array composed of all unique values. The integers do
                not need to be consecutive.
                shape=(n_stimuli,)

        """
        # Set number of stimuli.
        self.n_stimuli = len(filepath)

        # Set file paths.
        self.filepath = self._check_filepath(filepath)
        if common_path is None:
            common_path = Path('.')
        self._common_path = Path(common_path)

        # Set stimulus IDs.
        self.stimulus_id = self._check_stimulus_id(stimulus_id)
        self._version = '0.1.0'

    def _check_filepath(self, filepath):
        """Check `filepath` argument.

        Returns:
            filepath

        Raises:
            ValueError

        """
        path_list = []
        for i_file in filepath:
            path_list.append(Path(i_file))
        return path_list

    def _check_stimulus_id(self, stimulus_id):
        """Check `stimulus_id` argument.

        Returns:
            stimulus_id

        Raises:
            ValueError

        """
        if stimulus_id is None:
            stimulus_id = np.arange(self.n_stimuli)

        if len(stimulus_id.shape) != 1:
            raise ValueError((
                "The argument `stimulus_id` must be a NumPy 1D array of "
                "non-negative integers. The array you supplied is "
                "not 1D."
            ))

        if len(stimulus_id) != self.n_stimuli:
            raise ValueError((
                'The argument `stimulus_id` must have the same length as '
                '`filepath`.'
            ))

        if not issubclass(stimulus_id.dtype.type, np.integer):
            raise ValueError((
                "The argument `stimulus_id` must be a 1D NumPy array of "
                "unique non-negative integers. The array you supplied is not "
                "composed of integers."
            ))

        if np.sum(np.less(stimulus_id, 0)) > 0:
            raise ValueError((
                "The argument `stimulus_id` must be a 1D NumPy array of "
                "unique non-negative integers. The array you supplied is not "
                "non-negative."
            ))

        if len(np.unique(stimulus_id)) != self.n_stimuli:
            raise ValueError((
                "The argument `stimulus_id` must be a 1D NumPy array of "
                "unique non-negative integers. The array you supplied is not "
                "composed of unique values."
            ))

        return stimulus_id

    @property
    def common_path(self):
        """Getter method for common_path."""
        return self._common_path

    @common_path.setter
    def common_path(self, c):
        """Setter method for common_path."""
        self._common_path = Path(c)

    def fullpath(self):
        """Return list of complete stimuli filepaths."""
        file_path_list = self.filepath
        file_path_list = [
            self.common_path / i_file for i_file in file_path_list
        ]
        return file_path_list

    def save(self, filepath):
        """Save the Catalog object as an HDF5 file.

        Arguments:
            filepath: A string or pathlib.Path object specifying the
                path to save the catalog object.

        """
        # Convert stimuli filepaths to strings for saving.
        path_str = []
        for i_file in self.filepath:
            path_str.append(os.fspath(i_file))
        max_filepath_length = len(max(path_str, key=len))
        path_array = np.asarray(
            path_str, dtype="S{0}".format(max_filepath_length)
        )
        common_path = os.fspath(self.common_path)

        f = h5py.File(filepath, "w")
        f.create_dataset("stimulus_filepath", data=path_array)
        f.create_dataset("common_path", data=common_path)
        f.create_dataset("stimulus_id", data=self.stimulus_id)
        f.close()

    def subset(self, idx_array):
        """Return a subset of catalog."""
        if idx_array.dtype == 'bool':
            dmy_idx = np.arange(len(idx_array))
            idx_array = dmy_idx[idx_array]

        filepath = []
        for idx in idx_array:
            filepath.append(self.filepath[idx])

        catalog = Catalog(
            filepath,
            common_path=self.common_path,
            stimulus_id=self.stimulus_id[idx_array],
        )
        return catalog

    def convert_filenames(self, filepath_list_in):
        """Convert filepaths to corresponding catalog indices.

        Arguments:
            filepath_list_in: A list of filepaths.

        Returns:
            catalog_idx_list: A NumPy array of indices.

        """
        catalog_idx_list = -1 * np.ones(len(filepath_list_in), dtype=int)
        for idx_in, filename_in in enumerate(filepath_list_in):
            was_found = False
            for idx_catalog, filename_catalog in enumerate(self.filepath):
                if filename_in.name == filename_catalog.name:
                    catalog_idx_list[idx_in] = idx_catalog
                    was_found = True
                    break
        return catalog_idx_list


def load_catalog(filepath, verbose=0):
    """Load catalog saved via the save method.

    The loaded data is instantiated as a psixy.datasets.Catalog object.

    Arguments:
        filepath: The location of the hdf5 file to load.
        verbose (optional): Controls the verbosity of printed summary.

    Returns:
        Loaded catalog.

    """
    f = h5py.File(filepath, "r")
    stimulus_filepath = f["stimulus_filepath"][()].astype('U')
    common_path = f["common_path"][()]
    stimulus_id = f["stimulus_id"][()]
    catalog = Catalog(
        stimulus_filepath, stimulus_id=stimulus_id, common_path=common_path
    )
    f.close()

    if verbose > 0:
        print("Catalog Summary")
        print('  n_stimuli: {0}'.format(catalog.n_stimuli))
        print('')
    return catalog
