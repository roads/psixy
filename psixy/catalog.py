# -*- coding: utf-8 -*-
# Copyright 2019 The PsiXy Authors. All Rights Reserved.
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
    Catalog: An object for keeping track of stimuli, stimuli filepaths
        and stimuli class labels.

Functions:
    load_catalog: Load a hdf5 file as a psixy.models.Catalog
        object.


"""

import os
from pathlib import Path

import h5py
import numpy as np


class Catalog(object):
    """Class to keep track of stimuli information.

    Attributes:
        n_stimuli:  Integer indicating number of unique stimuli.
        stimulus_id: A integer array indicating the unique stimulus IDs.
            shape=(n_stimuli,)
        filepath: A list of pathlib.Path objects indicating the
            corresponding filepath for each stimulus. Note that this is
            a list of paths relative to a common filepath shared by all
            stimuli.
            len=n_stimuli
        common_path: A pathlib.Path object indicating the common
            filepath (if any).
        class_id: An integer array indicating the corresponding class
            ID.
            shape=(n_stimuli, n_task)
        n_task: Integer indicating number of classification tasks
            defined for the catalog.
        class_label: A dictionary mapping between the (integer)
            class_id and a (string) label.
        version: A string indicating the version of the class to
            facilitate loading old class version.

    Methods:
        task: Return class IDs associated with a particular task.
        fullpath: Return the full filepath for the stimuli.
        save: Save the catalog to disk using hdf5 format.
        subset: Return a subset of the catalog.

    """

    def __init__(
            self, stimulus_id, filepath, class_id=None, class_label=None):
        """Initialize.

        Arguments:
            stimulus_id: A 1D integer array.
                shape=(n_stimuli,)
            filepath: A list of strings or pathlib.Path objects.
                len=n_stimuli
            class_id (optional): A ND integer array.
                shape=(n_stimuli, n_task)
            class_label (optional): A dictionary mapping between each
                (integer) class_id and a (string) label.
        """
        # Set stimulus ID.
        self.stimulus_id = self._check_stimulus_id(stimulus_id)
        self.n_stimuli = len(stimulus_id)

        # Set file paths.
        # self.common_path = os.path.commonpath(filepath)  TODO
        self.filepath = self._check_filepath(filepath)
        self._common_path = Path('.')

        # Set class_id.
        if class_id is None:
            class_id = np.zeros([self.n_stimuli, 1], dtype=int)
        else:
            class_id = self._check_class_id(class_id)
        self.class_id = class_id
        self.n_task = class_id.shape[1]

        # Set class label mapping. TODO
        if class_label is None:
            # Make the label the ID.
            class_label = {}
            class_id_unique = np.unique(self.class_id)
            for class_id in class_id_unique:
                class_label[class_id] = '{0}'.format(class_id)
        else:
            class_label = self._check_class_label(class_label)
        self.class_label = class_label
        self.version = '0.1.0'

    def _check_stimulus_id(self, stimulus_id):
        """Check `stimulus_id` argument.

        Returns:
            stimulus_id

        Raises:
            ValueError

        """
        if len(stimulus_id.shape) != 1:
            raise ValueError((
                "The argument `stimulus_id` must be a 1D array of "
                "integers."))

        if not issubclass(stimulus_id.dtype.type, np.integer):
            raise ValueError((
                "The argument `stimulus_id` must be a 1D array of "
                "integers."))

        n_stimuli = len(stimulus_id)

        is_contiguous = False
        if np.array_equal(np.unique(stimulus_id), np.arange(0, n_stimuli)):
            is_contiguous = True
        if not is_contiguous:
            raise ValueError((
                'The argument `stimulus_id` must contain a contiguous set of '
                'integers [0, n_stimuli[.'))
        return stimulus_id

    def _check_filepath(self, filepath):
        """Check `filepath` argument.

        Returns:
            filepath

        Raises:
            ValueError

        """
        if len(filepath) != self.n_stimuli:
            raise ValueError((
                'The argument `filepath` must have the same length as '
                '`stimulus_id`.'))

        path_list = []
        for i_file in filepath:
            path_list.append(Path(i_file))

        return path_list

    def _check_class_id(self, class_id):
        """Check `class_id` argument.

        Returns:
            class_id

        Raises:
            ValueError

        """
        if len(class_id.shape) == 1:
            class_id = np.expand_dims(class_id, 1)

        if len(class_id.shape) > 2:
            raise ValueError((
                "The argument `stimulus_id` must be a 1D or 2D array "
                "of integers."
            ))

        if not issubclass(class_id.dtype.type, np.integer):
            raise ValueError((
                "The argument `stimulus_id` must be an ND array of "
                "integers."))

        return class_id

    def _check_class_label(self, class_label):
        """Check `class_label` argument.

        Returns:
            class_label

        Raises:
            ValueError

        """
        # Check to make sure there is a label for all class IDs.
        class_id_unique = np.unique(self.class_id)
        for class_id in class_id_unique:
            if not (class_id in class_label):
                raise ValueError((
                    "The argument `class_label` must contain a label "
                    "for all class IDs. Missing label for "
                    "class_id={0}.".format(class_id)
                ))
        return class_label

    @property
    def common_path(self):
        """Getter method for phi."""
        return self._common_path

    @common_path.setter
    def common_path(self, c):
        """Setter method for common_path."""
        self._common_path = Path(c)

    def task(self, task_idx=0):
        """Return class IDs associated with task."""
        return self.class_id[:, task_idx]

    def fullpath(self):
        """Return full stimuli filepaths."""
        file_path_list = self.filepath
        file_path_list = [
            self.common_path / i_file for i_file in file_path_list
        ]
        return file_path_list

    def save(self, filepath):
        """Save the Catalog object as an HDF5 file.

        Arguments:
            filepath: String specifying the path to save the data.

        """
        # Convert stimuli filepaths to strings for saving.
        path_str = []
        for i_file in self.filepath:
            path_str.append(os.fspath(i_file))
        max_filepath_length = len(max(path_str, key=len))
        path_array = np.asarray(
            path_str, dtype="S{0}".format(max_filepath_length)
        )

        f = h5py.File(filepath, "w")
        f.create_dataset("stimulus_id", data=self.stimulus_id)
        f.create_dataset(
            "stimulus_filepath",
            data=path_array
        )
        f.create_dataset("class_id", data=self.class_id)

        # Handle class ID to label mapping.
        max_label_length = len(max(self.class_label.values(), key=len))
        n_class = len(self.class_label)
        class_map_class_id = np.empty(n_class, dtype=np.int)
        class_map_label = np.empty(n_class, dtype="S{0}".format(
            max_label_length
        ))
        idx = 0
        for key, value in self.class_label.items():
            class_map_class_id[idx] = key
            class_map_label[idx] = value
            idx = idx + 1

        f.create_dataset(
            "class_map_class_id",
            data=class_map_class_id
        )
        f.create_dataset(
            "class_map_label",
            data=class_map_label
        )

        f.close()

    def subset(self, idx, squeeze=False):
        """Return a subset of catalog with new stimulus IDs."""
        catalog = copy.deepcopy(self)
        catalog.stimuli = catalog.stimuli.iloc[idx]
        catalog.n_stimuli = len(catalog.stimuli)
        if squeeze:
            catalog.stimuli.at[:, "id"] = np.arange(0, catalog.n_stimuli)
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

    def convert_labels(self, label_list_in):
        """Convert class labels to corresponding class IDs.

        Returns -1 for any class labels that are not found.

        Arguments:
            label_list_in: A list of labels.

        Returns:
            class_id_list: A NumPy array of indices.

        """
        class_id_list = -1 * np.ones(len(label_list_in), dtype=int)
        for idx_in, label_in in enumerate(label_list_in):
            was_found = False
            for class_id_cat, label_cat in self.class_label.items():
                if label_in == label_cat:
                    class_id_list[idx_in] = class_id_cat
                    was_found = True
                    break
        return class_id_list


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
    stimulus_id = f["stimulus_id"][()]
    stimulus_filepath = f["stimulus_filepath"][()].astype('U')
    class_id = f["class_id"][()]

    class_map_class_id = f["class_map_class_id"][()]
    class_map_label = f["class_map_label"][()]
    class_label_dict = {}
    for idx in np.arange(len(class_map_class_id)):
        class_label_dict[class_map_class_id[idx]] = (
            class_map_label[idx].decode('ascii')
        )

    catalog = Catalog(
        stimulus_id, stimulus_filepath, class_id, class_label_dict)
    f.close()

    if verbose > 0:
        print("Catalog Summary")
        print('  n_stimuli: {0}'.format(catalog.n_stimuli))
        # print('  n_task: {0}'.format())  TODO
        print('')
    return catalog
