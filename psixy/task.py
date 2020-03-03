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
    Task: An object that maps stimuli to a class.

Functions:
    load_task: Load a hdf5 file as a psixy.models.Task object.

TODO:
    * `map` method
    * `convert` method
    * implement save
    * implement load_task

"""

import numpy as np

import psixy.catalog


class Task(object):
    """Class that maps stimuli to class IDs.

    Attributes:
        catalog: A psixy.catalog.Catalog object.
        class_id: An array of non-negative integers indicating the
            class membership of each stimulus in the provided catalog.
        n_class: The number of unique classes in the task.
        class_label: A dictionary mapping between each
            (integer) class_id and a (string) label.
        name: A (string) name for the task.

    Methods:
        map: A helper function that maps an ND NumPy array representing
            stimulus_id's to a corresponding ND NumPy array of
            class_id's.
        convert: A helper function that takes class_label's
            (class_id's) and converts to class_id's (class_label's).
        save: Save the Task object to disk.

    """

    def __init__(self, catalog, class_id, class_label=None, name=None):
        """Initialize.

        Arguments:
            catalog: A psixy.catalog.Catalog object.
            class_id: An array of non-negative integers indicating the
                class membership of each stimulus in the provided
                catalog.
                shape=(n_stimuli,)
            class_label (optional): A dictionary mapping between each
                (integer) class_id and a (string) class label.
            name (optional): A (string) name for the task.

        """
        # Hook up catalog.
        self.catalog = catalog
        self.n_stimuli = catalog.n_stimuli
        self.stimulus_id = catalog.stimulus_id

        # Set task information.
        self.class_id = self._check_class_id(class_id)
        self.n_class = len(np.unique(self.class_id))
        self.class_label = self._check_class_label(class_label)
        self.name = self._check_task_name(name)

    def _check_class_id(self, class_id):
        """Check `class_id` argument.

        Returns:
            class_id

        Raises:
            ValueError

        """
        if len(class_id.shape) != 1:
            raise ValueError((
                "The argument `class_id` must be a 1D NumPy array of "
                "non-negative integers. The array you supplied is "
                "not 1D."
            ))

        if len(class_id) != self.catalog.n_stimuli:
            raise ValueError((
                'The argument `class_id` must have the same length as '
                'the number of stimuli in `catalog`.'
            ))

        if not issubclass(class_id.dtype.type, np.integer):
            raise ValueError((
                "The argument `class_id` must be a 1D NumPy array of "
                "non-negative integers. The array you supplied is not "
                "composed of integers."
            ))

        if np.sum(np.less(class_id, 0)) > 0:
            raise ValueError((
                "The argument `class_id` must be a 1D NumPy array of "
                "non-negative integers. The array you supplied is not "
                "non-negative."
            ))

        return class_id

    def _check_class_label(self, class_label):
        """Check `class_label` argument.

        Returns:
            class_label

        Raises:
            ValueError

        """
        if class_label is None:
            # Make the label the ID.
            class_label = {}
            class_id_unique = np.unique(self.class_id)
            for class_id in class_id_unique:
                class_label[class_id] = '{0}'.format(class_id)
        else:
            # Check to make sure there is a label for all class IDs.
            class_id_unique = np.unique(self.class_id)
            for class_id in class_id_unique:
                if not (class_id in class_label):
                    raise ValueError((
                        "The argument `class_label` must contain a label "
                        "for all class IDs. Missing at least one label "
                        "(e.g., class_id={0}).".format(class_id)
                    ))
        return class_label

    def _check_task_name(self, name):
        """Check `name` argument.

        Returns:
            name

        Raises:
            ValueError

        """
        if name is None:
            name = 'Task 0'
        else:
            if type(name) != str:
                raise ValueError("The argument `name` must be a string.")
        return name


def shepard_hovland_jenkins_1961():
    """Generate task list with six category types from 1961 paper.

    Type I
    Type II
    Type III
    Type IV
    Type V
    Type VI

    References:
        [1] Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961).
            Learning and Memorization of Classifications. Psychological
            Monographs: General and Applied, 75(13), 1-42.
            https://doi.org/10.1037/h0093825

    """
    # Placeholder filepaths.
    filepath = [
        '0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg'
    ]
    stimulus_id = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    catalog = psixy.catalog.Catalog(filepath, stimulus_id=stimulus_id)

    name_list = ['I', 'II', 'III', 'IV', 'V', 'VI']
    class_id = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 1]
    ])
    task_list = []
    for idx, name in enumerate(name_list):
        task_list.append(
            Task(catalog, class_id[idx], name=name)
        )

    feature_matrix = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])

    # TODO maybe return the encoder, still need the feature matrix
    # for RBF model initialization.
    # encoder = psixy.models.Deterministic(stimulus_id, feature_matrix)

    return task_list, feature_matrix


def kruschke_rules_and_exceptions():
    """Return a rules and exception category structure.

    See [1].

    References:
        [1] Kruschke, J. K. (1992). ALCOVE: an exemplar-based
            connectionist model of category learning. Psychological
            Review, 99(1), 22-44.
            http://dx.doi.org/10.1037/0033-295X.99.1.22.

    """
    # Placeholder filepaths.
    filepath = [
        '0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6jpg',
        '7.jpg', '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg'
    ]
    stimulus_id = np.arange(len(filepath))
    catalog = psixy.catalog.Catalog(filepath, stimulus_id=stimulus_id)

    class_id = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    task = Task(catalog, class_id)

    feature_matrix = np.array([
        [1, 1],
        [2, 1],
        [3, 1],
        [1, 3],
        [2, 3],
        [3, 3],
        [1, 4.4],
        [3, 4.6],
        [1, 6],
        [2, 6],
        [3, 6],
        [1, 8],
        [2, 8],
        [3, 8],
    ])

    stimulus_label = np.array([
        'A_1', 'A_2', 'A_3', 'A_4', 'A_5', 'A_6', 'B_e',
        'A_e', 'B_6', 'B_5', 'B_4', 'B_3', 'B_2', 'B_1',
    ])
    return task, feature_matrix, stimulus_label


def nosofsky_1986():
    """Category structure used in [1].

    There are three category structures: dimensional, criss-cross,
    interior-exterior, and diagonal.

    References:
    [1] Nosofsky, R. M. (1986). Attention, similarity, and the
        identification-categorization relationship. Journal of
        Experimental Psychology: General, 115, 39–57.

    """
    # Placeholder filepaths.
    filepath = [
        '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6jpg',
        '7.jpg', '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg',
        '13.jpg', '14.jpg', '15.jpg', '16.jpg',
    ]
    stimulus_id = np.arange(len(filepath))
    catalog = psixy.catalog.Catalog(filepath, stimulus_id=stimulus_id)

    name_list = [
        'Dimensional', 'Criss-Cross', 'Interior-Exterior', 'Diagonal'
    ]
    # See Figure 3 [1]. Stimuli are listed left to right, bottom to top in
    # order to start at coordinate (1,1). Class membership of transfer stimuli
    # is determined by using the ideal case, not by Figure 8 [1]. TODO
    is_train = np.array([
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    ], dtype=bool)
    class_id = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2],
        [2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2],
        [1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2]
    ], dtype=int)
    task_list = []
    for idx, name in enumerate(name_list):
        task_list.append(
            Task(catalog, class_id[idx], name=name)
        )

    # Subject 1 (Table 2 [1]).
    feature_matrix_1 = np.array([
        [-1.855, -1.532],
        [-0.687, -1.617],
        [0.436, -1.633],
        [1.331, -1.647],
        [-1.615, -0.469],
        [-0.531, -0.558],
        [0.500, -0.590],
        [1.373, -0.535],
        [-1.522, 0.657],
        [-0.395, 0.518],
        [0.648, 0.469],
        [1.513, 0.481],
        [-1.427, 1.770],
        [-0.301, 1.639],
        [0.767, 1.541],
        [1.764, 1.512]
    ])

    feature_matrix_2 = np.array([
        [-1.356, -1.430],
        [-0.492, -1.211],
        [0.349, -1.235],
        [1.088, -1.121],
        [-1.413, -0.528],
        [-0.412, -0.425],
        [0.518, -0.382],
        [1.275, -0.215],
        [-1.477, 0.302],
        [-0.388, 0.294],
        [0.607, 0.368],
        [1.317, 0.539],
        [-1.389, 1.189],
        [-0.313, 1.153],
        [0.673, 1.187],
        [1.414, 1.518]
    ])

    feature_matrix_list = [feature_matrix_1, feature_matrix_2]
    return task_list, feature_matrix_list
