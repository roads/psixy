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

"""Module for testing `task.py`."""

from pathlib import Path
import pytest
import numpy as np

import psixy.catalog
import psixy.task


@pytest.fixture(scope="module")
def simple_catalog():
    """Return a simple psixy.catalog.Catalog object."""
    path_list = [
            Path('romeo/alpha.jpg'), Path('romeo/brave.jpg'),
            Path('romeo/charlie.jpg'), Path('romeo/delta.jpg'),
            Path('romeo/echo.jpg'), Path('romeo/foxtrot.jpg')
        ]
    catalog = psixy.catalog.Catalog(path_list)
    return catalog


class TestTask:
    """Test functionality of class Task."""

    def test_init_minimum(self, simple_catalog):
        """Test object initialization with minimum arguments."""
        # Minimum initialization.
        class_id = np.array([0, 0, 1, 6, 1, 0])
        task = psixy.task.Task(simple_catalog, class_id)
        class_label = {0: '0', 1: '1', 6: '6'}
        task_name = 'Task 0'
        assert task.n_stimuli == 6
        np.testing.assert_array_equal(
            task.stimulus_id, simple_catalog.stimulus_id
        )
        np.testing.assert_array_equal(
            task.class_id, class_id
        )
        assert task.n_class == 3
        assert task.class_label == class_label
        assert task.name == task_name

        # Minimum initialization (invalid length input).
        class_id = np.array([0, 0, 0, 1, 6])
        with pytest.raises(Exception) as e_info:
            task = psixy.task.Task(simple_catalog, class_id)

        # Minimum initialization (invalid length input).
        class_id = np.array([0, 0, 0, 1, 1, 6, 1])
        with pytest.raises(Exception) as e_info:
            task = psixy.task.Task(simple_catalog, class_id)

        # Minimum initialization (invalid float input).
        class_id = np.array([0, 0, 1.0, 0, 1, 6])
        with pytest.raises(Exception) as e_info:
            task = psixy.task.Task(simple_catalog, class_id)

        # Minimum initialization (invalid negative input).
        class_id = np.array([0, 0, -1, 0, 1, 6])
        with pytest.raises(Exception) as e_info:
            task = psixy.task.Task(simple_catalog, class_id)

    def test_init_with_labels(self, simple_catalog):
        """Test initialization with optional labels."""
        class_id = np.array([0, 0, 1, 6, 1, 0])
        class_label = {0: 'Alpha', 1: 'Bravo', 6: 'Charlie'}

        task = psixy.task.Task(
            simple_catalog, class_id, class_label=class_label
        )

        assert task.n_stimuli == 6
        np.testing.assert_array_equal(
            task.stimulus_id, simple_catalog.stimulus_id
        )
        np.testing.assert_array_equal(
            task.class_id, class_id
        )
        assert task.n_class == 3
        assert task.class_label == class_label

        # Test when provided labels are too few.
        class_label = {0: 'Alpha', 6: 'Charlie'}
        with pytest.raises(Exception) as e_info:
            task = psixy.task.Task(
                simple_catalog, class_id, class_label=class_label
            )

        # Test when provided labels don't match.
        class_label = {0: 'Alpha', 2: 'Delta', 6: 'Charlie'}
        with pytest.raises(Exception) as e_info:
            task = psixy.task.Task(
                simple_catalog, class_id, class_label=class_label
            )

    def test_init_with_name(self, simple_catalog):
        """Test initialization with optional labels."""
        class_id = np.array([0, 0, 1, 6, 1, 0])
        task_name = 'Task 42'
        task = psixy.task.Task(
            simple_catalog, class_id, name=task_name
        )
        assert task.name == task_name

        with pytest.raises(Exception) as e_info:
            task = psixy.task.Task(
                simple_catalog, class_id, name=26
            )
