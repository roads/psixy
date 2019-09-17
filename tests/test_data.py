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

"""Module for testing `catalog.py`."""

from pathlib import Path
import pytest
import numpy as np

import psixy.catalog


class TestCatalog:
    """Test class Catalog."""

    def test_initialization(self):
        """Test initialization of class."""
        # No provided task.
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = [
            'rex/a.jpg', 'rex/b.jpg', 'rex/c.jpg', 'rex/d.jpg', 'rex/e.jpg',
            'rex/f.jpg'
        ]
        path_list = [
            Path('rex/a.jpg'), Path('rex/b.jpg'), Path('rex/c.jpg'),
            Path('rex/d.jpg'), Path('rex/e.jpg'), Path('rex/f.jpg')
        ]

        catalog = psixy.catalog.Catalog(stimulus_id, stimulus_filepath)
        np.testing.assert_array_equal(catalog.stimulus_id, stimulus_id)
        assert catalog.n_stimuli == 6

        assert catalog.filepath == path_list
        assert catalog.common_path == Path('.')
        assert catalog.fullpath() == path_list

        np.testing.assert_array_equal(catalog.class_id, np.zeros([6, 1]))
        assert catalog.n_task == 1

        # One provided task.
        class_id = np.array([0, 1, 2, 3, 4, 5])
        catalog = psixy.catalog.Catalog(
            stimulus_id, stimulus_filepath, class_id=class_id
        )
        np.testing.assert_array_equal(
            catalog.class_id, np.expand_dims(class_id, 1)
        )
        assert catalog.n_task == 1

        # Two provided tasks.
        class_id = np.array(
            [[0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]]
        )
        class_id = np.transpose(class_id)
        catalog = psixy.catalog.Catalog(
            stimulus_id, stimulus_filepath, class_id=class_id
        )
        np.testing.assert_array_equal(
            catalog.class_id, class_id
        )
        assert catalog.n_task == 2

        # Common path.
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = [
            'a.jpg', 'b.jpg', 'c.jpg', 'd.jpg', 'e.jpg', 'f.jpg'
        ]
        path_list_rel = [
            Path('a.jpg'), Path('b.jpg'), Path('c.jpg'),
            Path('d.jpg'), Path('e.jpg'), Path('f.jpg')
        ]
        path_list_full = [
            Path('rex/a.jpg'), Path('rex/b.jpg'), Path('rex/c.jpg'),
            Path('rex/d.jpg'), Path('rex/e.jpg'), Path('rex/f.jpg')
        ]

        catalog = psixy.catalog.Catalog(stimulus_id, stimulus_filepath)
        catalog.common_path = 'rex'
        assert catalog.filepath == path_list_rel
        assert catalog.common_path == Path('rex')
        assert catalog.fullpath() == path_list_full

        # Common path.
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = [
            'a.jpg', 'b.jpg', 'c.jpg', 'd.jpg', 'e.jpg', 'f.jpg'
        ]
        path_list_rel = [
            Path('a.jpg'), Path('b.jpg'), Path('c.jpg'),
            Path('d.jpg'), Path('e.jpg'), Path('f.jpg')
        ]
        path_list_full = [
            Path('cold/rex/a.jpg'), Path('cold/rex/b.jpg'),
            Path('cold/rex/c.jpg'), Path('cold/rex/d.jpg'),
            Path('cold/rex/e.jpg'), Path('cold/rex/f.jpg')
        ]

        catalog = psixy.catalog.Catalog(stimulus_id, stimulus_filepath)
        catalog.common_path = 'cold/rex'
        assert catalog.filepath == path_list_rel
        assert catalog.common_path == Path('cold', 'rex')
        assert catalog.fullpath() == path_list_full

        # Bad input shape.
        stimulus_id = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
        stimulus_filepath = [
            'rex/a.jpg', 'rex/b.jpg', 'rex/c.jpg', 'rex/d.jpg', 'rex/e.jpg',
            'rex/f.jpg'
        ]
        with pytest.raises(Exception) as e_info:
            catalog = psixy.catalog.Catalog(stimulus_id, stimulus_filepath)

        # A non-integer stimulus_id value.
        stimulus_id = np.array([0, 1, 2., 3, 4, 5])
        stimulus_filepath = [
            'r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg', 'r/f.jpg']
        with pytest.raises(Exception) as e_info:
            catalog = psixy.catalog.Catalog(stimulus_id, stimulus_filepath)

        # Zero stimulus_id not present.
        stimulus_id = np.array([1, 2, 3, 4, 5])
        stimulus_filepath = [
            'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg', 'r/f.jpg']
        with pytest.raises(Exception) as e_info:
            catalog = psixy.catalog.Catalog(stimulus_id, stimulus_filepath)

        # Two stimulus_id's not present.
        stimulus_id = np.array([0, 1, 2, 5])
        stimulus_filepath = ['r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/f.jpg']
        with pytest.raises(Exception) as e_info:
            catalog = psixy.catalog.Catalog(stimulus_id, stimulus_filepath)

        # Bad shape.
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = [
            ['r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg'],
            ['r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg']]
        with pytest.raises(Exception) as e_info:
            catalog = psixy.catalog.Catalog(stimulus_id, stimulus_filepath)

        # Mismatch in number (too few).
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = [
            'r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg']
        with pytest.raises(Exception) as e_info:
            catalog = psixy.catalog.Catalog(stimulus_id, stimulus_filepath)

        # Mismatch in number (too many).
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = [
            'r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg', 'r/f.jpg',
            'f/g.jpg']
        with pytest.raises(Exception) as e_info:
            catalog = psixy.catalog.Catalog(stimulus_id, stimulus_filepath)

    def test_persistence(self, tmpdir):
        """Test object persistence."""
        # Create Catalog object.
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = [
            'r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg', 'r/f.jpg'
        ]
        catalog = psixy.catalog.Catalog(stimulus_id, stimulus_filepath)
        # Save Catalog.
        fn = tmpdir.join('catalog_test.hdf5')
        catalog.save(fn)
        # Load the saved catalog.
        loaded_catalog = psixy.catalog.load_catalog(fn)
        # Check that the loaded Docket object is correct.
        assert catalog.n_stimuli == loaded_catalog.n_stimuli
        np.testing.assert_array_equal(
            catalog.stimulus_id, loaded_catalog.stimulus_id
        )
        assert catalog.filepath == loaded_catalog.filepath
        np.testing.assert_array_equal(
            catalog.class_id, loaded_catalog.class_id
        )
