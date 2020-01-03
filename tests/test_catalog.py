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

"""Module for testing `catalog.py`."""

from pathlib import Path
import pytest
import numpy as np

import psixy.catalog


class TestCatalog:
    """Test class Catalog."""

    def test_initialization(self):
        """Test initialization of class."""
        # Minimal input as string. No `common_path` or `stimulus_id`.
        stimulus_filepath = [
            'rex/a.jpg', 'rex/b.jpg', 'rex/c.jpg', 'rex/d.jpg', 'rex/e.jpg',
            'rex/f.jpg'
        ]
        path_list = [
            Path('rex/a.jpg'), Path('rex/b.jpg'), Path('rex/c.jpg'),
            Path('rex/d.jpg'), Path('rex/e.jpg'), Path('rex/f.jpg')
        ]
        catalog = psixy.catalog.Catalog(stimulus_filepath)

        assert catalog.n_stimuli == 6
        assert catalog.filepath == path_list
        assert catalog.common_path == Path('.')
        assert catalog.fullpath() == path_list
        np.testing.assert_array_equal(
            catalog.stimulus_id, np.arange(6, dtype=int)
        )

        # Minimal input as Path. No `common_path` or `stimulus_id`.
        path_list = [
            Path('rex/a.jpg'), Path('rex/b.jpg'), Path('rex/c.jpg'),
            Path('rex/d.jpg'), Path('rex/e.jpg'), Path('rex/f.jpg')
        ]
        catalog = psixy.catalog.Catalog(path_list)

        assert catalog.n_stimuli == 6
        assert catalog.filepath == path_list
        assert catalog.common_path == Path('.')
        assert catalog.fullpath() == path_list
        np.testing.assert_array_equal(
            catalog.stimulus_id, np.arange(6, dtype=int)
        )

        # Provide optional common path.
        common_path = 'rex'
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
        catalog = psixy.catalog.Catalog(
            stimulus_filepath, common_path=common_path
        )

        assert catalog.n_stimuli == 6
        assert catalog.filepath == path_list_rel
        assert catalog.common_path == Path('rex')
        assert catalog.fullpath() == path_list_full
        np.testing.assert_array_equal(
            catalog.stimulus_id, np.arange(6, dtype=int)
        )

        # Provide optional stimulus id (valid input).
        stimulus_id = np.array([52, 7, 2, 3, 4, 5])
        path_list = [
            Path('rex/a.jpg'), Path('rex/b.jpg'), Path('rex/c.jpg'),
            Path('rex/d.jpg'), Path('rex/e.jpg'), Path('rex/f.jpg')
        ]
        catalog = psixy.catalog.Catalog(path_list, stimulus_id=stimulus_id)

        assert catalog.n_stimuli == 6
        assert catalog.filepath == path_list
        assert catalog.common_path == Path('.')
        assert catalog.fullpath() == path_list
        np.testing.assert_array_equal(
            catalog.stimulus_id, stimulus_id
        )

        # Provide optional stimulus id (float input).
        stimulus_id = np.array([0, 1.0, 2, 3, 4, 5])
        with pytest.raises(Exception) as e_info:
            catalog = psixy.catalog.Catalog(path_list, stimulus_id=stimulus_id)

        # Provide optional stimulus id (negative values).
        stimulus_id = np.array([-1, 1, 2, 3, 4, 5])
        with pytest.raises(Exception) as e_info:
            catalog = psixy.catalog.Catalog(path_list, stimulus_id=stimulus_id)

        # Provide optional stimulus id (non-unique values).
        stimulus_id = np.array([1, 1, 2, 3, 4, 5])
        with pytest.raises(Exception) as e_info:
            catalog = psixy.catalog.Catalog(path_list, stimulus_id=stimulus_id)

    def test_persistence(self, tmpdir):
        """Test object persistence."""
        stimulus_id = np.array([6, 0, 2, 7, 4, 5])
        stimulus_filepath = [
            'r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg', 'r/f.jpg'
        ]

        # Save and load Catalog object without common path.
        catalog = psixy.catalog.Catalog(
            stimulus_filepath, stimulus_id=stimulus_id
        )
        fn = tmpdir.join('catalog_test.hdf5')
        catalog.save(fn)
        loaded_catalog = psixy.catalog.load_catalog(fn)

        # Check that the loaded Catalog object is correct.
        assert catalog.n_stimuli == loaded_catalog.n_stimuli
        assert catalog.filepath == loaded_catalog.filepath
        assert catalog.common_path == loaded_catalog.common_path
        np.testing.assert_array_equal(
            catalog.stimulus_id, loaded_catalog.stimulus_id
        )

        # Save and load Catalog object with common path.
        catalog = psixy.catalog.Catalog(
            stimulus_filepath, common_path='bravo', stimulus_id=stimulus_id
        )
        fn = tmpdir.join('catalog_test.hdf5')
        catalog.save(fn)
        loaded_catalog = psixy.catalog.load_catalog(fn)

        # Check that the loaded Catalog object is correct.
        assert catalog.n_stimuli == loaded_catalog.n_stimuli
        assert catalog.filepath == loaded_catalog.filepath
        assert catalog.common_path == loaded_catalog.common_path
        np.testing.assert_array_equal(
            catalog.stimulus_id, loaded_catalog.stimulus_id
        )

    def test_subset(self):
        """Test subset method of Catalog object."""
        # Provide optional stimulus id (valid input).
        stimulus_id = np.array([52, 7, 2, 33, 0, 5])
        path_list = [
            Path('rex/a.jpg'), Path('rex/b.jpg'), Path('rex/c.jpg'),
            Path('rex/d.jpg'), Path('rex/e.jpg'), Path('rex/f.jpg')
        ]
        catalog = psixy.catalog.Catalog(path_list, stimulus_id=stimulus_id)

        # Use integer indexing.
        idx = np.array([0, 2, 3, 5])
        path_list_sub = [
            Path('rex/a.jpg'), Path('rex/c.jpg'),
            Path('rex/d.jpg'), Path('rex/f.jpg')
        ]
        catalog_b = catalog.subset(idx)

        assert catalog_b.n_stimuli == 4
        assert catalog_b.filepath == path_list_sub
        assert catalog_b.common_path == Path('.')
        assert catalog_b.fullpath() == path_list_sub
        np.testing.assert_array_equal(
            catalog_b.stimulus_id, stimulus_id[idx]
        )

        # Use boolean indexing.
        idx = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
        catalog_b = catalog.subset(idx)

        assert catalog_b.n_stimuli == 4
        assert catalog_b.filepath == path_list_sub
        assert catalog_b.common_path == Path('.')
        assert catalog_b.fullpath() == path_list_sub
        np.testing.assert_array_equal(
            catalog_b.stimulus_id, stimulus_id[idx]
        )
