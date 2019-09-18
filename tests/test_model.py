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

import psixy.models
import psixy.sequence


class TestFunctions:
    """Test stand-alone functions in the models module."""

    def test_determine_class_loc(self):
        """Test `determine_class_loc`."""
        # class_id: (n_seq, n_trial)
        # output_class_id: (n_class,)
        output_class_id = np.array([1, 5, 7])
        class_id = np.array([
            [1, 1, 5, 1, 5, 7, 5, 7, 7, 7],
            [1, 1, 5, 1, 5, 7, 5, 7, 7, 7],
            [1, 1, 5, 1, 5, 7, 5, 7, 7, 7],
            [1, 1, 5, 1, 5, 7, 5, 7, 7, 7]
        ], dtype=int)

        # (n_seq, n_class, n_trial)
        correct_category_loc_desired = np.array([
            [
                [ True,  True, False, True, False, False, False, False, False, False],
                [False, False, True, False,  True, False,  True, False, False, False],
                [False, False, False, False, False,  True, False,  True,  True, True]
            ],[
                [ True,  True, False,  True, False, False, False, False, False, False],
                [False, False,  True, False,  True, False,  True, False, False, False],
                [False, False, False, False, False,  True, False,  True,  True,  True]
            ],[
                [ True,  True, False,  True, False, False, False, False, False, False],
                [False, False,  True, False,  True, False,  True, False, False, False],
                [False, False, False, False, False,  True, False,  True,  True,  True]
            ],[
                [ True,  True, False,  True, False, False, False, False, False, False],
                [False, False,  True, False,  True, False,  True, False, False, False],
                [False, False, False, False, False,  True, False,  True,  True,  True]
            ]
        ])

        correct_category_loc = psixy.models.determine_class_loc(class_id, output_class_id)
        np.testing.assert_array_equal(
            correct_category_loc_desired, correct_category_loc
        )
