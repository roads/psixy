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

"""Nosofsky example."""

import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import psixy.catalog
import psixy.task
import psixy.models
import psixy.sequence


def main():
    """Execute script."""
    task_list, feature_matrix = psixy.task.nosofsky_1986()

    n_task = len(task_list)
    class_list = np.array([1, 2], dtype=int)

    encoder = psixy.models.Deterministic(
        task_list[0].stimulus_id, feature_matrix
    )

    # Subject 1, dimensional model.
    subj_1 = psixy.models.GCM(feature_matrix, class_list, encoder)
    model.params['rho'] = 2.0
    model.params['tau'] = 2.0
    model.params['gamma'] = 0.0
    model.params['beta'] = 1.099
    model.params['phi'] = 2.0
    model.params['alpha'] = np.array([0.0, 1.0])
    model.params['kappa'] = np.array([.444, .556])

    s_seq, b_seq = transfer_data()
    # TODO get -LL
    # TODO plot observed versus predicted


def transfer_data():
    """Create transfer data sequences."""
    # Subject 1 stimulus responses.
    # Dimensional.
    np.array([
        [213, 4],
        [253, 1],
        [192, 2],
        [218, 1],
        [185, 57],
        [193, 47],
        [187, 40],
        [162, 36],
        [24, 194],
        [33, 198],
        [31, 190],
        [40, 181],
        [0, 204],
        [0, 235],
        [0, 220],
        [0, 258],
    ])

    s_seq = None
    b_seq = None
    return s_seq, b_seq
