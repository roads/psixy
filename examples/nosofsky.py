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
    task_list, feature_matrix_list = psixy.task.nosofsky_1986()
    class_id_array = np.array([1, 2], dtype=int)

    s_seq_list_1, b_seq_list_1 = subject_1(task_list, class_id_array)
    s_seq_list_2, b_seq_list_2 = subject_2(task_list, class_id_array)

    # n_task = len(task_list)
    class_list = np.array([1, 2], dtype=int)

    idx_subj = 0
    idx_task = 0
    encoder_1 = psixy.models.Deterministic(
        task_list[0].stimulus_id, feature_matrix_list[idx_subj]
    )

    # Build ideal association matrix.
    dmy_idx = np.arange(task_list[idx_task].n_class)
    association_matrix = np.zeros([task_list[idx_task].n_stimuli, task_list[idx_task].n_class])
    for i_stim in range(task_list[idx_task].n_stimuli):
        curr_class_id = task_list[idx_task].class_id[i_stim]
        loc = np.equal(curr_class_id, class_list)
        association_matrix[i_stim, dmy_idx[loc]] = 1.0

    # Subject 1, dimensional model.
    model_s1 = psixy.models.GCM(
        feature_matrix_list[idx_subj], class_list, encoder_1,
        association=association_matrix
    )
    model_s1.params['rho'] = 2.0
    model_s1.params['tau'] = 2.0
    model_s1.params['gamma'] = 0.0
    model_s1.params['beta'] = 1.099
    model_s1.params['phi'] = 2.0
    model_s1.params['alpha'] = np.array([0.0, 1.0])
    model_s1.params['kappa'] = np.array([.444, .556])

    p = model_s1.predict(s_seq_list_1[idx_task])
    # TODO get -LL
    # TODO plot observed versus predicted


def subject_1(task_list, class_id_array):
    """Convert subject 1 summary statistics to data."""
    subj_id = 1

    # Dimensional.
    i_task = 0
    count_matrix = np.array([
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
    s_seq_0, b_seq_0 = create_sequence(
        count_matrix, task_list[i_task], class_id_array, subj_id
    )

    # Criss-cross.
    i_task = 1
    count_matrix = np.array([
        [7, 203],
        [27, 187],
        [183, 61],
        [206, 9],
        [58, 162],
        [73, 151],
        [152, 54],
        [187, 47],
        [193, 21],
        [147, 64],
        [46, 155],
        [66, 154],
        [212, 4],
        [149, 44],
        [35, 214],
        [13, 216],
    ])
    s_seq_1, b_seq_1 = create_sequence(
        count_matrix, task_list[i_task], class_id_array, subj_id
    )

    # Interior-exterior.
    i_task = 2
    count_matrix = np.array([
        [19, 238],
        [78, 162],
        [76, 181],
        [65, 219],
        [36, 216],
        [179, 72],
        [161, 65],
        [99, 159],
        [60, 189],
        [206, 62],
        [171, 75],
        [101, 150],
        [32, 238],
        [128, 126],
        [116, 157],
        [39, 223],
    ])
    s_seq_2, b_seq_2 = create_sequence(
        count_matrix, task_list[i_task], class_id_array, subj_id
    )

    # Diagonal.
    i_task = 3
    count_matrix = np.array([
        [226, 0],
        [231, 20],
        [165, 69],
        [92, 168],
        [214, 6],
        [206, 67],
        [109, 151],
        [44, 212],
        [208, 20],
        [108, 135],
        [31, 264],
        [12, 245],
        [209, 41],
        [71, 191],
        [13, 211],
        [3, 258],
    ])
    s_seq_3, b_seq_3 = create_sequence(
        count_matrix, task_list[i_task], class_id_array, subj_id
    )

    s_seq = [s_seq_0, s_seq_1, s_seq_2, s_seq_3]
    b_seq = [b_seq_0, b_seq_1, b_seq_2, b_seq_3]
    return s_seq, b_seq


def subject_2(task_list, class_id_array):
    """Convert subject 2 summary statistics to data."""
    subj_id = 2

    # Dimensional.
    i_task = 0
    count_matrix = np.array([
        [196, 7],
        [185, 4],
        [214, 9],
        [197, 8],
        [155, 27],
        [150, 35],
        [152, 55],
        [165, 51],
        [59, 120],
        [86, 116],
        [57, 135],
        [58, 170],
        [9, 193],
        [11, 171],
        [7, 195],
        [4, 199],
    ])
    s_seq_0, b_seq_0 = create_sequence(
        count_matrix, task_list[i_task], class_id_array, subj_id
    )

    # Criss-cross.
    i_task = 1
    count_matrix = np.array([
        [30, 190],
        [95, 161],
        [164, 26],
        [192, 14],
        [101, 139],
        [88, 128],
        [155, 70],
        [169, 62],
        [176, 51],
        [131, 118],
        [62, 152],
        [75, 147],
        [199, 23],
        [122, 101],
        [22, 199],
        [18, 219],
    ])
    s_seq_1, b_seq_1 = create_sequence(
        count_matrix, task_list[i_task], class_id_array, subj_id
    )

    # Interior-exterior
    i_task = 2
    count_matrix = np.array([
        [14, 132],
        [48, 81],
        [54, 116],
        [24, 118],
        [38, 106],
        [95, 40],
        [89, 55],
        [41, 99],
        [61, 83],
        [131, 16],
        [122, 45],
        [33, 98],
        [40, 106],
        [70, 71],
        [34, 101],
        [15, 124],
    ])
    s_seq_2, b_seq_2 = create_sequence(
        count_matrix, task_list[i_task], class_id_array, subj_id
    )

    # Diagonal
    i_task = 3
    count_matrix = np.array([
        [216, 3],
        [199, 15],
        [139, 58],
        [41, 195],
        [203, 5],
        [193, 36],
        [53, 178],
        [14, 215],
        [219, 12],
        [151, 67],
        [36, 194],
        [6, 237],
        [189, 24],
        [75, 143],
        [6, 233],
        [3, 241],
    ])
    s_seq_3, b_seq_3 = create_sequence(
        count_matrix, task_list[i_task], class_id_array, subj_id
    )

    s_seq = [s_seq_0, s_seq_1, s_seq_2, s_seq_3]
    b_seq = [b_seq_0, b_seq_1, b_seq_2, b_seq_3]
    return s_seq, b_seq


def create_sequence(count_matrix, task, class_id_array, subj_id):
    """Convert summary statistics in [1] to sequences."""
    stimulus_id_array = task.catalog.stimulus_id
    n_stim = count_matrix.shape[0]
    n_class = count_matrix.shape[1]

    seq_stimulus_id = np.array([], dtype=int)
    seq_class_id = np.array([], dtype=int)
    seq_response_id = np.array([], dtype=int)
    for i_stim in range(n_stim):
        for i_class in range(n_class):
            n_trial = count_matrix[i_stim, i_class]
            curr_seq_stimulus_id = (
                stimulus_id_array[i_stim] * np.ones([n_trial], dtype=int)
            )
            curr_seq_class_id = (
                task.class_id[i_stim] * np.ones([n_trial], dtype=int)
            )
            curr_seq_response_id = (
                class_id_array[i_class] * np.ones([n_trial], dtype=int)
            )

            seq_stimulus_id = np.hstack(
                (seq_stimulus_id, curr_seq_stimulus_id)
            )
            seq_class_id = np.hstack(
                (seq_class_id, curr_seq_class_id)
            )
            seq_response_id = np.hstack(
                (seq_response_id, curr_seq_response_id)
            )

    seq_stimulus_id = np.expand_dims(seq_stimulus_id, axis=0)
    seq_class_id = np.expand_dims(seq_class_id, axis=0)
    seq_response_id = np.expand_dims(seq_response_id, axis=0)

    rt_ms_fake = 1000 * np.ones(seq_stimulus_id.shape)

    group_id = subj_id * np.ones(
        np.expand_dims(seq_stimulus_id, axis=2).shape, dtype=int
    )
    s_seq = psixy.sequence.StimulusSequence(seq_stimulus_id, seq_class_id)
    b_seq = psixy.sequence.AFCSequence(seq_response_id, rt_ms_fake)

    return s_seq, b_seq


if __name__ == "__main__":
    main()
