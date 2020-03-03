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

"""An example that tests the fit routine.

Synthetic behavioral data is generated using a known model. A new model
is fit to this data, blind to the true model parameters. The routine is
evaluated based on its ability to recover the true model parameters.

"""

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
    # Settings.
    n_sequence = 20
    n_epoch = 50
    task_idx = 2

    task_list, feature_matrix = psixy.task.shepard_hovland_jenkins_1961()
    task = task_list[2]

    encoder = psixy.models.Deterministic(task.stimulus_id, feature_matrix)

    class_id_list = np.array([0, 1], dtype=int)

    # Generate some sequences.
    s_seq_list = generate_stimulus_sequences(
        task.class_id, n_sequence, n_epoch
    )

    # Define known model.
    model_true = psixy.models.ALCOVE(feature_matrix, class_id_list, encoder)
    model_true.params['rho'] = 1.0
    model_true.params['tau'] = 1.0
    model_true.params['beta'] = 6.5
    model_true.params['phi'] = 2.0
    model_true.params['lambda_w'] = 0.03
    model_true.params['lambda_a'] = 0.0033

    # Simulate behavioral data with known model.
    prob_resp_attn = model_true.predict(s_seq_list, mode='all')
    # n_seq, n_trial, n_output
    n_trial = prob_resp_attn.shape[1]
    sampled_class = np.zeros([n_sequence, n_trial], dtype=int)
    for i_seq in range(n_sequence):
        for i_trial in range(n_trial):
            sampled_class[i_seq, i_trial] = np.random.choice(
                class_id_list, p=prob_resp_attn[i_seq, i_trial, :]
            )
    response_time_ms = 1000 * np.ones(sampled_class.shape)
    b_seq_list = psixy.sequence.AFCSequence(sampled_class, response_time_ms)

    # Fit a new model.
    model_inf = psixy.models.ALCOVE(feature_matrix, class_id_list, encoder)
    model_inf.params['rho'] = 1.0
    model_inf.params['tau'] = 1.0
    model_inf.params['beta'] = 6.5
    model_inf.params['phi'] = 2.0
    model_inf.params['lambda_w'] = 0.03
    model_inf.params['lambda_a'] = 0.0033

    loss_train = model_inf.fit(s_seq_list, b_seq_list, verbose=1)


def generate_stimulus_sequences(class_id_in, n_sequence, n_epoch):
    """Generate stimulus sequences."""
    np.random.seed(seed=245)
    n_stimuli = len(class_id_in)
    cat_idx = np.arange(n_stimuli, dtype=int)

    cat_idx_all = np.zeros([n_sequence, n_epoch * n_stimuli], dtype=int)
    for i_seq in range(n_sequence):
        curr_cat_idx = np.array([], dtype=int)
        for i_epoch in range(n_epoch):
            curr_cat_idx = np.hstack(
                [curr_cat_idx, np.random.permutation(cat_idx)]
            )
        cat_idx_all[i_seq, :] = curr_cat_idx

    # cat_idx_all = cat_idx_all[:, 0:2]  # TODO

    class_id = class_id_in[cat_idx_all]
    stimulus_sequence = psixy.sequence.StimulusSequence(
        cat_idx_all, class_id
    )
    return stimulus_sequence


if __name__ == "__main__":
    main()
