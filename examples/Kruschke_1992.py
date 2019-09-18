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

"""Reproduce Figure 5 from Kruschke, 1992.

Each datum shows the probability of selecting the correct category,
averaged across the eight exemplars within an epoch. For both graphs,
the response mapping constant was set to phi = 2.0, the specificity was
fixed at c = 6.5, and the learning rate for association weights was
lambda_w = 0.03. In Figure 5A, there was no attention learning
(lambda_a = 0.0), and it can be seen that Type II is learned much too
slowly. In Figure 5B, the attention-learning rate was raised to
lambda_a = 0.0033, and consequently Type II was learned second fastest,
as observed in human data. (pg. 27)

References:
    [1] Kruschke, J. K. (1992). ALCOVE: an exemplar-based connectionist model
        of category learning. Psychological review, 99(1), 22-44.
        http://dx.doi.org/10.1037/0033-295X.99.1.22.

"""

import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import psixy.catalog
import psixy.models
import psixy.sequence


def main():
    """Execute script."""
    # Settings.
    n_sequence = 10
    n_epoch = 50
    fp_fig = Path('examples', 'kruschke_1992_fig_5.pdf')
    task_label_list = ['I', 'II', 'III', 'IV', 'V', 'VI']

    catalog, feature_matrix = generate_catalog()
    class_list = np.array([0, 1], dtype=int)

    # Model without attention.
    model_0 = psixy.models.ALCOVE(feature_matrix, class_list)
    model_0.params['rho'] = 1.0
    model_0.params['tau'] = 1.0
    model_0.params['beta'] = 6.5
    model_0.params['phi'] = 2.0
    model_0.params['lambda_w'] = 0.03
    model_0.params['lambda_a'] = 0.0

    # Model with attention.
    model_attn = psixy.models.ALCOVE(feature_matrix, class_list)
    model_attn.params['rho'] = 1.0
    model_attn.params['tau'] = 1.0
    model_attn.params['beta'] = 6.5
    model_attn.params['phi'] = 2.0
    model_attn.params['lambda_w'] = 0.03
    model_attn.params['lambda_a'] = 0.0033

    accuracy_epoch_0 = np.zeros([catalog.n_task, n_epoch])
    accuracy_epoch_attn = np.zeros([catalog.n_task, n_epoch])
    for i_task in range(catalog.n_task):
        stimulus_sequence = generate_stimulus_sequences(
            catalog.task(task_idx=i_task), feature_matrix, n_sequence, n_epoch
        )

        prob_correct_attn = model_attn.predict(stimulus_sequence, mode='correct')
        accuracy_epoch_attn[i_task, :] = epoch_analysis(
            prob_correct_attn, catalog.n_stimuli
        )

        prob_correct_0 = model_0.predict(stimulus_sequence, mode='correct')
        accuracy_epoch_0[i_task, :] = epoch_analysis(
            prob_correct_0, catalog.n_stimuli
        )

    # Plot figure.
    # Create color map of green and red shades, but don't use middle
    # yellow.
    cmap = matplotlib.cm.get_cmap('RdYlGn')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=catalog.n_task + 1)
    color_array = cmap(norm(np.flip(range(catalog.n_task + 1))))
    # Drop yellow color.
    locs = np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool)
    color_array = color_array[locs, :]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax = plt.subplot(1, 2, 1)
    for i_task in range(catalog.n_task):
        ax.plot(
            accuracy_epoch_0[i_task, :],
            marker='o', markersize=3,
            c=color_array[i_task, :],
            label='{0}'.format(task_label_list[i_task])
        )
    ax.set_xlabel('epoch')
    ax.set_ylabel('Pr(correct)')
    ax.set_title('Without Attention')
    ax.legend(title='Category Type')

    ax = plt.subplot(1, 2, 2)
    for i_task in range(catalog.n_task):
        ax.plot(
            accuracy_epoch_attn[i_task, :],
            marker='o', markersize=3,
            c=color_array[i_task, :],
            label='{0}'.format(task_label_list[i_task])
        )
    ax.set_xlabel('epoch')
    ax.set_ylabel('Pr(correct)')
    ax.set_title('With Attention')
    ax.legend(title='Category Type')

    plt.tight_layout()
    plt.savefig(
        os.fspath(fp_fig), format='pdf', bbox_inches='tight', dpi=400
    )


def epoch_analysis(prob_correct, n_stimuli):
    """Epoch analysis."""
    n_sequence = prob_correct.shape[0]
    n_epoch = int(prob_correct.shape[1] / n_stimuli)

    # Use reshape in order to get epoch-level accuracy averages.
    prob_correct_2 = np.reshape(
        prob_correct, (n_sequence, n_stimuli, n_epoch), order='F'
    )
    seq_epoch_avg = np.mean(prob_correct_2, axis=1)

    # Average over all sequences.
    epoch_avg = np.mean(seq_epoch_avg, axis=0)
    return epoch_avg


def generate_catalog():
    """Generate catalog."""
    stimulus_id = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    filepath = [
        '0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6jpg', '7.jpg'
    ]
    class_id = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 1]
    ])
    class_id = np.transpose(class_id)
    catalog = psixy.catalog.Catalog(stimulus_id, filepath, class_id=class_id)

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

    return catalog, feature_matrix


def generate_stimulus_sequences(class_id_in, feature_matrix, n_sequence, n_epoch):
    """Generate stimulus sequences."""
    # TODO
    n_stimuli = 8
    cat_idx = np.arange(n_stimuli, dtype=int)

    cat_idx_all = np.zeros([n_sequence, n_epoch * n_stimuli], dtype=int)
    for i_seq in range(n_sequence):
        curr_cat_idx = np.array([], dtype=int)
        for i_epoch in range(n_epoch):
            curr_cat_idx = np.hstack([curr_cat_idx, np.random.permutation(cat_idx)])
        cat_idx_all[i_seq, :] = curr_cat_idx

    z = feature_matrix[cat_idx_all,:]
    class_id = class_id_in[cat_idx_all]
    stimulus_sequence = psixy.sequence.StimulusSequence(z, class_id)
    return stimulus_sequence


if __name__ == "__main__":
    main()
