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

"""Reproduce Figure 5 and Figure 14 from Kruschke, 1992.

Figure 5 Description (pg. 27):
Each datum shows the probability of selecting the correct category,
averaged across the eight exemplars within an epoch. For both graphs,
the response mapping constant was set to phi = 2.0, the specificity was
fixed at c = 6.5, and the learning rate for association weights was
lambda_w = 0.03. In Figure 5A, there was no attention learning
(lambda_a = 0.0), and it can be seen that Type II is learned much too
slowly. In Figure 5B, the attention-learning rate was raised to
lambda_a = 0.0033, and consequently Type II was learned second fastest,
as observed in human data.

Figure 14 Description (pg. 37-38):
It is now shown that ALCOVE can exhibit three-stage learning of high-
frequency exceptions to rules in a highly simplified abstract analogue
of the verb-acquisition situation. For this demonstration, the input
stimuli are distributed over two continuously varying dimensions as
shown in Figure 13.
...
ALCOVE was applied to the structure in Figure 13, using 14 hidden nodes
and parameter values near the values used to fit the Medin et al.
(1982) data: phi = 1.00, lambda_w = 0.025, c = 3.50, and
lambda_a = 0.010. Epoch updating was used, with each rule exemplar
occurring once per epoch and each exceptional case occurring four times
per epoch, for a total of 20 patterns per epoch. (The same qualitative
effects are produced with trial-by-trial updating, with superimposed
trial-by-trial "sawteeth," what Plunket and Marchman, 1991, called
micro U-shaped learning.) The results are shown in Figure 14.

References:
    [1] Kruschke, J. K. (1992). ALCOVE: an exemplar-based connectionist
        model of category learning. Psychological Review, 99(1), 22-44.
        http://dx.doi.org/10.1037/0033-295X.99.1.22.

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
    fp_fig_5 = Path('examples', 'kruschke_1992_fig_5.pdf')
    fp_fig_14 = Path('examples', 'kruschke_1992_fig_14.pdf')

    create_figure_5(fp_fig_5)
    create_figure_14(fp_fig_14)


def create_figure_5(fp_fig_5):
    """Create figure 5."""
    n_sequence = 10
    n_epoch = 50

    task_list, feature_matrix = psixy.task.shepard_hovland_jenkins_1961()
    n_task = len(task_list)
    class_list = np.array([0, 1], dtype=int)

    encoder = psixy.models.Deterministic(
        task_list[0].stimulus_id, feature_matrix
    )

    # Model without attention.
    model_0 = psixy.models.ALCOVE(feature_matrix, class_list, encoder)
    model_0.params['rho'] = 1.0
    model_0.params['tau'] = 1.0
    model_0.params['beta'] = 6.5
    model_0.params['phi'] = 2.0
    model_0.params['lambda_w'] = 0.03
    model_0.params['lambda_a'] = 0.0

    # Model with attention.
    model_attn = psixy.models.ALCOVE(feature_matrix, class_list, encoder)
    model_attn.params['rho'] = 1.0
    model_attn.params['tau'] = 1.0
    model_attn.params['beta'] = 6.5
    model_attn.params['phi'] = 2.0
    model_attn.params['lambda_w'] = 0.03
    model_attn.params['lambda_a'] = 0.0033

    accuracy_epoch_0 = np.zeros([n_task, n_epoch])
    accuracy_epoch_attn = np.zeros([n_task, n_epoch])
    for i_task in range(n_task):
        curr_task = task_list[i_task]
        stimulus_sequence = generate_fig5_stimulus_sequences(
            curr_task.class_id, n_sequence, n_epoch
        )

        prob_correct_attn = model_attn.predict(
            stimulus_sequence, mode='correct'
        )
        accuracy_epoch_attn[i_task, :] = epoch_analysis_correct(
            prob_correct_attn, curr_task.n_stimuli
        )

        prob_correct_0 = model_0.predict(stimulus_sequence, mode='correct')
        accuracy_epoch_0[i_task, :] = epoch_analysis_correct(
            prob_correct_0, curr_task.n_stimuli
        )

    # Plot figure.
    # Create color map of green and red shades, but don't use middle
    # yellow.
    cmap = matplotlib.cm.get_cmap('RdYlGn')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=n_task + 2)
    color_array = cmap(norm(np.flip(range(n_task + 2))))
    # Drop middle yellow colors.
    locs = np.array([1, 1, 1, 0, 0, 1, 1, 1], dtype=bool)
    color_array = color_array[locs, :]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax = plt.subplot(1, 2, 1)
    for i_task in range(n_task):
        curr_task = task_list[i_task]
        ax.plot(
            accuracy_epoch_0[i_task, :],
            marker='o', markersize=3,
            c=color_array[i_task, :],
            label='{0}'.format(curr_task.name)
        )
    ax.set_xlabel('epoch')
    ax.set_ylabel('Pr(correct)')
    ax.set_title('Without Attention')
    ax.legend(title='Category Type')

    ax = plt.subplot(1, 2, 2)
    for i_task in range(n_task):
        curr_task = task_list[i_task]
        ax.plot(
            accuracy_epoch_attn[i_task, :],
            marker='o', markersize=3,
            c=color_array[i_task, :],
            label='{0}'.format(curr_task.name)
        )
    ax.set_xlabel('epoch')
    ax.set_ylabel('Pr(correct)')
    ax.set_title('With Attention')
    ax.legend(title='Category Type')

    plt.tight_layout()
    plt.savefig(
        os.fspath(fp_fig_5), format='pdf', bbox_inches='tight', dpi=400
    )


def create_figure_14(fp_fig_14):
    """Create figure 14."""
    n_sequence = 50
    n_epoch = 50
    n_stimuli_per_epoch = 20

    task, feature_matrix, stimulus_label = psixy.task.kruschke_rules_and_exceptions()
    class_list = np.array([0, 1], dtype=int)

    encoder = psixy.models.Deterministic(task.stimulus_id, feature_matrix)

    # Model without attention.
    model_attn = psixy.models.ALCOVE(feature_matrix, class_list, encoder)
    model_attn.params['rho'] = 1.0
    model_attn.params['tau'] = 1.0
    model_attn.params['beta'] = 3.5
    model_attn.params['phi'] = 1.0
    model_attn.params['lambda_w'] = 0.025
    model_attn.params['lambda_a'] = 0.010

    stimulus_sequence = generate_fig_14_stimulus_sequences(
        task.class_id, n_sequence, n_epoch
    )

    prob_correct_attn = model_attn.predict(stimulus_sequence, mode='correct')
    accuracy_epoch_attn = epoch_analysis_stimulus(
        stimulus_sequence, prob_correct_attn, task.stimulus_id,
        n_stimuli_per_epoch
    )

    # Plot figure.
    marker_list = [
        'o', 'o', 'o', 'o', 'o', 'o', 's',
        'o', 's', 's', 's', 's', 's', 's',
    ]
    color_array = np.vstack([
        np.repeat(
            np.array([[0.07197232, 0.54071511, 0.28489043, .1]]), 6, axis=0
        ),
        np.array([[0.8899654, 0.28673587, 0.19815456, 1.]]),
        np.array([[0.4295271, 0.75409458, 0.39146482, 1.]]),
        np.repeat(np.array([[0.64705882, 0., 0.14901961, .1]]), 6, axis=0),
    ])

    fig, ax = plt.subplots(figsize=(4, 4))

    ax = plt.subplot(1, 1, 1)
    for i_stim in range(task.n_stimuli):
        ax.plot(
            accuracy_epoch_attn[i_stim, :],
            marker=marker_list[i_stim], markersize=3,
            c=color_array[i_stim, :],
            label='{0}'.format(stimulus_label[i_stim])
        )
    ax.set_xlabel('epoch')
    ax.set_ylabel('Pr(correct)')
    ax.set_title('Rules and Exceptions')
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.fspath(fp_fig_14), format='pdf', bbox_inches='tight', dpi=400
    )


def epoch_analysis_correct(prob_correct, n_stimuli_per_epoch):
    """Epoch analysis."""
    n_sequence = prob_correct.shape[0]
    n_epoch = int(prob_correct.shape[1] / n_stimuli_per_epoch)

    # Use reshape in order to get epoch-level accuracy averages.
    prob_correct_2 = np.reshape(
        prob_correct, (n_sequence, n_stimuli_per_epoch, n_epoch), order='F'
    )
    seq_epoch_avg = np.mean(prob_correct_2, axis=1)

    # Average over all sequences.
    epoch_avg = np.mean(seq_epoch_avg, axis=0)
    return epoch_avg


def epoch_analysis_stimulus(
        s_seq, prob_response, stim_id_list, n_stimuli_per_epoch):
    """Epoch analysis."""
    n_sequence = prob_response.shape[0]
    n_epoch = int(prob_response.shape[1] / n_stimuli_per_epoch)
    n_stimuli = len(stim_id_list)

    stimulus_id_2 = np.reshape(
        s_seq.stimulus_id, (n_sequence, n_stimuli_per_epoch, n_epoch),
        order='F'
    )
    # Use reshape in order to get epoch-level accuracy averages.
    prob_response_2 = np.reshape(
        prob_response, (n_sequence, n_stimuli_per_epoch, n_epoch), order='F'
    )
    # prob_response_2: (n_seq, n_trial, n_epoch)

    epoch_avg = np.zeros([n_stimuli, n_epoch])
    for i_epoch in range(n_epoch):
        prob_response_2_epoch = prob_response_2[:, :, i_epoch]
        stimulus_id_2_epoch = stimulus_id_2[:, :, i_epoch]
        for i_stim in range(n_stimuli):
            locs = np.equal(stimulus_id_2_epoch, stim_id_list[i_stim])
            epoch_avg[i_stim, i_epoch] = np.mean(prob_response_2_epoch[locs])

    return epoch_avg


def generate_fig5_stimulus_sequences(
        class_id_in, n_sequence, n_epoch):
    """Generate stimulus sequences."""
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

    class_id = class_id_in[cat_idx_all]
    stimulus_sequence = psixy.sequence.StimulusSequence(
        cat_idx_all, class_id
    )

    return stimulus_sequence


def generate_fig_14_stimulus_sequences(class_id_in, n_sequence, n_epoch):
    """Generate stimulus sequences."""
    n_stimuli = len(class_id_in)
    epoch_cat_idx = np.hstack([
        np.arange(n_stimuli, dtype=int),
        np.array([6, 6, 6, 7, 7, 7], dtype=int)
    ])

    n_stimuli_per_epoch = len(epoch_cat_idx)

    cat_idx_all = np.zeros(
        [n_sequence, n_epoch * n_stimuli_per_epoch], dtype=int
    )
    for i_seq in range(n_sequence):
        curr_cat_idx = np.array([], dtype=int)
        for i_epoch in range(n_epoch):
            curr_cat_idx = np.hstack(
                [curr_cat_idx, np.random.permutation(epoch_cat_idx)]
            )
        cat_idx_all[i_seq, :] = curr_cat_idx

    class_id = class_id_in[cat_idx_all]
    stimulus_sequence = psixy.sequence.StimulusSequence(
        cat_idx_all, class_id
    )
    return stimulus_sequence


if __name__ == "__main__":
    main()
