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

"""Module of category learning models.

Classes:
    CategoryLearningModel: Abstract base class for the simplest
        category learning task involving one stimulus and one response.
    ALCOVE:

Functions:
    load_model: Load a hdf5 file, that was saved with the `save`
        class method, as a psixy.models.CategoryLearningModel object.
    exp_similarity: Exponential similarity.
    project_attention: Function for projecting attention.

Notes:
    State not initialized on creation of model object. Rather state
    is created whenever actual sequences are provided or a state is
    given.

    initialized values
    free parameters
    state

"""

from abc import ABCMeta, abstractmethod
import copy
import warnings

import numpy as np
import scipy

import psixy.utils


class CategoryLearningModel(object):
    """Abstract base class for category learning models.

    Methods:
        fit: Fit free parameters of a category learning model using
            observations.
        evaluate: Evaluate the likelihood of some observations given
            a category learning model.
        predict: Predict behavior given a category learning model.

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(
            self, stimulus_seq, behavior_seq, group_id=None, weight=None,
            verbose=0):
        """Fit free parameters using supplied data."""
        pass

    @abstractmethod
    def evaluate(
            self, stimulus_seq, behavior_seq, group_id=None, weight=None,
            verbose=0):
        """Evaluate model using supplied data."""
        pass

    @abstractmethod
    def predict(self, stimulus_seq, group_id=None, verbose=0):
        """Predict behavior."""
        pass


class ALCOVE(CategoryLearningModel):
    """ALCOVE category learning model (Kruschke, 1992).

    Attributes:
        params: Dictionary for the model's free parameters.
            rho: Parameter governing the Minkowski metric [1,inf]
            tau: Parameter governing the shape of the RBF [1,inf]
            beta: the specificity of similarity [1,inf]
            phi: decision consistency [0,inf]
            lambda_w: learning rate of association weights [0,inf]
            lambda_a: learning rate of attention weights [0, inf]
        state: Dictionary for the model's state.
            attention:
            association:

    Methods:
        fit:
        evaluate:
        predict:

    This code implements the following version:
    "In the simplest version of ALCOVE, there is a hidden node placed
    at the position of every traning exemplar (pg. 23)."

    Alternatively, one could implement the following version:
    "In a more complicated version, discussed at the end of the
    article, hidden nodes ares scattered randomly across the space,
    forming a covering map of the input space (pg. 23)."

    References:
    [1] Kruschke, J. K. (1992). ALCOVE: an exemplar-based connectionist
        model of category learning. Psychological review, 99(1), 22-44.
        http://dx.doi.org/10.1037/0033-295X.99.1.22.

    """

    def __init__(self, z, class_id, verbose=0):
        """Initialize.

        Arguments:
            z: A two-dimension array denoting the location of the
                hidden nodes in psychological space. Each row is
                the representation corresponding to a single stimulus.
                Each column corresponds to a distinct feature
                dimension.
            class_id: A list of class ID's. The order of this list
                determines the output order of the model.
            verbose (optional): Verbosity of output.

        """
        # Model constants.
        self.name = "ALCOVE"
        # At initialization, ALCOVE must know the locations of the RBFs and
        # the number of unique classes.
        self.z = z
        self.n_hidden = z.shape[0]
        self.n_dim = z.shape[1]
        self.output_class_id = np.unique(class_id)
        self.n_class = self.output_class_id.shape[0]

        # Settings.
        self.attention_mode = 'classic'

        # Free parameters.
        self.params = {
            'rho': 2,
            'tau': 1,
            'beta': 1,
            'phi': 1,
            'lambda_w': .001,
            'lambda_a': .001
        }
        self._params = {
            'rho': {'bounds': [1, 100]},
            'tau': {'bounds': [1, 100]},
            'beta': {'bounds': [1, 100]},
            'phi': {'bounds': [0, 100]},
            'lambda_w': {'bounds': [0, 10]},
            'lambda_a': {'bounds': [0, 10]}
        }

        # State variables.
        self.state = {
            'init': {
                'attention': self._default_attention(),
                'association': self._default_association()
            },
            'attention': [],
            'association': []
        }

        if verbose > 0:
            print('ALCOVE initialized')
            print('  Input dimension: ', self.n_dim)
            print('  Number of hidden nodes: ', self.n_hidden)
            print('  Number of output classes: ', self.n_class)

    def fit(
            self, stimulus_sequence, behavior_sequence, options=None,
            verbose=0):
        """Fit free parameters of model.

        Arguments:
            stimulus_sequence: A psixy.sequence.StimulusSequence object.
            behavior_sequence: A psixy.sequence.BehaviorSequence object.
            options (optional): A dictionary of optimization options.
                n_restart (10): Number of indpendent restarts to use
                    when fitting the free parameters.

        Returns:
            loss_train: The negative log-likelihood of the data given
                the fitted model.

        """
        def obj_fun(params):
            return -self._log_likelihood_opt(
                params, stimulus_sequence, behavior_sequence
            )

        loss_train = self.evaluate(stimulus_sequence, behavior_sequence)
        beat_initialization = False

        if verbose > 0:
            print('Starting configuration:')
            print('  loss: {0:.2f}'.format(loss_train))
            print('')

        for i_restart in range(options['n_restart']):
            params0 = self._rand_param()
            bnds = self._get_bnds()
            # SLSQP L-BFGS-B
            res = scipy.optimize.minimize(
                obj_fun, params0, method='SLSQP', bounds=bnds,
                options={'disp': False}
            )

            if verbose > 1:
                print('Restart {0}'.format(i_restart))
                print('  loss: {0:.2f} | iterations: {1}'.format(res.fun, res.nit))
                print('  exit mode: {0} | {1}'.format(res.status, res.message))
                print('')

            if res.fun < loss_train:
                beat_initialization = True
                params_opt = res.x
                loss_train = res.fun

        # Set free parameters using best run.
        if beat_initialization:
            self._set_params(params_opt)
            if verbose > 0:
                print('Final')
                print('  loss: {0:.2f}'.format(loss_train))
        else:
            if verbose > 0:
                print('Final')
                print('  Did not beat starting configuration.')

        return loss_train

    def evaluate(self, stimulus_sequence, behavior_sequence, verbose=0):
        """Evaluate."""
        loss = -1 * self._log_likelihood(
            self.params, stimulus_sequence, behavior_sequence
        )
        if verbose > 0:
            print("loss: {0:.2f}".format(loss))
        return loss

    def predict(
            self, stimulus_sequence, group_id=None, mode='correct',
            stateful=False, verbose=0):
        """Predict behavior."""
        prob_response, prob_response_correct = self._run(
            self.params, stimulus_sequence, stateful=stateful
        )
        if mode == 'correct':
            res = prob_response_correct
        elif mode == 'all':
            res = prob_response
        else:
            raise ValueError(
                'Undefined option {0} for mode argument.'.format(str(mode))
            )
        return res

    def _init_state(self, n_sequence=1):
        """Initialize model state.

        Arguments:
            n_sequence: The number of sequences.
        """
        self.state['attention'] = np.repeat(
            self.state['init']['attention'], n_sequence, axis=0
        )
        self.state['association'] = np.repeat(
            self.state['init']['association'], n_sequence, axis=0
        )

    def _default_attention(self):
        """Initialize attention weights using default settings."""
        # NOTE: The original version of ALCOVE uses attention weights
        # that sum to one. However we use attention weights that sum to
        # the number of dimensions in order to preserve the standard case
        # of minkowski distance
        if self.attention_mode == 'classic':
            a = np.ones([1, self.n_dim]) / self.n_dim
        else:
            a = np.ones([1, self.n_dim])
        return a

    def _default_association(self):
        """Initialize association weights using default settings.

        "The association weights were initialized at zero, reflecting
        the notion that before training there should be no associations
        between any exemplars and particular categories (pg. 27)."

        """
        # w = (
        #     np.random.rand(1, self.n_hidden, self.n_class) *
        #     .00001
        # ) - .000005
        w = np.zeros([1, self.n_hidden, self.n_class])
        return w

    def _set_params(self, params_opt):
        """Set free parameters from optimizer format."""
        self.params['rho'] = params_opt[0]
        self.params['tau'] = params_opt[1]
        self.params['beta'] = params_opt[2]
        self.params['phi'] = params_opt[3]
        self.params['lambda_w'] = params_opt[4]
        self.params['lambda_a'] = params_opt[5]

    def _get_params(self, params_opt):
        """Set free parameters from optimizer format."""
        params_opt = [
            self.params['rho'],
            self.params['tau'],
            self.params['beta'],
            self.params['phi'],
            self.params['lambda_w'],
            self.params['lambda_a']
        ]
        return params_opt

    def _run(self, params_local, stimulus_seq, stateful=False):
        """Run model.

        Arguments:
            params_local: Locally scoped free parameters.
            stimulus_seq: A psixy.sequence.StimulusSequence object.
            stateful (optional): Boolean that controls initialization
                of state parameters.

        """
        n_sequence = stimulus_seq.n_sequence
        n_trial = stimulus_seq.n_trial
        prob_response = np.zeros((n_sequence, self.n_class, n_trial))

        # Initialize state.
        if not stateful:
            self._init_state(n_sequence)

        correct_category_loc = determine_class_loc(
            stimulus_seq.class_id, self.output_class_id
        )

        delta_association_batch = np.zeros(
            [n_sequence, self.n_hidden, self.n_class]
        )
        delta_attention_batch = np.zeros([n_sequence, self.n_dim])
        # Iterate over trials
        for i_trial in range(stimulus_seq.n_trial):
            # Propagate input activation through the network.
            (act_hid, act_out) = self._forward_pass(
                params_local, stimulus_seq.z[:, i_trial, :]
            )

            # Compute response probability.
            prob_response[:, :, i_trial] = self._response_rule(
                params_local, act_out
            )

            # Compute state updates.
            (delta_attention, delta_association) = self._compute_update(
                params_local, stimulus_seq.z[:, i_trial, :], act_hid, act_out,
                correct_category_loc[:, :, i_trial],
                np.logical_and(
                    stimulus_seq.is_real[:, i_trial],
                    stimulus_seq.is_feedback[:, i_trial]
                )
            )

            # TODO temporary batch updating.
            # if (i_trial > 0) and (np.mod(i_trial + 1, 8) == 0):
            #     # Apply state updates.
            #     self.state['attention'] = np.maximum(
            #         0, self.state['attention'] + delta_attention_batch
            #     )
            #     self.state['association'] = (
            #         self.state['association'] + delta_association_batch
            #     )
            #     delta_attention_batch = np.zeros([n_sequence, self.n_dim])
            #     delta_association_batch = np.zeros(
            #         [n_sequence, self.n_hidden, self.n_class]
            #     )
            # else:
            #     delta_attention_batch = delta_attention_batch + delta_attention
            #     delta_association_batch = delta_association_batch + delta_association

            # Apply state updates.
            self.state['association'] = (
                self.state['association'] + delta_association
            )
            self.state['attention'] = np.maximum(
                0, self.state['attention'] + delta_attention
            )
            # self.state['attention'] = project_attention(
            #     self.state['attention'] + delta_attention,
            #     attention_mode=self.attention_mode
            # )

        # Determine the probability the learner responds correctly.
        prob_response_correct = copy.copy(prob_response)
        incorrect_loc = np.logical_not(correct_category_loc)
        prob_response_correct[incorrect_loc] = 0
        prob_response_correct = np.sum(prob_response_correct, axis=1)

        return prob_response, prob_response_correct

    def _forward_pass(self, params_local, act_in):
        """Propagate activity forward through the network.

        "For a given input stimulus, each hidden node is activated
        according to the psychological similarity of the stimulus to
        the exemplar at the position of the hidden node (pg. 23)."

        "Psychologically, the specificity of a hidden node indicates
        the overall cognitive discriminability or memorability of
        the corresponding exemplar (pg. 23)."

        "The region of stimulus space that significantly activates a
        hidden node will be loosely referred to as the node's
        receptive field (pg. 23)."

        Arguments:
            params_local: Locally scoped free parameters.
            act_in: Input activations.
                shape = (n_seq, n_dim)

        Returns:
            act_hidden: Hidden unit activations.
                shape = (n_seq, n_hidden)
            act_out: Output unit activations.
                shape = (n_seq, n_class)

        """
        theta = {
            'rho': {'value': params_local['rho']},
            'tau': {'value': params_local['tau']},
            'gamma': {'value': 0.},
            'beta': {'value': params_local['beta']}
        }
        # Compute the activation of hidden exemplar nodes based on similarity.
        # (Equation 1, pg. 23)
        # act_in: (n_seq, n_dim)
        # hidden_nodes: (n_hidden, n_dim)

        # Take advantage of vectorization.
        # act_in: (n_seq, n_dim, 1)
        # hidden_nodes: (1, n_dim, n_hidden)
        # attention: (n_seq, n_dim, 1)
        act_in = np.expand_dims(act_in, axis=2)
        z = np.expand_dims(np.transpose(self.z), axis=0)
        attention = np.expand_dims(self.state["attention"], axis=2)
        act_hidden = exp_similarity(
            act_in, z, theta, attention
        )

        # Compute the activation of the output (category) nodes.
        # (Equation 2, pg. 24)
        # NOTE: association: (n_seq, n_hidden_node, n_class)
        # NOTE: Equivalent to doing the following for each sequence separately:
        # (1, n_hidden) dot (n_hidden, n_class) => (1, n_class)
        act_out = (
            np.expand_dims(act_hidden, axis=2) * self.state["association"]
        )
        act_out = np.sum(act_out, axis=1)

        return (act_hidden, act_out)

    def _response_rule(self, params_local, act_out):
        """Pass output_activation through soft-max response rule.

        Arguments:
            params_local: Locally scoped free parameters.
            act_out:

        """
        # Map output (category) node activations to response probabilities.
        # (Equation 3, pg. 24)
        prob_response = psixy.utils.softmax(
            params_local['phi'] * act_out, axis=1
        )
        return prob_response

    def _compute_update(
            self, params_local, act_in, act_hid, act_out,
            correct_category_loc, do_update):
        """Compute state updates for model (i.e., weights).

        ALCOVE assumes that each presentation of a training exemplar is
        followed by feedback indicating the correct response.
        Formally, this is implemented using gradient decent on squared-
        error.

        Arguments:
            params_local: Locally scoped free parameters.
            act_in:
            act_hid:
            act_out:
            correct_category_loc:
            do_update:

        Returns:
            delta_attention: Updates to attention weights.
            delta_association: Updates to association weights.

        """
        n_sequence = act_in.shape[0]

        # Humble teacher values for each output node
        # (Equation 4b, pg. 24)
        teacher_value = self._humble_teacher(
            act_out, correct_category_loc
        )

        # Compute change for the association weights
        # (Equation 5, pg. 24)
        output_diff = teacher_value - act_out
        # output_diff: (n_seq, n_class)

        # Known slow answer.
        # n_sequence = act_in.shape[0]
        # delta_association_0 = np.zeros(
        #     [n_sequence, self.n_hidden, self.n_class]
        # )
        # for s_seq in range(n_sequence):
        #     for j_hid in range(self.n_hidden):
        #         for k_out in range(self.n_class):
        #             delta_association_0[s_seq, j_hid, k_out] = (
        #                 output_diff[s_seq, k_out] * act_hid[s_seq, j_hid]
        #             )
        # delta_association_0 = params_local['lambda_w'] * delta_association_0

        # NOTE:
        # delta_association: (n_seq, n_hidden, n_class)
        # output_diff: (n_seq, n_class)
        # act_hid: (n_seq, n_hidden)

        # delta_association ~ (n_seq, n_hidden, n_class)
        # Fastest implementation.
        delta_association = np.zeros(
            [n_sequence, self.n_hidden, self.n_class]
        )
        for s_seq in range(n_sequence):
            delta_association[s_seq] = np.matmul(
                np.expand_dims(act_hid[s_seq], axis=1),
                np.expand_dims(output_diff[s_seq], axis=0)
            )
        delta_association = params_local['lambda_w'] * delta_association

        # Move to test.
        # np.testing.assert_almost_equal(delta_association_0, delta_association)

        # Compute change for the attention weights
        # (Equation 6, pg. 24)

        # Known slow answer.
        # delta_attention_0 = np.zeros([n_sequence, self.n_dim])
        # for s_seq in range(n_sequence):
        #     for i_dim in range(self.n_dim):
        #         outer_sum = 0
        #         for j_hid in range(self.n_hidden):
        #             inner_sumk = 0
        #             for k_out in range(self.n_class):
        #                 inner_sumk = inner_sumk + (output_diff[s_seq, k_out] * self.state['association'][s_seq, j_hid, k_out])
        #             outer_sum = outer_sum + (inner_sumk * act_hid[s_seq, j_hid] * params_local['beta'] * np.abs(self.z[j_hid, i_dim] - act_in[s_seq, i_dim]))
        #         delta_attention_0[s_seq, i_dim] = outer_sum
        # delta_attention_0 = -1 * params_local['lambda_a'] * delta_attention_0

        # All sequences together.
        # delta_attention = np.zeros([n_sequence, self.n_dim])
        # for i_dim in range(self.n_dim):
        #     outer_sum = 0
        #     for j_hid in range(self.n_hidden):
        #         inner_sumk = 0
        #         for k_out in range(self.n_class):
        #             inner_sumk = inner_sumk + (output_diff[:, k_out] * self.state['association'][:, j_hid, k_out])
        #         outer_sum = outer_sum + (inner_sumk * act_hid[:, j_hid] * params_local['beta'] * np.abs(self.z[j_hid, i_dim] - act_in[:, i_dim]))
        #     delta_attention[:, i_dim] = outer_sum
        # delta_attention = -1 * params_local['lambda_a'] * delta_attention

        # Intermediate version.
        # delta_attention = np.zeros([n_sequence, self.n_dim])
        # for i_dim in range(self.n_dim):
        #     outer_sum = 0
        #     for j_hid in range(self.n_hidden):
        #         inner_sumk = np.sum(output_diff * self.state['association'][:, j_hid, :], axis=1)
        #         outer_sum = outer_sum + (inner_sumk * act_hid[:, j_hid] * params_local['beta'] * np.abs(self.z[j_hid, i_dim] - act_in[:, i_dim]))
        #     delta_attention[:, i_dim] = outer_sum
        # delta_attention = -1 * params_local['lambda_a'] * delta_attention

        # Intermediate version.
        # delta_attention = np.zeros([n_sequence, self.n_dim])
        # inner_sumk = np.sum(np.expand_dims(output_diff, axis=1) * self.state['association'], axis=2)
        # for i_dim in range(self.n_dim):
        #     outer_sum = 0
        #     for j_hid in range(self.n_hidden):
        #         outer_sum = outer_sum + (inner_sumk[:, j_hid] * act_hid[:, j_hid] * params_local['beta'] * np.abs(self.z[j_hid, i_dim] - act_in[:, i_dim]))
        #     delta_attention[:, i_dim] = outer_sum
        # delta_attention = -1 * params_local['lambda_a'] * delta_attention

        # Intermediate version
        # delta_attention = np.zeros([n_sequence, self.n_dim])
        # inner_sumk = np.sum(np.expand_dims(output_diff, axis=1) * self.state['association'], axis=2)
        # part_1 = params_local['beta'] * np.abs(
        #     np.expand_dims(np.transpose(self.z), axis=0) - np.expand_dims(act_in, axis=2)
        # )
        # for i_dim in range(self.n_dim):
        #     outer_sum = 0
        #     for j_hid in range(self.n_hidden):
        #         outer_sum = outer_sum + (inner_sumk[:, j_hid] * act_hid[:, j_hid] * part_1[:, i_dim, j_hid])
        #     delta_attention[:, i_dim] = outer_sum
        # delta_attention = -1 * params_local['lambda_a'] * delta_attention

        # Fastest version.
        # z: (n_hidden, n_dim)
        # act_in: (n_seq, n_dim)
        # act_hid: (n_seq, n_hidden)
        delta_attention = np.zeros([n_sequence, self.n_dim])
        inner_sumk = np.sum(np.expand_dims(output_diff, axis=1) * self.state['association'], axis=2)
        part_1 = params_local['beta'] * np.abs(
            np.expand_dims(np.transpose(self.z), axis=0) - np.expand_dims(act_in, axis=2)
        )
        # part_1: (n_seq, n_dim, n_hidden)
        part_2 = inner_sumk * act_hid
        # part_2: (n_seq, n_hidden)
        part_3 = np.expand_dims(part_2, axis=1) * part_1
        outer_sum = np.sum(part_3, axis=2)
        delta_attention = -1 * params_local['lambda_a'] * outer_sum

        # Move to test.
        # np.testing.assert_almost_equal(delta_attention_0, delta_attention)
        # print('here')

        # inner_sumk = np.sum(np.expand_dims(output_diff, axis=1) * self.state['association'], axis=2)
        # # inner_sumk: (n_seq, n_hidden)
        # part_1 = params_local['beta'] * np.abs(
        #     np.expand_dims(np.transpose(self.z), axis=0) - np.expand_dims(act_in, axis=2)
        # )
        # # part_1: (n_seq, n_dim, n_hidden)
        # part_2 = np.expand_dims(act_hid, axis=1) * part_1
        # # part_2: (n_seq, n_dim, n_hidden)
        # part_3 = np.sum(np.expand_dims(inner_sumk, axis=1) + part_2, axis=2)
        # # part_3: (n_seq, n_dim)
        # delta_attention = -1 * params_local['lambda_a'] * part_3

        # Only update state for certain sequences.
        locs = np.logical_not(do_update)
        delta_association[locs, :, :] = 0
        delta_attention[locs, :] = 0

        # if np.sum(np.logical_not(np.isfinite(delta_association))) > 0:
        #     print('here')

        # if np.sum(np.logical_not(np.isfinite(delta_attention))) > 0:
        #     print('here')

        return delta_attention, delta_association

    def _humble_teacher(self, output_activation, correct_category_loc):
        """Humble teacher values for each output node.

        (Equation 4b, pg. 24)
        """
        teacher_value = np.minimum(-1, output_activation)
        teacher_value[correct_category_loc] = np.maximum(1, output_activation[correct_category_loc])
        return teacher_value

    def _log_likelihood(
            self, params_local, stimulus_sequence, behavior_sequence):
        """Compute the log-likelihood of the data given model."""
        (prob_response, prob_response_correct) = self._run(
            params_local, stimulus_sequence
        )

        loc_response = determine_class_loc(
            behavior_sequence.class_id, self.output_class_id
        )

        cap = 2.2204e-16
        probs = np.maximum(cap, prob_response[loc_response])
        # TODO sequence weighted version
        # ll = np.sum(np.log(probs))
        ll = np.mean(np.log(probs))

        # if np.isnan(ll):
            # print('here')
            # ll = -np.inf

        # avg_loss = coldcog.utils.logavg(loss_list)
        # avg_acc = np.mean(acc_list)
        # print("Avg LL: {0:.2f} Avg Acc: {6:.2f} | {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}".format(avg_loss, params[0], params[1], params[2], params[3], params[4], avg_acc))

        # Compute accuracy. TODO
        # is_correct = np.max(prob_response, 1) == prob_response_correct
        # n_correct = sum(is_correct)
        # n_total = len(is_correct)
        # acc_list[seq_idx] = n_correct / n_total

        return ll

    def _log_likelihood_opt(
            self, params_opt, stimulus_sequence, behavior_sequence):
        """Compute the log-likelihood of the data given model.

        Parameters are structured for using scipy.optimize.minimize.
        """
        params_local = {
            'rho': params_opt[0],
            'tau': params_opt[1],
            'beta': params_opt[2],
            'phi': params_opt[3],
            'lambda_w': params_opt[4],
            'lambda_a': params_opt[5]
        }
        ll = self._log_likelihood(
            params_local, stimulus_sequence, behavior_sequence
        )
        return ll

    def _rand_param(self):
        """Randomly sample parameter setting."""
        param_0 = []
        for bnd_set in self._get_bnds():
            start = bnd_set[0]
            width = bnd_set[1] - bnd_set[0]
            param_0.append(start + (np.random.rand(1)[0] * width))
        return param_0

    def _get_bnds(self):
        """Return bounds."""
        bnds = [
            self._params['rho']['bounds'],
            self._params['tau']['bounds'],
            self._params['beta']['bounds'],
            self._params['phi']['bounds'],
            self._params['lambda_w']['bounds'],
            self._params['lambda_a']['bounds'],
        ]
        return bnds


def exp_similarity(z_q, z_r, theta, attention):
    """Exponential family similarity kernel.

    Arguments:
        z_q: A set of embedding points.
            shape = (n_trial, n_dim)
        z_r: A set of embedding points.
            shape = (n_trial, n_dim)
        theta: A dictionary of algorithm-specific parameters
            governing the similarity kernel.
        attention: The weights allocated to each dimension
            in a weighted minkowski metric.
            shape = (n_trial, n_dim)

    Returns:
        The corresponding similarity between rows of embedding
            points.
            shape = (n_trial,)

    """
    # Algorithm-specific parameters governing the similarity kernel.
    rho = theta['rho']["value"]
    tau = theta['tau']["value"]
    gamma = theta['gamma']["value"]
    beta = theta['beta']["value"]

    # Weighted Minkowski distance.
    d_qref = (np.abs(z_q - z_r))**rho
    d_qref = np.multiply(d_qref, attention)
    d_qref = np.sum(d_qref, axis=1)**(1. / rho)

    # Exponential family similarity kernel.
    sim_qr = np.exp(np.negative(beta) * d_qref**tau) + gamma
    return sim_qr


def determine_class_loc(class_id, output_class_id):
        """Determine output locations.

        Arguments:
            class_id: (n_seq, n_trial)
            output_class_id: (n_class,)

        Returns:
            loc_class: The corresponding location.
                (n_seq, n_class, n_trial)

        """
        n_class = len(output_class_id)

        n_sequence = class_id.shape[0]
        n_trial = class_id.shape[1]
        loc_class = np.zeros(
            [n_sequence, n_trial, n_class], dtype=bool
        )
        for idx, i_class in enumerate(output_class_id):
            locs = np.equal(class_id, i_class)
            loc_class[locs, idx] = True
        loc_class = np.swapaxes(loc_class, 1, 2)
        return loc_class


def project_attention(attention_in, attention_mode='classic'):
    """Return projection of attention weights.

    Arguments:
        attention_in: Incoming attention weights.

    Returns;
        attention_out: Projected attention weights.
    """
    # Settings.
    cap = 2.2204e-16

    # Make positive.
    attention_0 = np.maximum(0, attention_in)

    # Project.
    n_dim = attention_in.shape[1]
    attention_1 = np.sum(attention_0, axis=1, keepdims=True)
    # Do not divide by zero.
    locs_zero = np.equal(attention_1, 0)
    attention_1[locs_zero] = cap
    if attention_mode == 'classic':
        attention_out = (attention_0 / attention_1)
    else:
        attention_out = n_dim * (attention_0 / attention_1)

    # Check for weights that have zeroed out and reset them.
    attention_sum = np.sum(attention_out, axis=1)
    locs_bad = np.equal(attention_sum, 0)
    attention_out[locs_bad, :] = 1

    attention_sum = np.sum(attention_out, axis=1)
    if np.sum(np.equal(attention_sum, 0)) > 0:
        print('here')

    if np.sum(np.logical_not(np.isfinite(attention_out))) > 0:
        print('here')

    # attention_sum = np.sum(attention_0, axis=1)
    # locs = np.logical_not(np.isfinite(attention_sum))
    # attention[locs, :] = 1

    return attention_out
