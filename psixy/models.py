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
    ALCOVE

Functions:
    load_model: Load a hdf5 file, that was saved with the `save`
        class method, as a psixy.models.CategoryLearningModel object.

Notes:
    State not initialized on creation of model object. Rather state
    is created whenever actual sequences are provided or a state is
    given.

    initialized values
    free parameters
    state

"""

from abc import ABCMeta, abstractmethod

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
    def fit(self, stimulus_seq, behavior_seq, group_id=None, weight=None):
        """Fit free parameters using supplied data."""
        pass

    @abstractmethod
    def evaluate(self, stimulus_seq, behavior_seq, group_id=None, weight=None):
        """Evaluate model using supplied data."""
        pass

    @abstractmethod
    def predict(self, stimulus_seq, group_id=None):
        """Predict behavior."""
        pass


class ALCOVE(CategoryLearningModel):
    """ALCOVE category learning model (Kruschke, 1992).

    Attributes:
        minkowski_rho: Parameter governing the Minkowski metric [1,inf]
        gen_specificity: the specificity of similarity [0,inf]
        resp_phi: decision consistency [0,inf]
        lr_assoc: learning rate of association weights [0,inf]
        lr_atten: learning rate of attention weights [0, inf]

    Methods:
        TODO

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
                hidden nodes in psychological space. Each column is
                the representation corresponding to a single stimulus.
                Each row corresponds to a distinct feature. TODO transpose
            class_id: A list of class ID's. The order of this list
                determines the output order of the model.
        """
        self.name = "ALCOVE"

        # Free Parameters
        self.minkowski_rho = 2
        self.gen_specificity = 1
        self.resp_phi = 1
        self.lr_assoc = .01
        self.lr_atten = .01

        # Model Variables
        # ALCOVE must know the locations of the RBFs at initialization.
        # NOTE: hidden nodes, observations stored as rows for distance
        # function.
        self.n_hidden = z.shape[0]
        self.n_dim = z.shape[1]
        self.z = z

        # ALCOVE must know the number of unique classes at initialization
        self.output_class_id = np.unique(class_id)
        # Add additional class that represents an answer that does not match
        # any class.
        # self.output_class_id = np.hstack(([0], self.output_class_id))
        self.n_class = self.output_class_id.shape[0]

        self.state = {'attention': [], 'association': []}

        if verbose > 0:
            print('ALCOVE initialized')
            print('  Number of Hidden Nodes: ', self.n_hidden)
            print('  Number of Output Classes: ', self.n_class)

    def _set_params(
            self, minkowski_rho=2, gen_specificity=1, resp_phi=1,
            lr_assoc=.01, lr_atten=.01):
        """Set Parameters."""
        # Free Parameters
        self.minkowski_rho = minkowski_rho
        self.gen_specificity = gen_specificity
        self.resp_phi = resp_phi
        self.lr_assoc = lr_assoc
        self.lr_atten = lr_atten

    def _init_attention(self, n_sequence):
        # NOTE: The original version of ALCOVE uses attention weights
        # that sum to one. However we use attention weights that sum to
        # the number of dimensions in order to preserve the standard case
        # of minkowski distance
        # TODO check if should use provided attention?
        return np.ones([n_sequence, self.n_dim])

    def _init_association(self, n_sequence):
        # NOTE: The original version of ALCOVE does not specify how to
        # initialize the association weights, so we are using a random
        # initialization about zero
        w = (
            np.random.rand(n_sequence, self.n_hidden, self.n_class) *
            .00001
        ) - .000005
        return w

    def _run(self, stimulus_seq, stateful=False):
        """Run model.

        Arguments:
            stimulus_seq: A psixy.sequence.StimulusSequence object.

        """
        n_sequence = stimulus_seq.n_sequence
        n_trial = stimulus_seq.n_trial
        response_probability = np.zeros((n_sequence, self.n_class, n_trial))
        probability_correct_response = np.zeros([n_sequence, n_trial])

        # Initialize state.
        if not stateful:
            self._init_state(n_sequence)

        correct_category_loc = self._determine_class_loc(stimulus_seq.class_id)

        # Iterate over trials
        for i_trial in range(stimulus_seq.n_trial):
            # Propagate input activation through the network.
            (act_hid, act_out) = self._forward_pass_vec(
                stimulus_seq.z[:, i_trial, :]
            )

            # Compute response probability.
            response_probability[:, :, i_trial] = self._response_rule_vec(
                act_out
            )

            # Update model state.
            self._update_state(
                stimulus_seq.z[:, i_trial, :], act_hid, act_out,
                correct_category_loc[:, :, i_trial],
                np.logical_and(
                    stimulus_seq.is_real[:, i_trial],
                    stimulus_seq.is_feedback[:, i_trial]
                )
            )

        # Determine the probability of correct response.
        probability_correct_response = response_probability[correct_category_loc]

        return response_probability, probability_correct_response

    def _forward_pass_vec(self, act_in):
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
            act_in: Input activations.
                shape = (n_seq, n_dim)

        Returns:
            act_hidden: Hidden unit activations.
                shape = (n_seq, n_hidden)
            act_out: Output unit activations.
                shape = (n_seq, n_class)

        """
        # TODO move this format to top.
        theta = {
            'rho': {'value': self.minkowski_rho},
            'tau': {'value': 1.},
            'gamma': {'value': 0.},
            'beta': {'value': self.gen_specificity}
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

    def _response_rule_vec(self, act_out):
        """Pass output_activation through soft-max response rule."""
        # Map output (category) node activations to response probabilities.
        # (Equation 3, pg. 24)
        response_probability = psixy.utils.softmax(
            self.resp_phi * act_out, axis=1
        )
        return response_probability

    def _update_state(
            self, act_in, act_hid, act_out, correct_category_loc,
            do_update):
        """Update state of model (i.e., weights) based on forward pass.

        ALCOVE assumes that each presentation of a training exemplar is
        followed by feedback indicating the correct response.
        Formally, this is implemented using gradient decent on squared-
        error.
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
        # delta_association_weights_0 = np.zeros(
        #     [n_sequence, self.n_hidden, self.n_class]
        # )
        # for s_seq in range(n_sequence):
        #     for j_hid in range(self.n_hidden):
        #         for k_out in range(self.n_class):
        #             delta_association_weights_0[s_seq, j_hid, k_out] = (
        #                 output_diff[s_seq, k_out] * act_hid[s_seq, j_hid]
        #             )
        # delta_association_weights_0 = self.lr_assoc * delta_association_weights_0

        # NOTE:
        # delta_association_weights: (n_seq, n_hidden, n_class)
        # output_diff: (n_seq, n_class)
        # act_hid: (n_seq, n_hidden)

        # delta_association_weights ~ (n_seq, n_hidden, n_class)
        delta_association_weights = np.zeros(
            [n_sequence, self.n_hidden, self.n_class]
        )
        for s_seq in range(n_sequence):
            delta_association_weights[s_seq] = np.matmul(
                np.expand_dims(act_hid[s_seq], axis=1),
                np.expand_dims(output_diff[s_seq], axis=0)
            )
        delta_association_weights = self.lr_assoc * delta_association_weights

        # Compute change for the attention weights
        # (Equation 6, pg. 24)
        # delta_attention_weights_0 = np.zeros([n_sequence, self.n_dim])
        # for s_seq in range(n_sequence):
        #     for i_dim in range(self.n_dim):
        #         outer_sum = 0
        #         for j_hid in range(self.n_hidden):
        #             inner_sumk = 0
        #             for k_out in range(self.n_class):
        #                 inner_sumk = inner_sumk + (output_diff[s_seq, k_out] * self.state['association'][s_seq, j_hid, k_out])
        #             outer_sum = outer_sum + (inner_sumk + act_hid[s_seq, j_hid] * self.gen_specificity * np.abs(self.z[j_hid, i_dim] - act_in[s_seq, i_dim]))
        #         delta_attention_weights_0[s_seq, i_dim] = outer_sum
        # delta_attention_weights_0 = -1 * self.lr_atten * delta_attention_weights_0

        # All sequences together.
        # delta_attention_weights = np.zeros([n_sequence, self.n_dim])
        # for i_dim in range(self.n_dim):
        #     outer_sum = 0
        #     for j_hid in range(self.n_hidden):
        #         inner_sumk = 0
        #         for k_out in range(self.n_class):
        #             inner_sumk = inner_sumk + (output_diff[:, k_out] * self.state['association'][:, j_hid, k_out])
        #         outer_sum = outer_sum + (inner_sumk + act_hid[:, j_hid] * self.gen_specificity * np.abs(self.z[j_hid, i_dim] - act_in[:, i_dim]))
        #     delta_attention_weights[:, i_dim] = outer_sum
        # delta_attention_weights = -1 * self.lr_atten * delta_attention_weights

        # Intermediate version.
        # delta_attention_weights = np.zeros([n_sequence, self.n_dim])
        # for i_dim in range(self.n_dim):
        #     outer_sum = 0
        #     for j_hid in range(self.n_hidden):
        #         inner_sumk = np.sum(output_diff * self.state['association'][:, j_hid, :], axis=1)
        #         outer_sum = outer_sum + (inner_sumk + act_hid[:, j_hid] * self.gen_specificity * np.abs(self.z[j_hid, i_dim] - act_in[:, i_dim]))
        #     delta_attention_weights[:, i_dim] = outer_sum
        # delta_attention_weights = -1 * self.lr_atten * delta_attention_weights

        # Intermediate version.
        # delta_attention_weights = np.zeros([n_sequence, self.n_dim])
        # inner_sumk = np.sum(np.expand_dims(output_diff, axis=1) * self.state['association'], axis=2)
        # for i_dim in range(self.n_dim):
        #     outer_sum = 0
        #     for j_hid in range(self.n_hidden):
        #         outer_sum = outer_sum + (inner_sumk[:, j_hid] + act_hid[:, j_hid] * self.gen_specificity * np.abs(self.z[j_hid, i_dim] - act_in[:, i_dim]))
        #     delta_attention_weights[:, i_dim] = outer_sum
        # delta_attention_weights = -1 * self.lr_atten * delta_attention_weights

        # Fastest version.
        # z: (n_hidden, n_dim)
        # act_in: (n_seq, n_dim)
        # act_hid: (n_seq, n_hidden)
        inner_sumk = np.sum(np.expand_dims(output_diff, axis=1) * self.state['association'], axis=2)
        # inner_sumk: (n_seq, n_hidden)
        part_1 = self.gen_specificity * np.abs(
            np.expand_dims(np.transpose(self.z), axis=0) - np.expand_dims(act_in, axis=2)
        )
        # part_1: (n_seq, n_dim, n_hidden)
        part_2 = np.expand_dims(act_hid, axis=1) * part_1
        # part_2: (n_seq, n_dim, n_hidden)
        part_3 = np.sum(np.expand_dims(inner_sumk, axis=1) + part_2, axis=2)
        # part_3: (n_seq, n_dim)
        delta_attention_weights = -1 * self.lr_atten * part_3

        # Move to test.
        # np.testing.assert_almost_equal(delta_attention_weights_0, delta_attention_weights)

        # Only update state for certain sequences.
        locs = np.logical_not(do_update)
        delta_association_weights[locs, :, :] = 0
        delta_attention_weights[locs, :] = 0

        # Update state.
        self.state['association'] = self.state['association'] + delta_association_weights
        # Update attention_weights (force attention weights to be possitive)
        attention_weights_new = np.maximum(0, self.state['attention'] + delta_attention_weights)
        attention_weights_new = project_attention(attention_weights_new)
        self.state['attention'] = attention_weights_new

    def _humble_teacher(self, output_activation, correct_category_loc):
        """Humble teacher values for each output node.

        (Equation 4b, pg. 24)
        """
        teacher_value = np.minimum(-1, output_activation)
        teacher_value[correct_category_loc] = np.maximum(1, output_activation[correct_category_loc])
        return teacher_value

    def _loglikelihood(
            self, stimulus_sequence, behavior_sequence, params):
        """Compute Log-likelihood of data given model."""
        # Set free parameters
        self._set_params(
            minkowski_rho=params[0], gen_specificity=params[1],
            resp_phi=params[2], lr_assoc=params[3], lr_atten=params[4]
        )

        (response_probability, probability_correct_response) = self._run(stimulus_sequence)

        loc_response_class = self._determine_class_loc(behavior_sequence.class_id)

        cap = 2.2204e-16
        probs = np.maximum(cap, response_probability[loc_response_class])
        # TODO
        ll = np.sum(np.log(probs))

        # for seq_idx in range(n_sequence):
        #     # Current sequence
        #     # trial_sequence = stimulus_sequence[seq_idx]
        #     # behavior_sequence = behavior_sequence[seq_idx]
        #     # n_trial = len(trial_sequence)

        #     # Clear model and run with new trial sequence
        #     self._init_state()
        #     (response_probability, probability_correct_response) = self.assimilate(trial_sequence, seq_idx)
        #     response_probability = np.transpose(response_probability)  # NOTE

        #     # Grab probability value that corresponds to subject's response
        #     row_idx = np.arange(n_trial)
        #     response_class_id = []
        #     for b in behavior_sequence:
        #         response_class_id.append(b.response_class_id)
        #     response_class_id = np.asarray(response_class_id)
        #     col_idx = coldcog.utils.id2idx(response_class_id, self.class_id_list)

        #     is_correct = np.max(response_probability, 1) == probability_correct_response
        #     n_correct = sum(is_correct)
        #     n_total = len(is_correct)
        #     acc_list[seq_idx] = n_correct / n_total

        #     probability_user_response = response_probability[row_idx, col_idx]

        #     # Compute log-likelihood of sequence
        #     probability_user_response = np.maximum(
        #         epsilon, probability_user_response
        #     )
        #     loss_list[seq_idx] = np.sum(np.log(probability_user_response))

        # avg_loss = coldcog.utils.logavg(loss_list)
        # avg_acc = np.mean(acc_list)
        # print("Avg LL: {0:.2f} Avg Acc: {6:.2f} | {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}".format(avg_loss, params[0], params[1], params[2], params[3], params[4], avg_acc))
        return ll

    def set_parameters(self, kwargs):
        """Set free parameters of model."""
        self._set_params(**kwargs)

    def fit(
            self, stimulus_sequence, behavior_sequence, n_restart=10,
            verbose=0):
        """Fit free parameters of model.

        Arguments:
            stimulus_sequence: A psixy.sequence.StimulusSequence object.
            behavior_sequence: A psixy.sequence.BehaviorSequence object.
            n_restart (optional): Number of indpendent restarts to use
                when fitting the free parameters.

        Returns:
            loss_train: The negative log-likelihood of the data given
                the fitted model.

        """
        # minkowski_rho       the minkowski metric space [1,inf]
        # gen_specificity     the specificity of similarity [0,inf]
        # resp_phi            decision consistency [0,inf]
        # lr_assoc            learning rate of association weights [0,inf]
        # lr_atten            learning rate of attention weights [0, inf]

        # bnds = ((1, None), (0, None), (0, None), (0, None), (0, None))
        bnds = ((1, 100), (0, 100), (0, 100), (0, 100), (0, 100))

        def obj_fun(params):
            return -self._loglikelihood(
                stimulus_sequence, behavior_sequence, params
            )

        params0 = (1., 1., 1., .01, .01)
        #   SLSQP L-BFGS-B
        res = scipy.optimize.minimize(
            obj_fun, params0, method='SLSQP', bounds=bnds,
            options={'disp': True}
        )
        params = res.x
        loss_train = res.fun

        # Set best params
        self._set_params(
            minkowski_rho=params[0], gen_specificity=params[1],
            resp_phi=params[2], lr_assoc=params[3], lr_atten=params[4]
        )
        return loss_train

    def evaluate(self, stimulus_sequence, behavior_sequence, verbose=0):
        """Evaluate."""
        params = np.array([
            self.minkowski_rho, self.gen_specificity, self.resp_phi,
            self.lr_assoc, self.lr_atten
        ])
        loss = -1 * self._loglikelihood(
            stimulus_sequence, behavior_sequence, params
        )
        return loss

    def _init_state(self, n_sequence=1):
        """Reset model state.

        Arguments:
            n_sequence: The number of sequences.
        """
        # Initialize connections in network
        self.state['attention'] = self._init_attention(
            n_sequence
        )
        self.state['association'] = self._init_association(
            n_sequence
        )

    def _determine_class_loc(self, class_id):
        """Determine output locations."""
        n_sequence = class_id.shape[0]
        n_trial = class_id.shape[1]
        loc_class = np.zeros(
            [n_sequence, n_trial, self.n_class], dtype=bool
        )
        for idx, i_class in enumerate(self.output_class_id):
            locs = np.equal(class_id, i_class)
            loc_class[locs, idx] = True
        loc_class = np.swapaxes(loc_class, 1, 2)
        return loc_class


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


def project_attention(attention):
    """Return projection of attention weights."""
    n_dim = attention.shape[1]

    # Check for weights that have zeroed out.
    cap = 2.2204e-16
    attention_sum = np.sum(attention, axis=1)
    locs = np.logical_not(np.isfinite(attention_sum))
    attention[locs, :] = 1

    attention_1 = np.sum(attention, axis=1, keepdims=True) / n_dim
    attention_proj = attention / attention_1
    # print(attention_proj)
    # if not np.isfinite(attention_1).all():
    #     print(attention_1)

    return attention_proj
