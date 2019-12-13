
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
    project_attention: Function for projecting attention.

Notes:
    State not initialized on creation of model object. Rather state
    is created whenever actual sequences are provided or a state is
    given.

    initialized values
    free parameters
    state

Todo:
    * different optimizers
    * loss term and humble teacher
    * WeightedMinkowski what should weight default be?
    * Use lazy weight initialization that doesn't require n_sequence 
        on initialization.
    * Makes sure gradients are being computed correctly.
    * self.n_class => self.n_output

"""

from abc import ABCMeta, abstractmethod
import collections
import copy
import time
import warnings

import numpy as np
import scipy
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, RNN
from tensorflow import constant_initializer
from tensorflow.keras import Model
from tensorflow.keras.constraints import NonNeg
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.keras.constraints import Constraint

import psixy.utils

# tf.keras.backend.set_floatx('float64')


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
    def fit(self, stimulus_seq, behavior_seq, options=None, verbose=0):
        """Fit free parameters using supplied data."""
        pass

    @abstractmethod
    def evaluate(
            self, stimulus_seq, behavior_seq, verbose=0):
        """Evaluate model using supplied data."""
        pass

    @abstractmethod
    def predict(self, stimulus_seq, group_id=None, verbose=0):
        """Predict behavior."""
        pass


class ALCOVE2(CategoryLearningModel):
    """ALCOVE category learning model (Kruschke, 1992).

    Attributes:
        params: Dictionary for the model's free parameters.
            rho: Parameter governing the Minkowski metric [1,inf]
            tau: Parameter governing the shape of the RBF [1,inf]
            beta: the specificity of similarity [1,inf]
            gamma: Governs degree to which similarity fades to
                indifference.
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

    This model can be used to instantiate either a 'training exemplar'
    or 'covering map' variant of ALCOVE. Just pass in the locations of
    the the hidden nodes.

    Training exemplar variant:
    "In the simplest version of ALCOVE, there is a hidden node placed
    at the position of every traning exemplar (pg. 23)."

    Covering map variant:
    "In a more complicated version, discussed at the end of the
    article, hidden nodes ares scattered randomly across the space,
    forming a covering map of the input space (pg. 23)."

    Notes:
        The model is implemented using TensorFlow, so that gradients
        for state updates can be computed for arbitrary model parameter
        settings. The original model presented in the paper assumed
        rho=1, tau=1, and gamma=0.

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
        # the unique classes.
        self.z = z.astype(dtype=K.floatx())
        self.n_hidden = z.shape[0]
        self.n_dim = z.shape[1]
        self.output_class_id = self._check_class_id(class_id)
        self.n_class = self.output_class_id.shape[0]
        # Map IDs. TODO
        self.class_map = {}
        for i_class in range(self.n_class):
            self.class_map[self.output_class_id[i_class]] = i_class

        # Settings.
        self.attention_mode = 'classic'

        # Free parameters.
        self.params = {
            'rho': 2.0,
            'tau': 1.0,
            'beta': 1.0,
            'gamma': 0.0,
            'phi': 1.0,
            'lambda_a': .001,
            'lambda_w': .001
        }
        self._params = {
            'rho': {'bounds': [1, 1]},
            'tau': {'bounds': [1, 1]},
            'beta': {'bounds': [1, 100]},
            'gamma': {'bounds': [0, 0]},
            'phi': {'bounds': [0, 100]},
            'lambda_a': {'bounds': [0, 10]},
            'lambda_w': {'bounds': [0, 10]}
        }

        # State variables.
        # self.state = {
        #     'init': {
        #         'attention': self._default_attention(),
        #         'association': self._default_association()
        #     },
        #     'attention': [],
        #     'association': []
        # }

        if verbose > 0:
            print('ALCOVE initialized')
            print('  Input dimension: ', self.n_dim)
            print('  Number of hidden nodes: ', self.n_hidden)
            print('  Number of output classes: ', self.n_class)

    def _check_class_id(self, class_id):
        """Check `class_id` argument."""
        if not len(np.unique(class_id)) == len(class_id):
            raise ValueError(
                'The argument `class_id` must contain all unique'
                ' integers.'
            )

        return class_id

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
        # TODO implement working fit method.
    
        # Settings.
        max_epoch = 10
        batch_size = stimulus_sequence.n_sequence  # TODO
        buffer_size = 10000

        # Prepare dataset.
        inputs = self._prepare_inputs(stimulus_sequence)
        targets = self._prepare_targets(behavior_sequence)
        seq_dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = seq_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

        theta = self._get_theta(self.params)  # TODO
        model, rnn, initial_states = self._build_tf_model(
            theta, stimulus_sequence.n_trial, batch_size
        )
        model_parameters = list(theta.values())
        model_parameters = model_parameters[4:]  # TODO

        optimizer = tf.keras.optimizers.Adam()

        # @tf.function
        def fix_gradient(grads):
            for idx,  grad in enumerate(grads):
                grads[idx] = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
            return grads

        # @tf.function
        def train_step(input_batch, target_batch):
            with tf.GradientTape(watch_accessed_variables=False) as model_tape:
                model_tape.watch(model_parameters)
                logit_response_batch = model(input_batch)
                loss = tf.keras.losses.categorical_crossentropy(
                    target_batch, logit_response_batch, from_logits=True
                )
                # prob_response = tf.math.softmax(logit_response_batch, axis=2)
                # prob_response_correct = prob_response * target_batch
                # prob_response_correct = tf.reduce_sum(prob_response_correct, axis=2)
                # loss = prob_response_correct
                # loss = tf.reduce_sum(tf.math.log(prob_response_correct))
                # loss = tf.reduce_mean(loss, axis=1)
                # loss = tf.reduce_mean(loss, axis=0)
                loss = tf.reduce_mean(loss)
            parameter_gradients = model_tape.gradient(loss, model_parameters)
            for idx, grad in enumerate(parameter_gradients):
                if grad is not None:
                    print(
                        '    {0}: {1:.6f} {2:.6g}'.format(
                            model_parameters[idx].name,
                            model_parameters[idx].numpy(), grad.numpy()
                        )
                    )
            parameter_gradients = fix_gradient(parameter_gradients)
            optimizer.apply_gradients(
                zip(parameter_gradients, model_parameters)
            )
            return loss

        loss_train = np.inf
        for i_epoch in range(max_epoch):
            start = time.time()
            for (i_batch, (input_batch, target_batch)) in enumerate(dataset):
                # Initialize the hidden state at the start of every batch
                # since using stateful model.
                rnn.reset_states(states=initial_states)
                loss = train_step(input_batch, target_batch)

                # if i_batch % 100 == 0:
                template = 'Epoch {} Batch {} Loss {:.4f}'
                print(template.format(i_epoch + 1, i_batch, loss))

            # print('Epoch {} Loss {:.4f}'.format(i_epoch + 1, loss))
            # print('Time taken for 1 epoch {:.2f} sec\n'.format(time.time() - start))

        # ===================================================================

        # loss_train = self.evaluate(stimulus_sequence, behavior_sequence)
        # beat_initialization = False

        # if verbose > 0:
        #     print('Starting configuration:')
        #     print('  loss: {0:.2f}'.format(loss_train))
        #     print('')

        # for i_restart in range(options['n_restart']):
        #     # Random initialization of variables (with bounds and trainable settings appropriately set).  TODO

        #     # Perform gradient descent.

        #     if verbose > 1:
        #         print('Restart {0}'.format(i_restart))
        #         print('  loss: {0:.2f} | iterations: {1}'.format(res.fun, res.nit))
        #         print('  exit mode: {0} | {1}'.format(res.status, res.message))
        #         print('')

        #     if res.fun < loss_train:
        #         beat_initialization = True
        #         params_opt = res.x
        #         loss_train = res.fun

        # # Set free parameters using best run.
        # if beat_initialization:
        #     self._set_params(params_opt)
        #     if verbose > 0:
        #         print('Final')
        #         print('  loss: {0:.2f}'.format(loss_train))
        # else:
        #     if verbose > 0:
        #         print('Final')
        #         print('  Did not beat starting configuration.')

        return loss_train

    def evaluate(self, stimulus_sequence, behavior_sequence, verbose=0):
        """Evaluate."""
        loss = self._loss(
            self.params, stimulus_sequence, behavior_sequence
        )
        if verbose > 0:
            print("loss: {0:.2f}".format(loss))
        return loss

    def predict(
            self, stimulus_sequence, group_id=None, mode='all',
            stateful=False, verbose=0):
        """Predict behavior.

        Arguments:
            stimulus_sequence. A psixy.sequence.StimulusSequence
                object.
            group_id: TODO
            mode: Determines which response probabilities to return.
                Can be 'all' or 'correct'. If 'all', then response
                probabilities for all output categories will be
                returned. If 'correct', only returns the response
                probabilities for the the correct category.
            stateful: TODO
            verbose: Integer indicating verbosity of outout.

        Returns:
            res: A Tensorflow.Tensor object containing response
                probabilities. TODO

        """
        # Settings
        batch_size = stimulus_sequence.n_sequence  # TODO handle smaller batches
        buffer_size = 10000

        # Prepare dataset.
        inputs = self._prepare_inputs(stimulus_sequence)
        seq_dataset = tf.data.Dataset.from_tensor_slices((inputs))
        # TODO right now, have to have same batch size for all batches.
        dataset = seq_dataset.batch(batch_size, drop_remainder=True)

        theta = self._get_theta(self.params)
        model, _, _ = self._build_tf_model(
            theta, stimulus_sequence.n_trial, batch_size
        )

        for input_batch in dataset.take(1):
            logit_response_batch = model(input_batch)

        logit_response = logit_response_batch

        # Convert logits to probabilities.
        prob_response = tf.math.softmax(logit_response, axis=2)

        if mode == 'correct':
            stimuli_labels = self._convert_labels(stimulus_sequence.class_id)
            stimuli_labels_one_hot = tf.one_hot(
                stimuli_labels, self.n_class, axis=2
            )
            prob_response_correct = prob_response * stimuli_labels_one_hot
            prob_response_correct = tf.reduce_sum(prob_response_correct, axis=2)
            res = prob_response_correct
        elif mode == 'all':
            res = prob_response
        else:
            raise ValueError(
                'Undefined option {0} for mode argument.'.format(str(mode))
            )
        return res.numpy()

    def _loss(self, params_local, stimulus_sequence, behavior_sequence):
        """Compute the negative log-likelihood of the data given model."""
        prob_response = self._run(params_local, stimulus_sequence)

        behavior_labels = self._convert_labels(behavior_sequence.class_id)
        behavior_labels_one_hot = tf.one_hot(
            behavior_labels, self.n_class, axis=2
        )
        # TODO use tf categorical cross entropy
        prob_response_correct = tf.reduce_sum(
            prob_response * behavior_labels_one_hot, axis=2
        )
        loss_all = -1 * tf.log(prob_response_correct)

        # Scalar loss (average within a sequence, then across sequences.)
        loss_train = tf.reduce_mean(loss_all, axis=1)
        loss_train = tf.reduce_mean(loss_train)
        loss_train = loss_train.numpy()

        if np.isnan(loss_train):
            loss_train = np.inf

        return loss_train

    def _prepare_inputs(self, stimulus_sequence):
        """Prepare inputs for TensorFlow model."""
        # [n_sequence, n_trial, n_dim]
        stimulus_z = stimulus_sequence.z.astype(dtype=K.floatx())
        # [n_sequence, n_output]
        stimulus_labels = self._convert_labels(stimulus_sequence.class_id)
        stimulus_labels_one_hot = tf.one_hot(
            stimulus_labels, self.n_class, axis=2
        )
        inputs = {'0': stimulus_z, '1': stimulus_labels_one_hot}
        return inputs

    def _prepare_targets(self, behavior_sequence):
        """Prepare targets."""
        # [n_sequence, n_output]
        behavior_labels = self._convert_labels(behavior_sequence.class_id)
        behavior_labels_one_hot = tf.one_hot(
            behavior_labels, self.n_class, axis=2
        )
        targets = behavior_labels_one_hot
        return targets

    def _build_tf_model(self, theta, n_timestep, batch_size):
        """Build tensorflow RNN model."""
        print('Build Model')
        z_hidden = self.z.astype(dtype=K.floatx())
        n_output = self.n_class

        n_hidden = z_hidden.shape[0]
        n_dim = z_hidden.shape[1]
        attention = np.ones([batch_size, n_dim], dtype=K.floatx()) / n_dim
        association = np.zeros(
            [batch_size, n_hidden, n_output], dtype=K.floatx()
        )
        initial_states = [attention, association]
        cell = ALCOVECell(theta, z_hidden, n_output, batch_size=batch_size)
        rnn = tf.keras.layers.RNN(
            cell, return_sequences=True, stateful=True
        )
        # TODO what shape information is actually necessary?
        # TODO Is timestep still necessary?
        inp_1 = tf.keras.Input(batch_shape=(batch_size, n_timestep, n_dim))  # shape=(None, n_dim), batch_size=batch_size
        inp_2 = tf.keras.Input(batch_shape=(batch_size, n_timestep, n_output))  # shape=(None, n_output), batch_size=batch_size
        output = rnn(NestedInput(z_in=inp_1, one_hot_label=inp_2))
        model = tf.keras.models.Model([inp_1, inp_2], output)

        rnn.reset_states(states=initial_states)

        return model, rnn, initial_states

    def _convert_labels(self, labels):
        """Convert labels."""
        labels_conv = np.zeros(labels.shape, dtype=int)
        for key, value in self.class_map.items():
            locs = np.equal(labels, key)
            labels_conv[locs] = value
        return labels_conv

    def _get_theta(self, params_local):
        """Return theta."""
        # TODO this isn't quite right.
        theta = {
            'rho': tf.constant(
                params_local['rho'], dtype=K.floatx(), name='rho'
            ),
            'tau': tf.constant(
                params_local['tau'], dtype=K.floatx(), name='tau'
            ),
            'beta': tf.constant(
                params_local['beta'], dtype=K.floatx(), name='beta'
            ),
            'gamma': tf.constant(
                params_local['gamma'], dtype=K.floatx(), name='gamma'
            ),
            # 'rho': tf.Variable(
            #     initial_value=params_local['rho'],
            #     trainable=False, constraint=GreaterEqualThan(min_value=1.),
            #     dtype=K.floatx(), name='rho'
            # ),
            # 'tau': tf.Variable(
            #     initial_value=params_local['tau'],
            #     trainable=False, constraint=GreaterEqualThan(min_value=1.),
            #     dtype=K.floatx(), name='tau'
            # ),
            # 'beta': tf.Variable(
            #     initial_value=params_local['beta'],
            #     trainable=False, constraint=GreaterEqualThan(min_value=1.),
            #     dtype=K.floatx(), name='beta'
            # ),
            # 'gamma': tf.Variable(
            #     initial_value=params_local['gamma'],
            #     trainable=False, constraint=NonNeg(),
            #     dtype=K.floatx(), name='gamma'
            # ),
            'phi': tf.Variable(
                initial_value=params_local['phi'],
                trainable=True, constraint=NonNeg(),
                dtype=K.floatx(), name='phi'
            ),
            'lambda_a': tf.Variable(
                initial_value=params_local['lambda_a'],
                trainable=True, constraint=NonNeg(),
                dtype=K.floatx(), name='lambda_a'
            ),
            'lambda_w': tf.Variable(
                initial_value=params_local['lambda_w'],
                trainable=True, constraint=NonNeg(),
                dtype=K.floatx(), name='lambda_w'
            )
        }
        return theta

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
            self._params['gamma']['bounds'],
            self._params['phi']['bounds'],
            self._params['lambda_w']['bounds'],
            self._params['lambda_a']['bounds'],
        ]
        return bnds


# @tf.function
def alcove_loss(desired_y, predicted_y):
    """ALCOVE loss."""
    min_val = math_ops.cast(-1.0, K.floatx())
    teacher_y_min = tf.minimum(min_val, predicted_y)

    # Zero out correct locations.
    teacher_y = teacher_y_min - tf.multiply(desired_y, teacher_y_min)

    # Add in correct locations.
    max_val = math_ops.cast(1.0, K.floatx())
    teacher_y_max = tf.maximum(max_val, predicted_y)
    teacher_y = teacher_y + tf.multiply(desired_y, teacher_y_max)

    # Sum over outputs.
    loss = tf.reduce_mean(tf.square(teacher_y - predicted_y), axis=1)

    return loss


NestedInput = collections.namedtuple('NestedInput', ['z_in', 'one_hot_label'])


NestedState = collections.namedtuple('NestedState', ['state1', 'state2'])


class ALCOVECell(Layer):
    """A RNN cell."""

    def __init__(self, theta, coordinates, n_output, batch_size, **kwargs):
        """Initialize."""
        self.theta = theta
        self.n_dim = coordinates.shape[1]
        self.n_hidden = coordinates.shape[0]
        self.n_output = n_output
        self.state_size = NestedState(
            state1=tf.TensorShape([self.n_dim]),
            state2=tf.TensorShape([self.n_hidden, self.n_output])
        )
        self.rbf = WeightedMinkowski(coordinates, theta['rho'])
        self.batch_size = batch_size
        super(ALCOVECell, self).__init__(**kwargs)
        print('  ACLOVECell init')

    # @tf.function
    def call(self, inputs, states):
        """Call.

        Arguments:
            inputs: Expect inputs to contain 2 items:
                z: shape=(batch, n_dim)
                one_hot_label: shape=(batch, n_output)]

        """
        # print('    ALCOVECell call') TODO remove

        z_in, one_hot_label = tf.nest.flatten(inputs)
        attention, association = states

        # Auto-diff version.
        state_variables = [attention, association]
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as state_tape:
            state_tape.watch(state_variables)
            # Compute RBF activations.
            d = self.rbf(z_in, attention)
            s = tf.exp(
                tf.negative(self.theta['beta']) * tf.pow(d, self.theta['tau'])
            ) + self.theta['gamma']
            # Compute output activations.
            # Convert to shape=(batch_size, n_hidden, n_output)
            s2 = tf.expand_dims(s, axis=2)
            x2 = tf.multiply(s2, association)
            x_out = tf.reduce_sum(x2, axis=1)
            # Compute ALCOVE loss.
            loss = alcove_loss(one_hot_label, x_out)
        dl_da = state_tape.gradient(loss, attention)
        dl_dw = state_tape.gradient(loss, association)
        del state_tape

        # dl_da = 0
        # dl_dw = 0

        # tf.debugging.check_numerics(
        #     dl_da,
        #     'Check numerics `dl_da` in ALCOVECell.call',
        #     name='check_alcovecell_call'
        # )
        # Handle nan's. TODO where are the nan's coming from?
        # locs = tf.math.is_nan(dl_da)
        # dl_da = tf.where(locs, tf.zeros(dl_da.shape), dl_da)

        # Response rule (keep as logits).
        x_out_scaled = tf.multiply(x_out, self.theta['phi'])

        # Update states.
        new_attention = attention - self.theta['lambda_a'] * dl_da
        new_attention = tf.math.maximum(new_attention, 0)
        new_association = association - self.theta['lambda_w'] * dl_dw

        new_states = NestedState(
            state1=new_attention, state2=new_association
        )
        return x_out_scaled, new_states


class WeightedMinkowski(Layer):
    """Compute the weighted Minkowski distance.

    Distance is computed between the input(s) and a pre-defined
    set of coordinates.
    """

    def __init__(self, coordinates, rho):
        """Initialize."""
        super(WeightedMinkowski, self).__init__()
        self.coordinates = coordinates
        self.n_hidden = coordinates.shape[0]
        self.n_dim = coordinates.shape[1]
        self.rho = rho
        # self.dist_qr_0 = tf.zeros((2, self.n_hidden, self.n_dim))  # TODO

    def call(self, z_in, attention):
        """Call.

        Arguments:
            z_in: (batch_size, n_dim)
            attention: shape=(batch_size, n_dim)

        """
        # Add dimensions to exploit broadcasting rules.
        # e.g., shape (batch size, n_hidden, n_dim)

        # Expand inputs to have singleton n_hidden dimension.
        z_q = tf.expand_dims(z_in, axis=1)

        # Expand inputs to have singleton n_hidden dimension.
        a = tf.expand_dims(attention, axis=1)

        # Expand coordinates to have singleton batch_size dimension.
        z_r = tf.expand_dims(self.coordinates, axis=0)

        return self._minkowski_distance(z_q, z_r, a)

    def _minkowski_distance(self, z_q, z_r, a):
        """Weighted Minkowski distance.

        Arguments:
            z_q: A set of embedding points.
                shape = (batch_size, 1, n_dim)
            z_r: A set of embedding points.
                shape = (1, n_hidden, n_dim)
            a: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (batch_size, 1, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (batch_size, n_hidden)

        """
        # Weighted Minkowski distance.
        dist_qr_0 = tf.pow(tf.abs(z_q - z_r), self.rho)
        dist_qr_1 = tf.multiply(dist_qr_0, a)
        dist_qr_2 = tf.pow(tf.reduce_sum(dist_qr_1, axis=-1), tf.math.divide(1., self.rho))

        tf.debugging.check_numerics(
            dist_qr_2,
            'Check numerics `dist_qr` in _minkowski distance',
            name='check_minkowski'
        )
        return dist_qr_2


class GreaterEqualThan(Constraint):
    """Constrains the weights to be greater than a specified value."""

    def __init__(self, min_value=0.):
        """Initialize."""
        self.min_value = min_value

    def __call__(self, w):
        """Call."""
        w_adj = w - self.min_value
        w2 = w_adj * math_ops.cast(math_ops.greater_equal(w_adj, 0.), K.floatx())
        w2 = w2 + self.min_value
        return w2


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
        attention_mode:

    Returns:
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

    # if np.sum(np.logical_not(np.isfinite(attention_out))) > 0:
    #     print('bad attention projection')
    # attention_sum = np.sum(attention_0, axis=1)
    # locs = np.logical_not(np.isfinite(attention_sum))
    # attention[locs, :] = 1

    return attention_out
