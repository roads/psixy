class ALCOVEGroupBlock(Layer):
    """ALCOVE block for creating composite models."""

    def __init__(self, n_sequence, coordinates, n_output, theta):
        """Initialize."""
        super(ALCOVEGroupBlock, self).__init__()
        n_hidden = coordinates.shape[0]
        self.rbf = ExponentialFamily(coordinates, theta)
        self.association = ParallelDense(
            n_sequence, n_hidden, n_output, activation=None, use_bias=False  # name='association'
        )
        self.response = ParallelDense(
            n_sequence, n_output, n_output, activation='softmax',
            use_bias=False, kernel_initializer='identity', trainable=False  # name='response'
        )
        self.theta = theta

    def call(self, inputs):
        """Call."""
        x = self.rbf(inputs)
        x = self.association(x)
        # TODO add phi
        x = tf.multiply(x, self.theta['phi'])
        return self.response(x)


class ALCOVEBlock(Layer):
    """ALCOVE block for creating composite models."""

    def __init__(self, coordinates, n_output, theta):
        """Initialize."""
        super(ALCOVEBlock, self).__init__()
        n_hidden = 10
        self.rbf = ExponentialFamily(coordinates, theta)
        self.association = Dense(
            n_output, activation=None, use_bias=False, name='association'
        )
        self.response = Dense(
            n_output, activation='softmax', use_bias=False,
            kernel_initializer='identity', trainable=False, name='response'
        )
        self.theta = theta

    def call(self, inputs):
        """Call."""
        x = self.rbf(inputs)
        x = self.association(x)
        # TODO add phi
        x = tf.multiply(x, self.theta['phi'])
        return self.response(x)


class GroupALCOVE(Model):
    """A TensorFlow model for a single agent."""

    def __init__(self, coordinates, n_sequence, n_output, rho, tau, beta, gamma, phi):
        """Initialize."""
        super(GroupALCOVE, self).__init__()
        self.theta = {}
        self.theta['rho'] = tf.Variable(
            initial_value=rho,
            trainable=False
        )
        self.theta['tau'] = tf.Variable(
            initial_value=tau,
            trainable=False
        )
        self.theta['beta'] = tf.Variable(
            initial_value=beta,
            trainable=False
        )
        self.theta['gamma'] = tf.Variable(
            initial_value=gamma,
            trainable=False
        )
        self.theta['phi'] = tf.Variable(
            initial_value=phi,
            trainable=False
        )

        self.n_sequence = n_sequence
        agent_list = []
        for i_agent in range(n_sequence):
            agent_list.append(ALCOVEBlock(coordinates, n_output, self.theta))
        self.agent_list = agent_list

    def call(self, inputs):
        """Call."""
        # (batch_size, n_seq, n_dim)
        # self.block = tf.keras.layers.concatenate(
        #     agent_list,
        #     axis=1
        # )
        x_list = []
        for i_agent in range(self.n_sequence):
            x0 = inputs[:, i_agent, :]
            x1 = self.agent_list[i_agent](x0)
            x1 = tf.expand_dims(x1, axis=1)
            x_list.append(x1)
        x = tf.keras.layers.concatenate(
            x_list,
            axis=1
        )
        return x


# ===========================================================================
class ALCOVE2(CategoryLearningModel):
    """A tensorflow implimentation of ALCOVE (Kruschke, 1992).

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
        self.z = z.astype(dtype=float)
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
        def obj_fun(params):
            return self._loss_opt(
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
        loss = self._loss(
            self.params, stimulus_sequence, behavior_sequence
        )
        if verbose > 0:
            print("loss: {0:.2f}".format(loss))
        return loss

    def predict(
            self, stimulus_sequence, group_id=None, mode='all',
            stateful=False, verbose=0):
        """Predict behavior."""
        prob_response = self._run(self.params, stimulus_sequence)

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

    def _loss_opt(
            self, params_opt, stimulus_sequence, behavior_sequence):
        """Compute the log-likelihood of the data given model.

        Parameters are structured for using scipy.optimize.minimize.
        """
        params_local = {
            'rho': params_opt[0],
            'tau': params_opt[1],
            'beta': params_opt[2],
            'gamma': params_opt[3],
            'phi': params_opt[4],
            'lambda_w': params_opt[5],
            'lambda_a': params_opt[6]
        }
        loss = self._loss(
            params_local, stimulus_sequence, behavior_sequence
        )
        return loss

    def _run(
            self, params_local, stimulus_sequence):
        """Run model.

        Arguments:
            stimulus_sequence: A psixy.sequence.StimulusSequence object.

        Returns:
            loss_train: The negative log-likelihood of the data given
                the fitted model.

        """
        # Float formatting for tensorflow. TODO
        stimulus_sequence.z = stimulus_sequence.z.astype(dtype=np.float32)
        z_hidden = self.z.astype(dtype=np.float32)

        n_trial = stimulus_sequence.n_trial
        n_sequence = stimulus_sequence.n_sequence

        # TODO how should we handle optimization parameters?
        lambda_a = params_local['lambda_a']
        lambda_w = params_local['lambda_w']

        theta = {
            'rho': tf.Variable(
                initial_value=params_local['rho'], trainable=True,
                dtype='float32', name='rho'
            ),
            'tau': tf.Variable(
                initial_value=params_local['tau'], trainable=True,
                dtype='float32', name='tau'
            ),
            'beta': tf.Variable(
                initial_value=params_local['beta'], trainable=True,
                dtype='float32', name='beta'
            ),
            'gamma': tf.Variable(
                initial_value=params_local['gamma'], trainable=True,
                dtype='float32', name='gamma'
            ),
            'phi': tf.Variable(
                initial_value=params_local['phi'], trainable=True,
                dtype='float32', name='phi'
            )
        }
        model_parameters = list(theta.values())

        model = InnerAlcove(
            n_sequence, z_hidden, self.n_class, theta
        )
        state_variables = [model.rbf.minkowski.a, model.association.w]

        state_optimizers = {
            'a': tf.keras.optimizers.SGD(
                learning_rate=lambda_a, momentum=0.0, nesterov=False
            ),
            'w': tf.keras.optimizers.SGD(
                learning_rate=lambda_w, momentum=0.0, nesterov=False
            )
        }

        # @tf.function
        # def parameter_update():
        #     with tf.GradientTape(watch_accessed_variables=False) as model_tape:
        #         model_tape.watch(model_parameters)
        #         prob_response = tf_predict(stimulus_sequence)
        #         loss = None  # TODO
        #     parameter_gradients = model_tape.gradient(loss, model_parameters)
        #     optimizer_model.apply_gradients(zip(gradients, model_parameters))
        #     return loss

        @tf.function
        def alcove_loss(desired_y, predicted_y):
            """ALCOVE loss."""
            desired_y = tf.one_hot(desired_y, self.n_class, axis=2)
            teacher_y_min = tf.minimum(-1.0, predicted_y)

            # Zero out correct locations.
            teacher_y = teacher_y_min - tf.multiply(desired_y, teacher_y_min)

            # Add in correct locations.
            teacher_y_max = tf.maximum(1.0, predicted_y)
            teacher_y = teacher_y + tf.multiply(desired_y, teacher_y_max)

            # Sum over outputs.
            loss = tf.reduce_mean(tf.square(teacher_y - predicted_y), axis=2)

            # Sum over batches (if any).
            loss = tf.reduce_mean(loss, axis=0)
            return loss

        @tf.function
        def state_update(model, state_variables, state_optimizers, inputs, labels):
            """Update model state."""
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as state_tape:
                state_tape.watch(state_variables)
                x_out = model(inputs)
                loss = alcove_loss(labels, x_out)
            dl_da = state_tape.gradient(loss, model.rbf.minkowski.a)
            dl_dw = state_tape.gradient(loss, model.association.w)
            state_optimizers['a'].apply_gradients([(dl_da, model.rbf.minkowski.a)])
            state_optimizers['w'].apply_gradients([(dl_dw, model.association.w)])
            del state_tape
            return x_out

        # @tf.function
        def tf_predict(stimulus_sequence):
            """One iteration."""
            prob_response_list = []
            for i_trial in range(n_trial):
                inputs = stimulus_sequence.z[:, i_trial, :]
                inputs = inputs[tf.newaxis, ...]
                stimulus_labels = stimulus_sequence.class_id[:, i_trial]
                stimulus_labels = self._convert_labels(stimulus_labels)
                stimulus_labels = stimulus_labels[tf.newaxis, ...]

                x_out = state_update(
                    model, state_variables, state_optimizers, inputs, stimulus_labels
                )
                p = general_softmax(x_out, theta['phi'])
                prob_response_list.append(p[0])  # TODO handle batch_size
            prob_response = tf.stack(prob_response_list, axis=1)
            return prob_response

        prob_response = tf_predict(stimulus_sequence)

        return prob_response

    def _convert_labels(self, labels):
        """Convert labels."""
        labels_conv = np.zeros(labels.shape, dtype=int)
        for key, value in self.class_map.items():
            locs = np.equal(labels, key)
            labels_conv[locs] = value
        return labels_conv

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


class MinimalRNNCell(Layer):
    """A RNN cell."""

    def __init__(self, units, **kwargs):
        """Initialize."""
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build."""
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        """Call."""
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]


class WeightedMinkowski(Layer):
    """Compute the weighted Minkowski distance.

    Distance is computed between the input(s) and a pre-defined
    set of coordinates.
    """

    def __init__(self, n_sequence, coordinates, rho, weight=None):
        """Initialize."""
        super(WeightedMinkowski, self).__init__()
        self.coordinates = coordinates
        self.n_hidden = coordinates.shape[0]
        self.n_dim = coordinates.shape[1]

        if weight is None:
            weight = np.ones([n_sequence, self.n_dim]) / self.n_dim
            # weight = np.ones([self.n_dim]) / self.n_dim

        self.a = self.add_weight(
            shape=(n_sequence, self.n_dim),
            initializer=constant_initializer(weight),
            constraint=NonNeg(),
            trainable=True,
            name='attention'
        )
        self.rho = rho

    def call(self, inputs):
        """Call."""
        # Add dimensions to exploit broadcasting rules.
        # e.g., shape (batch size, n_seq, n_hidden, n_dim)
        a_expand = tf.expand_dims(self.a, axis=0)
        a_expand = tf.expand_dims(a_expand, axis=2)

        coordinates_expand = tf.expand_dims(self.coordinates, axis=0)
        coordinates_expand = tf.expand_dims(self.coordinates, axis=0)

        inputs_expand = tf.expand_dims(inputs, axis=2)

        return self._minkowski_distance(
            inputs_expand, coordinates_expand, a_expand
        )

    def _minkowski_distance(self, z_q, z_r, tf_attention):
        """Weighted Minkowski distance.

        Arguments:
            z_q: A set of embedding points.
                shape = (n_trial, n_dim)
            z_r: A set of embedding points.
                shape = (n_trial, n_dim)
            tf_attention: The weights allocated to each dimension
                in a weighted minkowski metric.
                shape = (n_trial, n_dim)

        Returns:
            The corresponding similarity between rows of embedding
                points.
                shape = (n_trial,)

        """
        # Weighted Minkowski distance.
        d_qref = tf.pow(tf.abs(z_q - z_r), self.rho)
        d_qref = tf.multiply(d_qref, tf_attention)
        d_qref = tf.pow(tf.reduce_sum(d_qref, axis=-1), 1. / self.rho)

        return d_qref


class ExponentialFamily(Layer):
    """Exponential family RBF with attention.

    Returns:
        (batch_size, n_seq, n_hidden)

    """

    def __init__(self, n_sequence, coordinates, theta):
        """Initialize."""
        super(ExponentialFamily, self).__init__()
        self.minkowski = WeightedMinkowski(n_sequence, coordinates, rho=theta['rho'])
        self.theta = theta

    def call(self, inputs):
        """Call."""
        d = self.minkowski(inputs)
        s = self._similarity(d)
        return s

    def _similarity(self, d):
        """Exponential family similarity function.

        Arguments:
            d: An array of distances.

        Returns:
            sim: The transformed distances.

        """
        sim = tf.exp(
            tf.negative(self.theta['beta']) * tf.pow(d, self.theta['tau'])
        ) + self.theta['gamma']
        return sim


class LinearAssociation(Layer):
    """A linear association layer.

    There is a separate association layer for each sequence.

    Returns:
        (batch_size, n_sequence, n_output)

    """

    def __init__(
            self, n_sequence, n_hidden, n_output, trainable=True):
        """Initialize."""
        super(LinearAssociation, self).__init__()
        self.n_sequence = n_sequence
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Note that we intentionally do not use the build method so
        # that the variables are available to set GradientTape watching
        # rules.
        w_init = tf.zeros_initializer()
        self.w = tf.Variable(
            initial_value=w_init(
                shape=(self.n_sequence, self.n_hidden, self.n_output),
                dtype='float32'
            ),
            trainable=True,
            name='association'
        )

    def call(self, act_hidden):
        """Call."""
        # batch_size, n_sequence, n_hidden, n_output
        x1 = tf.expand_dims(act_hidden, axis=3)  # TODO
        w1 = tf.expand_dims(self.w, axis=0)
        x2 = tf.multiply(x1, self.w)  # + self.b
        x3 = tf.reduce_sum(x2, axis=2)
        return x3


class InnerAlcove(Model):
    """A TensorFlow model.

    Assumes:
        input is (batch_size, n_sequence, n_dim)
        output is (batch_size, n_sequence, n_output)
    """

    def __init__(self, n_sequence, coordinates, n_output, theta):
        """Initialize."""
        super(InnerAlcove, self).__init__()
        self.theta = theta
        n_hidden = coordinates.shape[0]
        self.rbf = ExponentialFamily(n_sequence, coordinates, self.theta)
        self.association = LinearAssociation(
            n_sequence, n_hidden, n_output
        )

    def call(self, inputs):
        """Call."""
        # Pass through RBF layer (with attention).
        x_hidden = self.rbf(inputs)

        # Apply association weight matrix.
        x_out = self.association(x_hidden)

        # # Apply response rule.
        # p = tf.multiply(x_out, self.theta['phi'])
        # return tf.math.softmax(p, axis=2)
        return x_out


def general_softmax(x_out, phi):
    """Apply generalized soft max."""
    p = tf.multiply(x_out, phi)
    p = tf.math.softmax(p, axis=2)
    return p


