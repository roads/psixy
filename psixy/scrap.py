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