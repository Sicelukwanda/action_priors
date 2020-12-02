from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

# make all layer data types consistent
tf.keras.backend.set_floatx('float32')

from IPython.core.debugger import set_trace

# Set random seeds
#tf.random.set_seed(0)
#np.random.seed(0)


class GaussianAgent(Model):
    """
    Uses subclassing and functional API in keras to define a Gaussian MLP policy.
    """

    def __init__(self,action_dim=1,state_dim=1):
        super(GaussianAgent, self).__init__()

        # declare action distribution
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_dist = None

        normalizer = tf.keras.layers.LayerNormalization(axis=1)

        # model inputs
        inputs = tf.keras.Input(shape=(state_dim,), dtype='float64')

        # normalize inputs
        # norm_inputs = normalizer(inputs)

        # model layers (output = activation(dot(input, kernel) + bias) )
        d1 = Dense(15, activation='relu', name="d1")(inputs)
        d1 = Dense(15, activation='relu', name="d12")(d1)
        d2_mu = Dense(action_dim, activation='tanh', name="d2_mu")(d1)
        # d2_sigma = Dense(action_dim, activation='exponential', name="d2_sigma")(d1)  # consider using sigmoid/exponential here

        # model outputs
        outputs = {"Loc": d2_mu} #, "Scale": d2_sigma}

        self.model = tf.keras.Model(inputs, outputs)  # model returns TFP Gaussian dist params as output

    def call(self, x):

        batch = True
        if np.ndim(x) == 1:
            batch = False
            x = np.expand_dims(x, axis=0)

        # 1. Define Policy
        GaussianParams = self.model(x)  # at this point, x is the output of d1

        min_stddev = 1e-4  # define minimum for sigma value, e.g. 1e-4

        # 2. Define Gaussian tf probability distribution layer
        self.action_dist = tfd.Normal(GaussianParams["Loc"], min_stddev, validate_args=True)

        # 3. Sample policy to get action
        action = self.action_dist.sample()

        # 4. Get log probability from tfp layer
        action_log_prob = self.action_dist.log_prob(action)  # add minimum log prob for actions

        if not batch:
             action = action.numpy().flatten()
             action_log_prob = action_log_prob.numpy().flatten()
        return dict({"Action": action, "LogProbability": action_log_prob}, **GaussianParams)

    def visualise_model(self):
        tf.keras.utils.plot_model(
            self.model, to_file='model.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=True, dpi=96
        )



class BetaAgent(Model):
    """
    Uses subclassing and functional API in keras to define a Gaussian MLP policy.
    """

    def __init__(self):
        super(BetaAgent, self).__init__()

        # declare action distribution
        self.action_dist = None

        normalizer = tf.keras.layers.LayerNormalization(axis=1)

        # model inputs
        inputs = tf.keras.Input(shape=(1,))

        # normalize inputs
        # norm_inputs = normalizer(inputs)
        # norm_inputs = inputs/10.0

        # model layers (output = activation(dot(input, kernel) + bias) )
        d1 = Dense(15, activation='relu', name="d1")(inputs)
        d1 = Dense(15, activation='relu', name="d12")(d1)
        d2_alpha = Dense(1, activation='exponential', name="d2_mu")(d1)
        d2_beta = Dense(1, activation='exponential', name="d2_sigma")(d1)  # consider using sigmoid/exponential here

        # model outputs
        outputs = {"Alpha": d2_alpha, "Beta": d2_beta}

        self.model = tf.keras.Model(inputs, outputs)  # model returns TFP Gaussian dist params as output

    def call(self, x):

        batch = True
        if np.ndim(x) == 1:
            batch = False
            x = np.expand_dims(x, axis=1)

        # 1. Define Policy
        BetaParams = self.model(x)  # at this point, x is the output of d1

        # 2. Define Gaussian tf probability distribution layer
        self.action_dist = tfd.Beta(BetaParams["Alpha"], BetaParams["Beta"], validate_args=True)
        # set_trace()

        # 3. Sample policy to get action
        action = self.action_dist.sample()

        # 4. Get log probability from tfp layer
        action_log_prob = self.action_dist.log_prob(tf.convert_to_tensor(action))  # add minimum log prob for actions


        if not batch:
             action = action.numpy().flatten()
             action_log_prob = action_log_prob.numpy().flatten()
        return dict({"Action": action, "LogProbability": action_log_prob}, **BetaParams)

    def visualise_model(self):
        tf.keras.utils.plot_model(
            self.model, to_file='model.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=True, dpi=96
        )
