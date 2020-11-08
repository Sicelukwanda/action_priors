from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(0)
np.random.seed(0)

class AgentMLPTF(Model):
    """
    Uses subclassing and functional API in keras to define a Gaussian MLP policy.
    """

    def __init__(self):
        super(AgentMLPTF, self).__init__()

        # model inputs
        inputs = tf.keras.Input(shape=(1,))

        # model layers (output = activation(dot(input, kernel) + bias) )
        d1 = Dense(15, activation='relu', name="d1")(inputs)
        d2_mu = Dense(1, activation='tanh', name="d2_mu")(d1)
        d2_sigma = Dense(1, activation='exponential', name="d2_sigma")(d1)  # consider using sigmoid/exponential here

        # model outputs
        outputs = {"Loc": d2_mu, "Scale": d2_sigma}

        self.model = tf.keras.Model(inputs, outputs)  # model returns TFP Gaussian dist params as output

    def call(self, x):

        batch = True
        if np.ndim(x) == 1:
            batch = False
            x = np.expand_dims(x, axis=1)

        # 1. Define Policy
        GaussianParams = self.model(x)  # at this point, x is the output of d1

        # 2. Sample policy to get action
        min_stddev = 0 #1e-4
        #print("Scale is:",GaussianParams["Scale"].numpy())
        action_dist = tfd.Normal(GaussianParams["Loc"], GaussianParams["Scale"]+min_stddev, validate_args=True)


        action = action_dist.sample()
        # the sample actions a vector of multple actions,
        action_log_prob = action_dist.log_prob(action) # add mininmum log prob for actions

        # if not batch:
        #     action = action.item()
        return dict({"Action": action.numpy().flatten(), "LogProbability": action_log_prob}, **GaussianParams)

    def visualise_model(self):
        tf.keras.utils.plot_model(
            self.model, to_file='model.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=True, dpi=96
        )


