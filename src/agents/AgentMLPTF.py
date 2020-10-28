class AgentMLPTF(Model):
    def __init__(self):
        super(AgentMLPTF, self).__init__()
        self.d1 = Dense(15, activation='relu')
        self.d2 = Dense(2)

    def __call__(self, x):
        # 1. Define Policy
        batch = True
        if np.ndim(x) == 1:
            batch = False
            x = np.expand_dims(x, axis=1)
        x = self.d1(x)
        action_logits = self.d2(x)

        # 2. Sample policy to get action
        action_loc = tf.nn.tanh(tf.slice(action_logits, [0], -1))
        action_scale = tf.exp(tf.slice(action_logits, [1], -1))

        # Create Gaussian head for NN
        action_dist = tfd.Normal(action_loc, action_scale)

        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        action = action.numpy().flatten()

        #         if not batch:
        #             action = action.item()
        return {"Action": action, "LogProbability": action_log_prob, "Loc": action_loc, "Scale": action_scale}