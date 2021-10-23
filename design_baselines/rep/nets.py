from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


class TanhMultiplier(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(TanhMultiplier, self).__init__(**kwargs)
        w_init = tf.constant_initializer(1.0)
        self.multiplier = tf.Variable(initial_value=w_init(
            shape=(1,), dtype="float32"), trainable=True)

    def call(self, inputs, **kwargs):
        exp_multiplier = tf.math.exp(self.multiplier)
        return tf.math.tanh(inputs / exp_multiplier) * exp_multiplier


def RepModel(input_shape,
                output_shape,
                activations=('relu', 'relu'),
                hidden_size=2048,
                final_tanh=False):
    final_out = output_shape[0] * output_shape[1]
    activations = [tfkl.LeakyReLU if act == 'leaky_relu' else
                   act for act in activations]

    layers = [tfkl.Flatten(input_shape=input_shape)]
    for act in activations:
        layers.extend([tfkl.Dense(hidden_size), tfkl.Activation(act)
                       if isinstance(act, str) else act()])
    layers.extend([tfkl.Dense(final_out)])
    if final_tanh:
        layers.extend([TanhMultiplier()])
    return tf.keras.Sequential(layers)

    
def ForwardModel(input_shape,
                 activations=('relu', 'relu'),
                 hidden_size=2048,
                 final_tanh=False):
    """Creates a tensorflow model that outputs a probability distribution
    specifying the score corresponding to an input x.

    Args:

    input_shape: tuple[int]
        the shape of input tensors to the model
    activations: tuple[str]
        the name of activation functions for every hidden layer
    hidden: int
        the global hidden size of the network
    max_std: float
        the upper bound of the learned standard deviation
    min_std: float
        the lower bound of the learned standard deviation
    """

    activations = [tfkl.LeakyReLU if act == 'leaky_relu' else
                   act for act in activations]

    layers = [] 
    # layers = [tfkl.Flatten(input_shape=input_shape)]
    for act in activations:
        layers.extend([tfkl.Dense(hidden_size), tfkl.Activation(act)
                       if isinstance(act, str) else act()])
    layers.extend([tfkl.Dense(1)])
    if final_tanh:
        layers.extend([TanhMultiplier()])
    return tf.keras.Sequential(layers)



class PolicyForwardModel(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""

    distribution = tfpd.MultivariateNormalDiag

    def __init__(self, task, latent_size=20, embedding_size=50, hidden_size=50,
                 num_layers=1, initial_max_std=1.5, initial_min_std=0.5):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases

        Args:

        task: StaticGraphTask
            a model-based optimization task
        latent_size: int
            the cardinality of the latent variable
        embedding_size: int
            the size of the embedding matrix for discrete tasks
        hidden_size: int
            the global hidden size of the neural network
        num_layers: int
            the number of hidden layers
        initial_max_std: float
            the starting upper bound of the standard deviation
        initial_min_std: float
            the starting lower bound of the standard deviation
        """

        self.max_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_max_std).astype(np.float32)), trainable=True)
        self.min_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_min_std).astype(np.float32)), trainable=True)

        layers = []
        if task.is_discrete:
            layers.append(tfkl.Embedding(task.num_classes, embedding_size,
                                         input_shape=task.input_shape))
        layers.append(tfkl.Flatten(input_shape=task.input_shape)
                      if len(layers) == 0 else tfkl.Flatten())
        for i in range(num_layers):
            layers.extend([tfkl.Dense(hidden_size), tfkl.LeakyReLU()])

        layers.append(tfkl.Dense(latent_size * 2))
        super(PolicyForwardModel, self).__init__(layers)

    def get_params(self, inputs, **kwargs):
        """Return a dictionary of parameters for a particular distribution
        family such as the mean and variance of a gaussian

        Args:

        inputs: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        parameters: dict
            a dictionary that contains 'loc' and 'scale_diag' keys
        """

        prediction = super(PolicyForwardModel, self).__call__(inputs, **kwargs)
        mean, logstd = tf.split(prediction, 2, axis=-1)
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        return {"loc": mean, "scale_diag": tf.math.softplus(logstd)}

    def get_distribution(self, inputs, **kwargs):
        """Return a distribution over the outputs of this model, for example
        a Multivariate Gaussian Distribution

        Args:

        inputs: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        distribution: tfp.distribution.Distribution
            a tensorflow probability distribution over outputs of the model
        """

        return self.distribution(**self.get_params(inputs, **kwargs))

    
    def get_density(self, inputs, **kwargs):
        self.dist = self.get_distribution(inputs, **kwargs)
        return self.dist.prob(inputs)

    def get_sample(self, size):
        return self.dist.sample(size)
