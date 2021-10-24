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
    activations = [tfkl.LeakyReLU if act == 'leaky_relu' else
                   act for act in activations]

    layers = [tfkl.Flatten(input_shape=input_shape)]
    for act in activations:
        layers.extend([tfkl.Dense(hidden_size), tfkl.Activation(act)
                       if isinstance(act, str) else act()])
    layers.extend([tfkl.Dense(np.prod(output_shape))])
    # if final_tanh:
    #     layers.extend([TanhMultiplier()])
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
    layers = [tfkl.Flatten(input_shape=input_shape)]
    for act in activations:
        layers.extend([tfkl.Dense(hidden_size), tfkl.Activation(act)
                       if isinstance(act, str) else act()])
    layers.extend([tfkl.Dense(1)])
    if final_tanh:
        layers.extend([TanhMultiplier()])
    return tf.keras.Sequential(layers)



class PolicyContinuousForwardModel(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""


    def __init__(self, input_shape, task, latent_size=20, embedding_size=50, hidden_size=10,
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
        self.distribution = tfpd.MultivariateNormalDiag
        self.max_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_max_std).astype(np.float32)), trainable=True)
        self.min_logstd = tf.Variable(tf.fill([1, 1], np.log(
            initial_min_std).astype(np.float32)), trainable=True)
        layers = [tfkl.Flatten(input_shape=input_shape)]

        for i in range(num_layers):
            layers.extend([tfkl.Dense(hidden_size), tfkl.LeakyReLU()])

        if task.is_discrete:
            layers.append(tfkl.Dense(np.prod(task.input_shape)))
        else:
            layers.append(tfkl.Dense(np.prod(task.input_shape)*2))
        super(PolicyContinuousForwardModel, self).__init__(layers)

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

        prediction = super(PolicyContinuousForwardModel, self).__call__(inputs, **kwargs)
        mean, logstd = tf.split(prediction, 2, axis=-1)
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        return {"loc": mean, "scale_diag": tf.math.softplus(logstd)}

    def get_distribution(self, **kwargs):
        """Return a distribution over the outputs of this model, for example
        a Multivariate Gaussian Distribution

        Args:

        inputs: tf.Tensor
            a batch of training inputs shaped like [batch_size, channels]

        Returns:

        distribution: tfp.distribution.Distribution
            a tensorflow probability distribution over outputs of the model
        """
        noise_input_shape = 10
        noise_dist = tfpd.MultivariateNormalDiag(loc=[0] * noise_input_shape, scale_diag=[0.1] * noise_input_shape)
        inputs = noise_dist.sample(1) # (10,1)
        return self.distribution(**self.get_params(inputs, **kwargs))

    
    def get_density(self, inputs, **kwargs):
        self.dist = self.get_distribution(**kwargs)
        return self.dist.prob(inputs)


    def get_sample(self, size, **kwargs):
        self.dist = self.get_distribution(**kwargs)
        return self.dist.sample(size)



class PolicyDiscreteForwardModel(tf.keras.Sequential):
    """A Fully Connected Network with 2 trainable layers"""


    def __init__(self, task, input_shape, hidden_size=50,
                 num_layers=1, **kwargs):
        """Create a fully connected architecture using keras that can process
        several parallel streams of weights and biases
        Args:
        task: StaticGraphTask
            a model-based optimization task
        latent_size: int
            the cardinality of the latent variable
        hidden_size: int
            the global hidden size of the neural network
        """
        self.distribution = tfpd.Categorical

        layers = [tfkl.Flatten(input_shape=input_shape)]
        for i in range(num_layers):
            layers.extend([tfkl.Dense(hidden_size, **kwargs),
                           tfkl.LeakyReLU()])

        layers.append(tfkl.Dense(np.prod(task.input_shape) * task.num_classes))
        layers.append(tfkl.Reshape(list(task.input_shape) + [task.num_classes]))
        super(PolicyDiscreteForwardModel, self).__init__(layers)

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
        x = super(PolicyDiscreteForwardModel, self).__call__(inputs, **kwargs)
        logits = tf.math.log_softmax(x, axis=-1)
        return {"logits": logits}

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

        return self.distribution(
            **self.get_params(inputs, **kwargs), dtype=tf.int32)

    def get_density(self, inputs, **kwargs):
        self.dist = self.get_distribution(inputs, **kwargs)
        logits = self.get_params(inputs, **kwargs)["logits"]
        density = tf.reduce_sum(tf.math.multiply(logits, inputs), axis=2)
        density = tf.exp(tf.reduce_sum(density, axis=1))
        return density


    def get_sample(self, size, **kwargs):
        self.dist = self.get_distribution(**kwargs)
        return self.dist.sample(size)
