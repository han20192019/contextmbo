from design_baselines.utils import spearman
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class RepresentationLearningModel(tf.Module):
    def __init__(self, rep_model, forward_model, policy_model,
                 rep_model_opt=tf.keras.optimizers.Adam,
                 rep_model_lr=0.001, 
                 forward_model_opt=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001, 
                 policy_model_opt=tf.keras.optimizers.Adam,
                 policy_model_lr=0.001,
                 L2_alpha=1.0,
                 MMD_alpha=1.0, reward_alpha=1.0
                 ):
        
        super().__init__()
        self.rep_model = rep_model
        self.forward_model = forward_model
        self.policy_model = policy_model
        self.forward_model_opt = \
            forward_model_opt(learning_rate=forward_model_lr)
        self.policy_model_opt = \
            policy_model_opt(learning_rate=policy_model_lr)
        self.rep_model_opt = \
            rep_model_opt(learning_rate=rep_model_lr)


    def train_step_fphi(self, x, y):
        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:
            
            repx = self.rep_model(x, training=True)
            d_pos = self.forward_model(repx, training=True)

            # Policy Reward
            density = self.policy_model.get_density(x, training=False)
            rewards = tf.tensordot(d_pos, density)
            statistics[f'train/rewards'] = rewards

            # L2 loss
            mse = tf.keras.losses.mean_squared_error(y, d_pos)
            statistics[f'train/mse'] = mse

            # MMD loss
            learned_x = self.policy_model.get_sample(x, training=False)
            learned_rep = tf.reduce_mean(self.rep_model(learned_x, training=True))
            logged_rep = tf.reduce_mean(repx)
            mmd = (learned_rep - logged_rep) ** 2
            statistics[f'train/mmd'] = mmd

        # calculate gradients using the model
        phi_grads = tape.gradient(-rewards + mse + mmd, self.rep_model.train)
        f_grads = tape.gradient(rewards + mse + mmd, self.forward_model.train)

        self.forward_model_opt.apply_gradients(zip(
            f_grads, self.forward_model.trainable_variables))

        self.rep_model_opt.apply_gradients(zip(
            phi_grads, self.forward_model.trainable_variables))


    def train_step_pi(self, x):
        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:
            
            repx = self.rep_model(x, training=False)
            d_pos = self.forward_model(repx, training=False)

            # Policy Reward
            density = self.policy_model.get_density(x, training=False)
            rewards = tf.tensordot(d_pos, density)
            statistics[f'train/rewards'] = rewards

            # MMD loss
            learned_x = self.policy_model.get_sample(x, training=False)
            learned_rep = tf.reduce_mean(self.rep_model(learned_x, training=True))
            logged_rep = tf.reduce_mean(repx)
            mmd = (learned_rep - logged_rep) ** 2
            statistics[f'train/mmd'] = mmd

        # calculate gradients using the model
        pi_grads = tape.gradient(-rewards + mmd, self.policy_model.train)
        self.policy_model_opt.apply_gradients(zip(
            pi_grads, self.policy_model.trainable_variables))


    def train_fphi(self, dataset):
        """Perform training using gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        dataset: tf.data.Dataset
            the training dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.train_step_fphi(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    
    def train_pi(self, dataset):
        """Perform training using gradient descent on an ensemble
        using bootstrap weights for each model in the ensemble

        Args:

        dataset: tf.data.Dataset
            the training dataset already batched and prefetched

        Returns:

        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.train_step_pi(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics


    def launch(self, train_data, validate_data, logger, epochs):
        """Launch training and validation for the model for the specified
        number of epochs, and log statistics

        Args:

        train_data: tf.data.Dataset
            the training dataset already batched and prefetched
        validate_data: tf.data.Dataset
            the validation dataset already batched and prefetched
        logger: Logger
            an instance of the logger used for writing to tensor board
        epochs: int
            the number of epochs through the data sets to take
        """

        for e in range(epochs):
            for name, loss in self.train_fphi(train_data).items():
                logger.record(name, loss, e)
            for name, loss in self.train_pi(train_data).items():
                logger.record(name, loss, e)
