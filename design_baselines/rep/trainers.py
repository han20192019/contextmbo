from design_baselines.utils import spearman
from collections import defaultdict
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class RepresentationLearningModel(tf.Module):
    def __init__(self, task, rep_model, forward_model, policy_model,
                 rep_model_opt=tf.keras.optimizers.Adam,
                 rep_model_lr=0.001, 
                 forward_model_opt=tf.keras.optimizers.Adam,
                 forward_model_lr=0.001, 
                 policy_model_opt=tf.keras.optimizers.Adam,
                 policy_model_lr=0.001,
                 L2_alpha=1.0,
                 mmd_alpha=1.0, reward_alpha=1.0
                 ):
        
        super().__init__()
        self.task = task
        self.rep_model = rep_model
        self.forward_model = forward_model
        self.policy_model = policy_model
        self.forward_model_opt = \
            forward_model_opt(learning_rate=forward_model_lr)
        self.policy_model_opt = \
            policy_model_opt(learning_rate=policy_model_lr)
        self.rep_model_opt = \
            rep_model_opt(learning_rate=rep_model_lr)
        self.L2_alpha = L2_alpha
        self.mmd_alpha = mmd_alpha
        self.reward_alpha = reward_alpha
        self.new_sample_size = 32 
        if self.task.is_discrete:
            self.num_classes = self.task.num_classes

    # @tf.function(experimental_relax_shapes=True)
    def train_step_fphi(self, x, y):
        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:
            if self.task.is_discrete:
                x = tf.one_hot(x, depth=self.num_classes) 
            repx = self.rep_model(x, training=True)
            d_pos = self.forward_model(repx, training=True)

            # Policy Reward

            if self.task.is_discrete:
                input_shape = x.shape[1:][0]
                uniform_vec = [1./self.num_classes] * self.num_classes
                draw_samples = tf.reshape(tf.random.categorical(tf.math.log([uniform_vec]), self.new_sample_size * input_shape), (self.new_sample_size, input_shape))
                learned_x = tf.one_hot(draw_samples, depth=self.num_classes) # (32,8,4)
                densities = self.policy_model.get_density(learned_x, training=False) #(32,1)
                rep_learnedx = self.rep_model(learned_x, training=True)
                d_pos_learned = tf.transpose(self.forward_model(rep_learnedx, training=True)) 
                rewards = tf.tensordot(d_pos_learned, densities, axes=1)
            else:
                input_shape = x.shape[1:][0]
                learned_x = self.policy_model.get_sample(size=self.new_sample_size, training=False)
                learned_x = tf.reshape(learned_x, (self.new_sample_size, input_shape))
                rep_learnedx = self.rep_model(learned_x, training=True)
                d_pos_learned = self.forward_model(rep_learnedx, training=True)
                rewards = tf.reduce_mean(d_pos_learned)
            statistics[f'train/rewards'] = rewards

            # L2 loss
            mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, d_pos))
            statistics[f'train/mse'] = mse

            # MMD loss
            learned_rep = tf.reduce_mean(rep_learnedx, axis=0)
            logged_rep = tf.reduce_mean(repx, axis=0)

            mmd = tf.reduce_mean(tf.keras.losses.mean_squared_error(learned_rep, logged_rep))
            statistics[f'train/mmd'] = mmd
 
            #loss1 = -self.reward_alpha * rewards + self.L2_alpha * mse + self.mmd_alpha * mmd 
            #loss2 = self.reward_alpha * rewards + self.L2_alpha * mse + self.mmd_alpha * mmd
            loss1 = -self.reward_alpha * rewards + self.L2_alpha * mse
            loss2 = self.reward_alpha * rewards + self.L2_alpha * mse

        # calculate gradients using the model
        phi_grads = tape.gradient(loss1, self.rep_model.trainable_variables)
        f_grads = tape.gradient(loss2, self.forward_model.trainable_variables)

        self.forward_model_opt.apply_gradients(zip(
            f_grads, self.forward_model.trainable_variables))

        self.rep_model_opt.apply_gradients(zip(
            phi_grads, self.rep_model.trainable_variables))
        
        return statistics

    # @tf.function(experimental_relax_shapes=True)
    def train_step_pi(self, x):
        statistics = dict()
        with tf.GradientTape(persistent=True) as tape:

            if self.task.is_discrete:
                x = tf.one_hot(x, depth=self.num_classes) 

            repx = self.rep_model(x, training=False)
            d_pos = self.forward_model(repx, training=False)
            if self.task.is_discrete:
                input_shape = x.shape[1:][0]
                uniform_vec = [1./self.num_classes] * self.num_classes
                draw_samples = tf.reshape(tf.random.categorical(tf.math.log([uniform_vec]), self.new_sample_size * input_shape), (self.new_sample_size, input_shape))
                learned_x = tf.one_hot(draw_samples, depth=self.num_classes) # (32,8,4)
                densities = self.policy_model.get_density(learned_x, training=True) #(32,1)
                rep_learnedx = self.rep_model(learned_x, training=False)
                d_pos_learned = tf.transpose(self.forward_model(rep_learnedx, training=False)) 
                rewards = tf.tensordot(d_pos_learned, densities, axes=1)
            else:
                # Policy Reward
                input_shape = x.shape[1:][0]
                learned_x = self.policy_model.get_sample(size=self.new_sample_size, training=True)
                learned_x = tf.reshape(learned_x, (self.new_sample_size, input_shape))
                rep_learnedx = self.rep_model(learned_x, training=False)
                d_pos_learned = self.forward_model(rep_learnedx, training=False)
                rewards = tf.reduce_mean(d_pos_learned)

            statistics[f'train/rewards'] = rewards
            loss = -rewards


        # calculate gradients using the model
        pi_grads = tape.gradient(loss, self.policy_model.trainable_variables)
        self.policy_model_opt.apply_gradients(zip(
            pi_grads, self.policy_model.trainable_variables))
        
        return statistics


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
            for name, tensor in self.train_step_pi(x).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics
    
    def validate_fphi(self, dataset):
        """Perform validation on an ensemble of models without
        using bootstrapping weights
        Args:
        dataset: tf.data.Dataset
            the validation dataset already batched and prefetched
        Returns:
        loss_dict: dict
            a dictionary mapping names to loss values for logging
        """

        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.validate_step_fphi(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def validate_step_fphi(self, x, y):
        """Perform a validation step on an ensemble of models
        without using bootstrapping weights
        Args:
        x: tf.Tensor
            a batch of validation inputs shaped like [batch_size, channels]
        y: tf.Tensor
            a batch of validation labels shaped like [batch_size, 1]
        Returns:
        statistics: dict
            a dictionary that contains logging information
        """

        statistics = dict()

        # calculate the prediction error and accuracy of the model
        x_rep = self.rep_model(x, training = False)
        d_pos = self.forward_model(x_rep, training=False)
        mse = tf.keras.losses.mean_squared_error(y, d_pos)
        statistics[f'validate/mse'] = mse
        #print(mse)

        # evaluate how correct the rank fo the model predictions are
        rank_corr = spearman(y[:, 0], d_pos[:, 0])
        statistics[f'validate/rank_corr'] = rank_corr
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
            print(e)
            for name, loss in self.train_fphi(train_data).items():
                logger.record(name, loss, e)
            for name, loss in self.train_pi(train_data).items():
                logger.record(name, loss, e)
            
            for name, loss in self.validate_fphi(validate_data).items():
                logger.record(name, loss, e)

        
