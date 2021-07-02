from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.coms_cleaned.trainers import ConservativeObjectiveModel
from design_baselines.coms_cleaned.trainers import VAETrainer
from design_baselines.coms_cleaned.nets import ForwardModel
from design_baselines.coms_cleaned.nets import SequentialVAE
import tensorflow as tf
import numpy as np
import os
import click
import json


@click.command()
@click.option('--logging-dir',
              default='coms-cleaned',
              help='The directory in which tensorboard data is logged '
                   'during the experiment.')
@click.option('--task',
              default='HopperController-Exact-v0',
              help='The name of the design-bench task to use during '
                   'the experiment.')
@click.option('--task-relabel/--no-task-relabel',
              default=False,
              help='Whether to relabel the real Offline MBO data with '
                   'predictions made by the oracle (this eliminates a '
                   'train-test discrepency if the oracle is not an '
                   'adequate model of the data).')
@click.option('--normalize-ys/--no-normalize-ys',
              default=True,
              help='Whether to normalize the y values in the Offline MBO '
                   'dataset before performing model-based optimization.')
@click.option('--normalize-xs/--no-normalize-xs',
              default=True,
              help='Whether to normalize the x values in the Offline MBO '
                   'dataset before performing model-based optimization. '
                   '(note that x must not be discrete)')
@click.option('--in-latent-space/--not-in-latent-space',
              default=False,
              help='Whether to embed the designs into the latent space of '
                   'a VAE before performing model-based optimization '
                   '(based on Gomez-Bombarelli et al. 2018).')
@click.option('--vae-hidden-size',
              default=64,
              help='The hidden size of the neural network encoder '
                   'and decoder models used in the VAE.')
@click.option('--vae-latent-size',
              default=256,
              help='The size of the VAE latent vector space.')
@click.option('--vae-activation',
              default='relu',
              help='The activation function used in the VAE.')
@click.option('--vae-kernel-size',
              default=3,
              help='When the VAE is a CNN the kernel size of kernel '
                   'tensor in convolution layers.')
@click.option('--vae-num-blocks',
              default=4,
              help='The number of convolution blocks operating at '
                   'different spatial resolutions.')
@click.option('--vae-lr',
              default=0.0003,
              help='The learning rate of the VAE.')
@click.option('--vae-beta',
              default=1.0,
              help='The weight of the KL loss when training the VAE.')
@click.option('--vae-batch-size',
              default=32,
              help='The batch size used to train the VAE.')
@click.option('--vae-val-size',
              default=200,
              help='The number of samples in the VAE validation set.')
@click.option('--vae-epochs',
              default=10,
              help='The number of epochs to train the VAE.')
@click.option('--particle-lr',
              default=0.05,
              help='The learning rate used in the COMs inner loop.')
@click.option('--particle-train-gradient-steps',
              default=50,
              help='The number of gradient ascent steps used in the '
                   'COMs inner loop.')
@click.option('--particle-evaluate-gradient-steps',
              default=50,
              help='The number of gradient ascent steps used in the '
                   'COMs inner loop.')
@click.option('--particle-entropy-coefficient',
              default=0.0,
              help='The entropy bonus when solving the optimization problem.')
@click.option('--forward-model-activations',
              default=['relu', 'relu'],
              multiple=True,
              help='The series of activation functions for every layer '
                   'in the forward model.')
@click.option('--forward-model-hidden-size',
              default=2048,
              help='The hidden size of the forward model.')
@click.option('--forward-model-final-tanh/--no-forward-model-final-tanh',
              default=False,
              help='Whether to use a final tanh activation as the final '
                   'layer of the forward model.')
@click.option('--forward-model-lr',
              default=0.0003,
              help='The learning rate of the forward model.')
@click.option('--forward-model-alpha',
              default=1.0,
              help='The initial lagrange multiplier of the forward model.')
@click.option('--forward-model-alpha-lr',
              default=0.01,
              help='The learning rate of the lagrange multiplier.')
@click.option('--forward-model-overestimation-limit',
              default=0.5,
              help='The target used when tuning the lagrange multiplier.')
@click.option('--forward-model-noise-std',
              default=0.0,
              help='Standard deviation of continuous noise added to '
                   'designs when training the forward model.')
@click.option('--forward-model-batch-size',
              default=32,
              help='The batch size used when training the forward model.')
@click.option('--forward-model-val-size',
              default=200,
              help='The number of samples in the forward model '
                   'validation set.')
@click.option('--forward-model-epochs',
              default=50,
              help='The number of epochs to train the forward model.')
@click.option('--evaluation-samples',
              default=128,
              help='The samples to generate when solving the model-based '
                   'optimization problem.')
def coms_cleaned(
        logging_dir="coms-cleaned",
        task="HopperController-Exact-v0",
        task_relabel=False,
        normalize_ys=True,
        normalize_xs=True,
        in_latent_space=False,
        vae_hidden_size=256,
        vae_latent_size=256,
        vae_activation="relu",
        vae_kernel_size=3,
        vae_num_blocks=4,
        vae_lr=0.0003,
        vae_beta=1.0,
        vae_batch_size=32,
        vae_val_size=200,
        vae_epochs=10,
        particle_lr=0.05,
        particle_train_gradient_steps=1,
        particle_evaluate_gradient_steps=1,
        particle_entropy_coefficient=0.0,
        forward_model_activations=("relu", "relu"),
        forward_model_hidden_size=2048,
        forward_model_final_tanh=False,
        forward_model_lr=0.0003,
        forward_model_alpha=1.0,
        forward_model_alpha_lr=0.01,
        forward_model_overestimation_limit=0.5,
        forward_model_noise_std=0.0,
        forward_model_batch_size=32,
        forward_model_val_size=200,
        forward_model_epochs=10,
        evaluation_samples=128):
    """Solve a Model-Based Optimization problem using the method:
    Conservative Objective Models (COMs).

    """

    # store the command line params in a dictionary
    params = dict(
        logging_dir=logging_dir,
        task=task,
        task_relabel=task_relabel,
        normalize_ys=normalize_ys,
        normalize_xs=normalize_xs,
        in_latent_space=in_latent_space,
        vae_hidden_size=vae_hidden_size,
        vae_latent_size=vae_latent_size,
        vae_activation=vae_activation,
        vae_kernel_size=vae_kernel_size,
        vae_num_blocks=vae_num_blocks,
        vae_lr=vae_lr,
        vae_beta=vae_beta,
        vae_batch_size=vae_batch_size,
        vae_val_size=vae_val_size,
        vae_epochs=vae_epochs,
        particle_lr=particle_lr,
        particle_train_gradient_steps=
        particle_train_gradient_steps,
        particle_evaluate_gradient_steps=
        particle_evaluate_gradient_steps,
        particle_entropy_coefficient=
        particle_entropy_coefficient,
        forward_model_activations=forward_model_activations,
        forward_model_hidden_size=forward_model_hidden_size,
        forward_model_final_tanh=forward_model_final_tanh,
        forward_model_lr=forward_model_lr,
        forward_model_alpha=forward_model_alpha,
        forward_model_alpha_lr=forward_model_alpha_lr,
        forward_model_overestimation_limit=
        forward_model_overestimation_limit,
        forward_model_noise_std=forward_model_noise_std,
        forward_model_batch_size=forward_model_batch_size,
        forward_model_val_size=forward_model_val_size,
        forward_model_epochs=forward_model_epochs,
        evaluation_samples=evaluation_samples)

    # create the logger and export the experiment parameters
    logger = Logger(logging_dir)
    with open(os.path.join(logging_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    # create a model-based optimization task
    task = StaticGraphTask(task, relabel=task_relabel)

    if normalize_ys:
        task.map_normalize_y()
    if task.is_discrete and not in_latent_space:
        task.map_to_logits()
    if normalize_xs:
        task.map_normalize_x()

    x = task.x
    y = task.y

    if task.is_discrete and in_latent_space:

        vae_model = SequentialVAE(
            task, hidden_size=vae_hidden_size,
            latent_size=vae_latent_size, activation=vae_activation,
            kernel_size=vae_kernel_size, num_blocks=vae_num_blocks)

        vae_trainer = VAETrainer(
            vae_model, optim=tf.keras.optimizers.Adam,
            lr=vae_lr, beta=vae_beta)

        # create the training task and logger
        train_data, val_data = build_pipeline(
            x=x, y=y, batch_size=vae_batch_size,
            val_size=vae_val_size)

        # estimate the number of training steps per epoch
        vae_trainer.launch(train_data, val_data,
                           logger, vae_epochs)

        # map the x values to latent space
        x = vae_model.encoder_cnn.predict(x)[0]

        mean = np.mean(x, axis=0, keepdims=True)
        standard_dev = np.std(x - mean, axis=0, keepdims=True)
        x = (x - mean) / standard_dev

    input_shape = x.shape[1:]

    # compute the normalized learning rate of the model
    particle_lr = particle_lr * np.sqrt(np.prod(input_shape))

    # make a neural network to predict scores
    forward_model = ForwardModel(
        input_shape, activations=forward_model_activations,
        hidden_size=forward_model_hidden_size,
        final_tanh=forward_model_final_tanh)

    # make a trainer for the forward model
    trainer = ConservativeObjectiveModel(
        forward_model, forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=forward_model_lr, alpha=forward_model_alpha,
        alpha_opt=tf.keras.optimizers.Adam, alpha_lr=forward_model_alpha_lr,
        overestimation_limit=forward_model_overestimation_limit,
        particle_lr=particle_lr, noise_std=forward_model_noise_std,
        particle_gradient_steps=particle_train_gradient_steps,
        entropy_coefficient=particle_entropy_coefficient)

    # create a data set
    train_data, validate_data = build_pipeline(
        x=x, y=y, batch_size=forward_model_batch_size,
        val_size=forward_model_val_size)

    # train the forward model
    trainer.launch(train_data, validate_data,
                   logger, forward_model_epochs)

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=evaluation_samples)[1]
    initial_x = tf.gather(x, indices, axis=0)
    initial_y = tf.gather(y, indices, axis=0)
    xt = initial_x

    scores = []
    predictions = []

    score = task.predict(xt)
    if normalize_ys:
        initial_y = task.denormalize_y(initial_y)
        score = task.denormalize_y(score)
    logger.record(f"dataset_score", initial_y, 0, percentile=True)
    logger.record(f"score", score, 0, percentile=True)

    for step in range(particle_evaluate_gradient_steps):

        # update the set of solution particles
        xt = trainer.optimize(xt, 1, training=False)
        final_xt = trainer.optimize(
            xt, particle_train_gradient_steps, training=False)

        # evaluate the solutions found by the model
        score = task.predict(xt)
        prediction = forward_model(xt, training=False).numpy()
        final_prediction = forward_model(final_xt, training=False).numpy()
        if normalize_ys:
            score = task.denormalize_y(score)
            prediction = task.denormalize_y(prediction)
            final_prediction = task.denormalize_y(final_prediction)

        # record the prediction and score to the logger
        logger.record(f"score", score, step, percentile=True)
        logger.record(f"solver/model_to_real",
                      spearman(prediction[:, 0], score[:, 0]), step)
        logger.record(f"solver/distance",
                      tf.linalg.norm(xt - initial_x), step)
        logger.record(f"solver/prediction",
                      prediction, step)
        logger.record(f"solver/model_overestimation",
                      final_prediction - prediction, step)
        logger.record(f"solver/overestimation",
                      prediction - score, step)

        scores.append(score)
        predictions.append(prediction)

        # save the model predictions and scores to be aggregated later
        np.save(os.path.join(logging_dir, "scores.npy"),
                np.concatenate(scores, axis=1))
        np.save(os.path.join(logging_dir, "predictions.npy"),
                np.stack(predictions, axis=1))


# run COMs using the command line interface
if __name__ == '__main__':
    coms_cleaned()
