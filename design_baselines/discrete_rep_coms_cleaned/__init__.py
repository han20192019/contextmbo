from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.rep_coms_cleaned.trainers import ConservativeObjectiveModel
from design_baselines.rep_coms_cleaned.trainers import VAETrainer
from design_baselines.rep_coms_cleaned.nets import ForwardModel
from design_baselines.rep_coms_cleaned.nets import RepModel
from design_baselines.rep_coms_cleaned.nets import SequentialVAE
from design_baselines.rep_coms_cleaned.nets import PolicyContinuousForwardModel
import tensorflow as tf
import numpy as np
import os
import click
import json
from tensorboard.plugins import projector
import random

def visualize1(data, labels, num):
    num = str(num)
    print(data.shape)
    print(labels.shape)
    # Set up a logs directory, so Tensorboard knows where to look for files.
    log_dir='/nfs/kun2/users/hanqi2019/input_graph'
    d1 = 'metadata.tsv'
    d2 = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    #d2 = "var1"
    d3 = "embedding.ckpt"
    print(d1)
    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, d1), "w") as f:
        for subwords in labels:
            f.write("{}\n".format(subwords))
    
    # Save the weights we want to analyze as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, here
    # we will remove this value.
    weights = tf.Variable(data)
    # Create a checkpoint from embedding, the filename and key are the
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, d3))

    # Set up config.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = d2
    embedding.metadata_path = d1
    projector.visualize_embeddings(log_dir, config)

def visualize2(data, labels, num):
    num = str(num)
    print(data.shape)
    print(labels.shape)
    # Set up a logs directory, so Tensorboard knows where to look for files.
    log_dir='/nfs/kun2/users/hanqi2019/solution_graph'
    d1 = 'metadata.tsv'
    d2 = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    #d2 = "var1"
    d3 = "embedding.ckpt"
    print(d1)
    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, d1), "w") as f:
        for subwords in labels:
            f.write("{}\n".format(subwords))

    # Save the weights we want to analyze as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, here
    # we will remove this value.
    weights = tf.Variable(data)
    # Create a checkpoint from embedding, the filename and key are the
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, d3))

    # Set up config.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = d2
    embedding.metadata_path = d1
    projector.visualize_embeddings(log_dir, config)


def coms_cleaned(
        logging_dir,
        task,
        task_relabel,
        normalize_ys,
        normalize_xs,
        in_latent_space,
        vae_hidden_size,
        vae_latent_size,
        vae_activation,
        vae_kernel_size,
        vae_num_blocks,
        vae_lr,
        vae_beta,
        vae_batch_size,
        vae_val_size,
        vae_epochs,
        particle_lr,
        particle_train_gradient_steps,
        particle_evaluate_gradient_steps,
        particle_entropy_coefficient,
        forward_model_activations,
        forward_model_hidden_size,
        forward_model_final_tanh,
        forward_model_lr,
        forward_model_alpha,
        forward_model_alpha_lr,
        forward_model_overestimation_limit,
        forward_model_noise_std,
        forward_model_batch_size,
        forward_model_val_size,
        forward_model_epochs,
        evaluation_samples,
        fast,
        latent_space_size,
        rep_model_activations,
        rep_model_lr,
        rep_model_hidden_size,
        noise_input,
        policy_model_lr,
        mmd_param,
        seed=10):
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
        evaluation_samples=evaluation_samples,
        fast=fast)

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
    print(task.is_discrete)


    input_shape = x.shape[1:]
    print("input_shape:")
    print(input_shape)

    output_shape = latent_space_size
    
    # make a neural network to predict scores
    rep_model_final_tanh = False
    rep_model = RepModel(
        input_shape, output_shape, activations=rep_model_activations,
        hidden_size=rep_model_hidden_size,
        final_tanh=rep_model_final_tanh)

    forward_model = ForwardModel(
        output_shape, activations=forward_model_activations,
        hidden_size=forward_model_hidden_size,
        final_tanh=forward_model_final_tanh)
    
    policy_model = PolicyContinuousForwardModel(noise_input, task) 

    # compute the normalized learning rate of the model
    particle_lr = particle_lr * np.sqrt(np.prod(input_shape))

    # make a trainer for the forward model
    trainer = ConservativeObjectiveModel(mmd_param = mmd_param, policy_model = policy_model, 
        policy_model_lr = policy_model_lr, rep_model=rep_model, 
        rep_model_lr=rep_model_lr,
        forward_model=forward_model, forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=forward_model_lr, alpha=forward_model_alpha,
        alpha_opt=tf.keras.optimizers.Adam, alpha_lr=forward_model_alpha_lr,
        overestimation_limit=forward_model_overestimation_limit,
        particle_lr=particle_lr, noise_std=forward_model_noise_std,
        particle_gradient_steps=particle_train_gradient_steps,
        entropy_coefficient=particle_entropy_coefficient)

    #forward_model_val_size = int(task.x.shape[0]*0.3)
    # create a data set

    train_data, validate_data = build_pipeline(
        x=x, y=y, batch_size=forward_model_batch_size,
        val_size=forward_model_val_size)

    np.random.seed(seed)
    # train the forward model
    trainer.launch(train_data, validate_data,
                   logger, forward_model_epochs)

    # select the top k initial designs from the dataset
    x = task.x
    y = task.y
    indices = tf.math.top_k(y[:, 0], k=evaluation_samples)[1]
    initial_x = tf.gather(x, indices, axis=0)
    initial_y = tf.gather(y, indices, axis=0)
    xt = initial_x


    scores = []
    predictions = []

    solution = xt

    score = task.predict(solution)

    if normalize_ys:
        initial_y = task.denormalize_y(initial_y)
        score = task.denormalize_y(score)

    logger.record(f"dataset_score", initial_y, 0, percentile=True)
    logger.record(f"score", score, 0, percentile=True)

    xt_ori = trainer.optimize(xt, 1, training=False)
    xt_ori_rep = rep_model(xt_ori, training = False)
    prediction_ori = forward_model(xt_ori_rep, training=False).numpy()

    #visualize1(rep_model(x, training = False), y, 0)
    # add zero x
    score = task.predict(xt)
    logger.record(f"score/score", score, 0, percentile=True)
    for step in range(0, 1 + particle_evaluate_gradient_steps):

        # update the set of solution particles
        xt = trainer.optimize(xt, 1, training=False)
        final_xt = trainer.optimize(
            xt, particle_train_gradient_steps, training=False)

        
        #solution = final_xt
        solution = xt

        np.save(os.path.join(logging_dir, "testrep.npy"), solution)

        # evaluate the solutions found by the model
        score = task.predict(solution)

        xt_rep = rep_model(solution, training = False)

        #if step == 5:
            #visualize2(xt_rep, score, step+1)

        prediction = forward_model(xt_rep, training=False).numpy()
        print(step)
        mean_score = tf.reduce_mean(score)
        print(mean_score)

        if normalize_ys:
            score = task.denormalize_y(score)
            prediction = task.denormalize_y(prediction)

        # record the prediction and score to the logger
        logger.record(f"score/score", score, step+1, percentile=True)
        logger.record(f"score/score_mean", mean_score, step+1, percentile=True)
        logger.record(f"solver/model_to_real",
                        spearman(prediction[:, 0], score[:, 0]), step+1)
        logger.record(f"solver/distance",
                        tf.linalg.norm(xt - initial_x), step+1)
        logger.record(f"solver/prediction",
                        prediction, step+1)
        logger.record(f"solver/model_overestimation",
                        prediction_ori - prediction, step+1)
        logger.record(f"solver/overestimation",
                        prediction - score, step+1)


        scores.append(score)
        predictions.append(prediction)

        # save the model predictions and scores to be aggregated later
        np.save(os.path.join(logging_dir, "testrep.npy"),
                np.concatenate(scores, axis=1))
        np.save(os.path.join(logging_dir, "testrep.npy"),
                np.stack(predictions, axis=1))


# run COMs using the command line interface
if __name__ == '__main__':
    coms_cleaned()
