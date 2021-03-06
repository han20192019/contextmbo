from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.rep.trainers import RepresentationLearningModel
from design_baselines.rep.nets import ForwardModel
from design_baselines.rep.nets import PolicyContinuousForwardModel
from design_baselines.rep.nets import PolicyDiscreteForwardModel
from design_baselines.rep.nets import RepModel
import tensorflow as tf
import numpy as np
import os
import click
import json
from sklearn.utils import shuffle


def rep(
        logging_dir,
        task,
        task_relabel,
        normalize_ys,
        normalize_xs,
        latent_space_size,
        noise_shape,
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
        forward_model_noise_std,
        forward_model_batch_size,
        forward_model_val_size,
        forward_model_epochs,
        rep_model_activations,
        rep_model_hidden_size,
        rep_model_final_tanh,
        rep_model_lr,
        policy_model_activations,
        policy_model_hidden_size,
        policy_model_final_tanh,
        policy_model_lr,
        evaluation_samples,
        fast):
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
        latent_space_size=latent_space_size,
        noise_shape=noise_shape,
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
        forward_model_noise_std=forward_model_noise_std,
        forward_model_batch_size=forward_model_batch_size,
        forward_model_val_size=forward_model_val_size,
        forward_model_epochs=forward_model_epochs,
        rep_model_activations=rep_model_activations,
        rep_model_hidden_size=rep_model_hidden_size,
        rep_model_final_tanh=rep_model_final_tanh,
        rep_model_lr=rep_model_lr,
        policy_model_activations=policy_model_activations,
        policy_model_hidden_size=policy_model_hidden_size,
        policy_model_final_tanh=policy_model_final_tanh,
        policy_model_lr=policy_model_lr,
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
    # if task.is_discrete and not in_latent_space:
    #     task.map_to_logits()
    if normalize_xs and not task.is_discrete:
        task.map_normalize_x()
    if task.is_discrete:
        task.map_to_integers()

    x = task.x
    y = task.y


    input_shape = x.shape[1:]
    output_shape = latent_space_size
    noise_input = noise_shape
    if task.is_discrete:
        input_shape = (input_shape[0], task.num_classes)
    
    #one round
    # make a neural network to predict scores
    rep_model = RepModel(
        input_shape, output_shape, activations=rep_model_activations,
        hidden_size=rep_model_hidden_size,
        final_tanh=rep_model_final_tanh)

    forward_model = ForwardModel(
        output_shape, activations=forward_model_activations,
        hidden_size=forward_model_hidden_size,
        final_tanh=forward_model_final_tanh)

    if task.is_discrete:
        policy_model = PolicyDiscreteForwardModel(task, input_shape) 
    else: 
        policy_model = PolicyContinuousForwardModel(noise_input, task) 

    # make a trainer for the forward model
    trainer = RepresentationLearningModel(task,
        rep_model, forward_model, policy_model,
        forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=forward_model_lr,
        rep_model_opt=tf.keras.optimizers.Adam,
        rep_model_lr=forward_model_lr, 
        policy_model_opt=tf.keras.optimizers.Adam,
        policy_model_lr=forward_model_lr)

    # create a data set
    train_data, validate_data = build_pipeline(
        x=x, y=y, batch_size=forward_model_batch_size,
        val_size=forward_model_val_size)
    print("train data shape:")
    print(x.shape)

    # train the forward model
    trainer.launch(train_data, validate_data,
                   logger, forward_model_epochs)


    ## evaluation TODO
    print("start evaluation")
    indices = tf.math.top_k(y[:, 0], k=evaluation_samples)[1]
    initial_x = tf.gather(x, indices, axis=0)
    initial_y = tf.gather(y, indices, axis=0)
    xt = initial_x  # (128, 60)
    score = task.predict(xt)
    if task.is_discrete:
        print("evaluation to be implemented")
    else:
        solution = policy_model.get_sample(size=evaluation_samples, training=False).numpy() # (128,1,60)
    solution = tf.reshape(solution, [solution.shape[0], solution.shape[2]])
    prediction = task.predict(solution) 

    if normalize_ys:
        score = task.denormalize_y(score)
        prediction = task.denormalize_y(prediction)
    print("results")
    print(score)
    print(prediction)

    # record the prediction and score to the logger
    step = 0
    logger.record(f"score", score, step, percentile=True)
    logger.record(f"solver/model_to_real",
                    spearman(prediction[:, 0], score[:, 0]), step)
    logger.record(f"solver/prediction",
                    prediction, step)
    logger.record(f"solver/overestimation",
                    prediction - score, step)

    

    """
    #cross-validation
    x, y = shuffle(x, y)
    print(x.shape)
    print(y.shape)
    val_size = int(len(x)/5)
    for i in range(5):
        print("round" + str(i))
        x = np.concatenate((x[val_size:], x[:val_size]), axis = 0)
        y = np.concatenate((y[val_size:], y[:val_size]), axis = 0)

        train_data, validate_data = build_pipeline(
            x=x, y=y, batch_size=forward_model_batch_size,
            val_size=val_size)

        rep_model = RepModel(
        input_shape, output_shape, activations=rep_model_activations,
        hidden_size=rep_model_hidden_size,
        final_tanh=rep_model_final_tanh)

        forward_model = ForwardModel(
            output_shape, activations=forward_model_activations,
            hidden_size=forward_model_hidden_size,
            final_tanh=forward_model_final_tanh)

        if task.is_discrete:
            policy_model = PolicyDiscreteForwardModel(task, input_shape) 
        else: 
            policy_model = PolicyContinuousForwardModel(noise_input, task) 
        
        trainer = RepresentationLearningModel(task,
        rep_model, forward_model, policy_model,
        forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=forward_model_lr,
        rep_model_opt=tf.keras.optimizers.Adam,
        rep_model_lr=forward_model_lr, 
        policy_model_opt=tf.keras.optimizers.Adam,
        policy_model_lr=forward_model_lr)

        trainer.launch(train_data, validate_data,
                   logger, forward_model_epochs)

        print("start evaluation")
        indices = tf.math.top_k(y[:, 0], k=evaluation_samples)[1]
        initial_x = tf.gather(x, indices, axis=0)
        initial_y = tf.gather(y, indices, axis=0)
        xt = initial_x  # (128, 60)
        score = task.predict(xt)
        if task.is_discrete:
            print("evaluation to be implemented")
        else:
            solution = policy_model.get_sample(size=evaluation_samples, training=False).numpy() # (128,1,60)
        solution = tf.reshape(solution, [solution.shape[0], solution.shape[2]])
        prediction = task.predict(solution) 

        if normalize_ys:
            score = task.denormalize_y(score)
            prediction = task.denormalize_y(prediction)
        print("results")
        print(score)
        print(prediction)

        # record the prediction and score to the logger
        step = 0
        logger.record(f"score", score, step, percentile=True)
        logger.record(f"solver/model_to_real",
                        spearman(prediction[:, 0], score[:, 0]), step)
        logger.record(f"solver/prediction",
                        prediction, step)
        logger.record(f"solver/overestimation",
                        prediction - score, step)
    """


