from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.permmdtraining_rep_coms_cleaned.trainers import ConservativeObjectiveModel
from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.context_trainAoptimizemmd_rep_coms_cleaned.trainers import ConservativeObjectiveModel
from design_baselines.context_trainAoptimizemmd_rep_coms_cleaned.nets import ForwardModel
from design_baselines.context_trainAoptimizemmd_rep_coms_cleaned.nets import RepModel1
from design_baselines.context_trainAoptimizemmd_rep_coms_cleaned.nets import RepModel2
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
        mmd_param,
        optmmd_param,
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
        fast=fast,
        latent_space_size = latent_space_size,
        rep_model_activations = rep_model_activations,
        rep_model_lr = rep_model_lr,
        rep_model_hidden_size = rep_model_hidden_size,
        noise_input = noise_input,
        mmd_param = mmd_param,
        seed = seed)

    # create the logger and export the experiment parameters
    logger = Logger(logging_dir)
    with open(os.path.join(logging_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    # create a model-based optimization task
    total = 2
    name_list = ["TFBind8-GP-v01", "TFBind8-GP-v02"]
    task_list = []
    for i in name_list:
        task1 = StaticGraphTask(i, relabel=task_relabel)
        task_list.append(task1)
    #all eight/four four

    if normalize_ys:
        for i in task_list:
            i.map_normalize_y()
    if task_list[0].is_discrete and not in_latent_space:
        for i in task_list:
            i.map_to_logits()
    if normalize_xs:
        for i in task_list:
            i.map_normalize_x()

    x_list = []
    y_list = []
    for i in task_list:
        x_list.append(i.x)
        y_list.append(i.y)

    x1 = task_list[0].x
    y1 = task_list[0].y


    x = list(x1)
    y = list(y1)
    for i in range(1, total):
        x += list(x_list[i])
        y += list(y_list[i])

    x = tf.constant(x)
    y = tf.constant(y)

    input_shape = x1.shape[1:]
    print("input_shape:")
    print(input_shape)

    #print(x)
    #print(y)

    output_shape = latent_space_size
    
    # make a neural network to predict scores
    rep_model_final_tanh = False
    rep_model1 = RepModel1(
        input_shape, (64,1), activations=rep_model_activations,
        hidden_size=rep_model_hidden_size,
        final_tanh=rep_model_final_tanh)

    rep_model2 = RepModel2(
        (72,1), output_shape, activations=rep_model_activations,
        hidden_size=rep_model_hidden_size,
        final_tanh=rep_model_final_tanh)

    forward_model = ForwardModel(
        output_shape, activations=forward_model_activations,
        hidden_size=forward_model_hidden_size,
        final_tanh=forward_model_final_tanh)

    # compute the normalized learning rate of the model
    particle_lr = particle_lr * np.sqrt(np.prod(input_shape))

    # select the top k initial designs from the dataset
    initial_x_list = []
    initial_y_list = []

    for i in range(total):
        indices1 = tf.math.top_k(y_list[i][:, 0], k=evaluation_samples)[1]
        initial_x1 = tf.gather(x_list[i], indices1, axis=0)
        initial_y1 = tf.gather(y_list[i], indices1, axis=0)
        initial_x_list.append(initial_x1)
        initial_y_list.append(initial_y1)

    c_type = []
    for i in range(total):
        c1 = np.zeros(8, dtype='float32')
        c1[i] = 1
        c_type.append(c1)
    c = [c_type[0]]*x_list[0].shape[0]
    for i in range(1,total):
        c = c+[c_type[i]]*x_list[i].shape[0]
    c = tf.constant(c)
    print(c.shape)
    print(x.shape)
    print(y.shape)
    # make a trainer for the forward model
    trainer = ConservativeObjectiveModel(mmd_param = mmd_param, 
        rep_model1=rep_model1, rep_model2 = rep_model2,
        rep_model_lr=rep_model_lr,
        forward_model=forward_model, forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=forward_model_lr, alpha=forward_model_alpha,
        alpha_opt=tf.keras.optimizers.Adam, alpha_lr=forward_model_alpha_lr,
        overestimation_limit=forward_model_overestimation_limit,
        particle_lr=particle_lr, noise_std=forward_model_noise_std,
        particle_gradient_steps=particle_train_gradient_steps,
        entropy_coefficient=particle_entropy_coefficient, x_ori=x, optmmd_param = optmmd_param)

    # create a data set
    val_size = int(x1.shape[0]*0.3)
    train_data, validate_data = build_pipeline(c=c,
        x=x, y=y, batch_size=forward_model_batch_size,
        val_size=val_size)

    np.random.seed(seed)
    print("new")
    # train the forward model
    trainer.launch(train_data, validate_data,
                   logger, forward_model_epochs)
    

    x_final_list = initial_x_list
    y_final_list = initial_y_list
    score_list = []
    for i in range(2):
        c_here = np.zeros(8, dtype = 'float32')
        c_here[i]=1
        c_pass = tf.constant([c_here]*x_final_list[i].shape[0])
        print(c_pass)
        x_neg = trainer.optimize(False, c_pass, 
            x_final_list[i], 50, training=False)
        score = task1.predict(x_neg)
        score = task1.denormalize_y(score)
        score_list.append(score)
    print(score_list)



# run COMs using the command line interface
if __name__ == '__main__':
    coms_cleaned()

