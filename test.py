from design_baselines.cbas import cbas
from design_baselines.rep import rep
from design_baselines.rep_coms_cleaned import coms_cleaned
coms_cleaned(logging_dir = "data",
                        task='HopperController-Exact-v0', #ToyContinuous-Exact-V0,#'AntMorphology-Exact-v0', #Discrete: TFBind8-Exact-v0
                        task_relabel=True,
                        normalize_ys=True,
                        normalize_xs=True,
                        in_latent_space=False,
                        vae_hidden_size=64,
                        vae_latent_size=256,
                        vae_activation='relu',
                        vae_kernel_size=3,
                        vae_num_blocks=4,
                        vae_lr=0.0003,
                        vae_beta=1.0,
                        vae_batch_size=32,
                        vae_val_size=200,
                        vae_epochs=10,
                        particle_lr=0.05,
                        particle_train_gradient_steps=50,
                        particle_evaluate_gradient_steps=50,
                        particle_entropy_coefficient=0.0,
                        forward_model_activations=['relu', 'relu'],
                        forward_model_hidden_size=2048,
                        forward_model_final_tanh=False,
                        forward_model_lr=0.0003,
                        forward_model_alpha=0.1,
                        forward_model_alpha_lr=0.01,
                        forward_model_overestimation_limit=0.5,
                        forward_model_noise_std=0.0,
                        forward_model_batch_size=128,
                        forward_model_val_size=200,
                        forward_model_epochs=10,
                        evaluation_samples=128,
                        fast=False,
                        latent_space_size=[20,1],
                        rep_model_activations=['relu', 'relu'],
                        rep_model_lr=0.0003,
                        rep_model_hidden_size=2048,
                        policy_model_lr=0.0003,
                        noise_input = [1, 10]
)

"""
rep(logging_dir="data",
                        task='HopperController-Exact-v0', #'AntMorphology-Exact-v0', #Discrete: TFBind8-Exact-v0
                        task_relabel=True,
                        normalize_ys=True,
                        normalize_xs=True,
                        latent_space_size=[20,1],
                        noise_shape=[1,10],
                        in_latent_space=False,
                        vae_hidden_size=64,
                        vae_latent_size=256,
                        vae_activation='relu',
                        vae_kernel_size=3,
                        vae_num_blocks=4,
                        vae_lr=0.0003,
                        vae_beta=1.0,
                        vae_batch_size=32,
                        vae_val_size=200,
                        vae_epochs=10,
                        particle_lr=0.05,
                        particle_train_gradient_steps=50,
                        particle_evaluate_gradient_steps=50,
                        particle_entropy_coefficient=0.0,
                        forward_model_activations=['relu', 'relu'],
                        forward_model_hidden_size=2048,
                        forward_model_final_tanh=False,
                        forward_model_lr=0.0003,
                        rep_model_activations=['relu', 'relu'],
                        rep_model_hidden_size=2048,
                        rep_model_final_tanh=False,
                        rep_model_lr=0.0003,
                        policy_model_activations=['relu', 'relu'],
                        policy_model_hidden_size=2048,
                        policy_model_final_tanh=False,
                        policy_model_lr=0.0003,
                        forward_model_noise_std=0.0,
                        forward_model_batch_size=32,
                        forward_model_val_size=200,
                        forward_model_epochs=25,
                        evaluation_samples=128,
                        fast=False
)
"""

# cbas({
#         "logging_dir"="data",
#         "normalize_ys"=True,
#         "normalize_xs"=False,
#         "task"="TFBind8-Exact-v0",
#         "task_kwargs"={"relabel"=False},
#         "bootstraps"=5,
#         "val_size"=200,
#         "ensemble_batch_size"=100,
#         "vae_batch_size"=100,
#         "embedding_size"=256,
#         "hidden_size"=256,
#         "num_layers"=1,
#         "initial_max_std"=0.2,
#         "initial_min_std"=0.1,
#         "ensemble_lr"=0.0003,
#         "ensemble_epochs"=100,
#         "latent_size"=32,
#         "vae_lr"=0.0003,
#         "vae_beta"=1.0,
#         "offline_epochs"=200,
#         "online_batches"=10,
#         "online_epochs"=10,
#         "iterations"=20,
#         "percentile"=80.0,
#         "solver_samples"=128, "do_evaluation"=True})