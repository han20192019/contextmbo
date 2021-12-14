from design_baselines.cbas import cbas
from design_baselines.rep import rep
from design_baselines.rep_coms_cleaned import coms_cleaned
seed = [1,2,3,4,5]
mmd_param = [0.01]
d = 32
name = "tf"
task = "TFBind8-Exact-v0"
for s in seed:
    for m in mmd_param:
        coms_cleaned(logging_dir = name+"/"+name+"rep"+str(d)+"seed"+str(s)+"mmd"+str(m),
                                task=task,
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
                                forward_model_epochs=500,
                                evaluation_samples=128,
                                fast=False,
                                latent_space_size=[d,1],
                                rep_model_activations=['relu', 'relu'],
                                rep_model_lr=0.0003,
                                rep_model_hidden_size=2048,
                                policy_model_lr=0.0003,
                                noise_input = [1, 10],
                                mmd_param = m,
                                seed = s
                    )