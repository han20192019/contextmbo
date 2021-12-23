from design_baselines.cbas import cbas
from design_baselines.rep import rep
from design_baselines.rep_coms_cleaned import coms_cleaned
seed = [1,2,3,4,5]
mmd_param = [2000] #[0.1, 0, 10, 500, 2000, 5000, 10000]
d_list = [64] #[16, 32, 64, 128, 256]
name = "ant"
task = "AntMorphology-Exact-v0"
for d in d_list:
    for s in seed:
        for m in mmd_param:
            coms_cleaned(logging_dir = "/nfs/kun2/users/hanqi2019/1223ant/seed"+str(s)+"/"+name+"rep"+str(d)+"seed"+str(s)+"mmd"+str(m),
                                    task=task,
                                    task_relabel=True,
                                    normalize_ys=True,
                                    normalize_xs=True,
                                    in_latent_space=False,
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
                                    forward_model_epochs=300,
                                    evaluation_samples=128,
                                    fast=True,
                                    latent_space_size=[d,1],
                                    rep_model_activations=['relu', 'relu'],
                                    rep_model_lr=0.0003,
                                    rep_model_hidden_size=2048,
                                    noise_input = [1, 10],
                                    mmd_param = m,
                                    seed = s
                        )