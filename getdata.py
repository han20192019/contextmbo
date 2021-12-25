import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import numpy as np
"""
id is the experiment id
metric can be chosen from:
['score/score/100th' 'score/score/50th' 'score/score/80th'
 'score/score/90th' 
 'train/alpha/max' 'train/alpha/mean' 'train/alpha/min' 'train/alpha/std'
 'train/loss1/max' 'train/loss1/mean' 'train/loss1/min' 'train/loss1/std'
 'train/loss2/max' 'train/loss2/mean' 'train/loss2/min' 'train/loss2/std'
 'train/mmd_L1/max' 'train/mmd_L1/mean' 'train/mmd_L1/min'
 'train/mmd_L1/std' 'train/mmd_L2/max' 'train/mmd_L2/mean'
 'train/mmd_L2/min' 'train/mmd_L2/std' 'train/mse_L1/max'
 'train/mse_L1/mean' 'train/mse_L1/min' 'train/mse_L1/std'
 'train/mse_L2/max' 'train/mse_L2/mean' 'train/mse_L2/min'
 'train/mse_L2/std' 
 'validate/alpha/max' 'validate/alpha/mean''validate/alpha/min' 'validate/alpha/std' 
 'validate/loss1/max''validate/loss1/mean' 'validate/loss1/min' 'validate/loss1/std'
 'validate/loss2/max' 'validate/loss2/mean' 'validate/loss2/min'
 'validate/loss2/std' 'validate/mmd/max' 'validate/mmd/mean'
 'validate/mmd/min' 'validate/mmd/std' 'validate/mse/max'
 'validate/mse/mean' 'validate/mse/min' 'validate/mse/std']
 name = "mmd..."
 """
def plotmetric_line(id, metric, name):
    experiment_id = id
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()

    df_new = df[df.tag.str.endswith(metric)]
    df_new = df_new[df_new.run.str.endswith(name)]
    plt.figure(figsize=(16, 6))
    sns.lineplot(data=df_new, x="step", y="value", hue="run").set_title(metric+" "+name)
    plt.savefig("test1.jpg")

def plotmetric_point(id, metric, name):
    experiment_id = id
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()

    df_new = df[df.tag.str.endswith(metric)]
    df_new = df_new[df_new.run.str.endswith(name)]
    plt.figure(figsize=(16, 6))
    sns.pointplot(data=df_new, x="step", y="value", hue="run").set_title(metric+" "+name)
    plt.savefig("test2.jpg")

def average_line_single(id, metric, name):
    experiment_id = id
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()

    df_new = df[df.tag.str.endswith(metric)]
    df_new = df_new[df_new.run.str.endswith(name)]
    plt.figure(figsize=(16, 6))
    sns.lineplot(data=df_new, x="step", y="value", estimator='mean').set_title(metric+" "+name)
    plt.savefig("test3.jpg")

def average_line_all(id, metric):
    experiment_id = id
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()

    df_new = df[df.tag.str.endswith(metric)]
    n = df_new.run.apply(lambda run: run[19:])
    plt.figure(figsize=(16, 6))
    sns.lineplot(data=df_new, x="step", y="value", hue = n, estimator='mean').set_title(metric)
    plt.savefig("test4.jpg")

def average_point_single(id, metric, name):
    experiment_id = id
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()

    df_new = df[df.tag.str.endswith(metric)]
    df_new = df_new[df_new.run.str.endswith(name)]
    plt.figure(figsize=(16, 6))
    sns.pointplot(data=df_new, x="step", y="value", estimator=np.mean).set_title(metric+" "+name)
    plt.savefig("test5.jpg")

def average_point_all(id, metric):
    experiment_id = id
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()

    df_new = df[df.tag.str.endswith(metric)]
    n = df_new.run.apply(lambda run: run[19:])
    plt.figure(figsize=(16, 6))
    sns.pointplot(data=df_new, x="step", y="value", hue = n, estimator=np.mean).set_title(metric)
    plt.savefig("test6.jpg")

plotmetric_line("Vw8qFOMgRbCD7Kx9E0W18Q", "train/mmd_L1/mean", "mmd5000")
average_line_single("Vw8qFOMgRbCD7Kx9E0W18Q", "train/mmd_L1/mean", "mmd5000")
average_line_all("Vw8qFOMgRbCD7Kx9E0W18Q", "train/mmd_L1/mean")

plotmetric_point("Vw8qFOMgRbCD7Kx9E0W18Q", "score/score/100th", "mmd5000")
average_point_single("Vw8qFOMgRbCD7Kx9E0W18Q", "score/score/100th", "mmd5000")
average_point_all("Vw8qFOMgRbCD7Kx9E0W18Q", "score/score/100th")
