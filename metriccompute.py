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

def average_point_all(id, metric, task):
    experiment_id = id
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()

    df_new = df[df.tag.str.endswith(metric)]
    print(df_new.run)
    n = df_new.run.apply(lambda run: run[18:]) #21 #18
    mmd = df_new.run.apply(lambda run: float(run[21:])) #21
    plt.figure(figsize=(16, 6))
    sns.pointplot(data=df_new, x=mmd, y="value", hue = n, estimator=np.mean).set_title(metric+" "+task)
    plt.savefig(task+".jpg")

def getnum(df):
    df_prediction = df[df.tag.str.endswith("solver/prediction/mean")]
    prediction_value = float(np.mean(df_prediction.value))
    print(prediction_value)

    df_prediction_error = df[df.tag.str.endswith("validate/mse/mean")]
    df_prediction_error = df_prediction_error[df_prediction_error.step==299]
    df_prediction_error_value = float(np.sqrt(np.mean(df_prediction_error.value)))
    print(df_prediction_error_value)

    df_mmd_loss = df[df.tag.str.endswith("validate/mmd/mean")]
    df_mmd_loss = df_mmd_loss[df_mmd_loss.step==299]
    df_mmd_loss_value = float(np.sqrt(np.mean(df_mmd_loss.value)))
    print(df_mmd_loss_value)

    print("result:")
    print(prediction_value-df_prediction_error_value-df_mmd_loss_value)
    return prediction_value-400*df_prediction_error_value-500*df_mmd_loss_value


experiment_id = "jsa9OVIoTyyAlDjWv2ZbCQ"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
mmd_list = [0, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 500, 2000, 5000, 10000]
metric_result = []
for mmd in mmd_list:
    df_here = df[df.run.str.endswith("mmd"+str(mmd))]
    mmd_value = getnum(df_here)
    print(mmd_value)
    print("\n")
    metric_result.append(mmd_value)
print(mmd_list)
print(metric_result)
plt.figure(figsize=(16, 6))
sns.pointplot(x = mmd_list, y = metric_result)
plt.savefig("metric.jpg")

#average_point_all(experiment_id, "score/score/100th", "ant")




    