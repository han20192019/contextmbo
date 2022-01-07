import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import numpy as np
from tabulate import tabulate

from scipy.stats import kendalltau
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

def gettruth(df):
    df_prediction = df[df.tag.str.endswith("score/score/100th")]
    prediction_value = float(np.mean(df_prediction.value))
    print(prediction_value)
    return prediction_value

def getnum(df, c1, c2):
    df_prediction = df[df.tag.str.endswith("solver/prediction/mean")]
    prediction_value = float(np.mean(df_prediction.value))

    df_prediction_error = df[df.tag.str.endswith("validate/mse/mean")]
    df_prediction_error = df_prediction_error[df_prediction_error.step==299]
    df_prediction_error_value = float(np.mean(np.sqrt(df_prediction_error.value)))

    df_mmd_loss = df[df.tag.str.endswith("validate/mmd/mean")]
    df_mmd_loss = df_mmd_loss[df_mmd_loss.step==299]
    df_mmd_loss_value = float(np.mean(np.sqrt(df_mmd_loss.value)))
    """
    print("!!!")
    print(prediction_value)
    print(c1*df_prediction_error_value)
    print(c2*df_mmd_loss_value)
    print("!!!")
    """
    return prediction_value-c1*df_prediction_error_value-c2*df_mmd_loss_value

def score(arr1, arr2):
    corr, _ = kendalltau(arr1, arr2)
    return corr

def getlist(df):
    df_prediction = df[df.tag.str.endswith("solver/prediction/mean")]
    prediction_value = float(np.mean(df_prediction.value))

    df_prediction_error = df[df.tag.str.endswith("validate/mse/mean")]
    df_prediction_error = df_prediction_error[df_prediction_error.step==299]
    df_prediction_error_value = float(np.mean(np.sqrt(df_prediction_error.value)))

    df_mmd_loss = df[df.tag.str.endswith("validate/mmd/mean")]
    df_mmd_loss = df_mmd_loss[df_mmd_loss.step==299]
    df_mmd_loss_value = float(np.mean(np.sqrt(df_mmd_loss.value)))
    
    return prediction_value, df_prediction_error_value, df_mmd_loss_value

kittyboth_id = "pN37wMiASYS5ViNKZfQErg"
antboth_id = "orkjp7prR3edP6knSg7ONA"
superboth_id = "HPyuxygNQeOEzIlDvLgBeA"
hopperboth_id = "8waOEPorTlyEnJXBj0RjLQ"

ant_id = "jsa9OVIoTyyAlDjWv2ZbCQ"
kitty_id = "jsa9OVIoTyyAlDjWv2ZbCQ"
super_id = "jsa9OVIoTyyAlDjWv2ZbCQ"
hopper_id = "byR51y9PSryEVEBxdsQBhg"

id_list = [kittyboth_id, antboth_id, superboth_id, hopperboth_id]
df_list = []
for id in id_list:
    experiment = tb.data.experimental.ExperimentFromDev(id)
    df = experiment.get_scalars()
    df_list.append(df)


#mmd_list = [0.01, 0.05, 0.1, 0.5, 1, 10, 100, 500, 2000, 5000, 10000]
mmd_list = [0.1, 10, 100, 1000, 5000, 10000]

truth_list = []
for i in range(4):
    truth = []
    df = df_list[i]
    for mmd in mmd_list:
        df_here = df[df.run.str.endswith("mmd"+str(mmd))]
        truth.append(gettruth(df_here))
    truth_list.append(truth)

result = []

for c1 in [10,50,100,500,1000,2000,5000,10000,20000,50000]:
    temp = [c1]
    for c2 in [10,50,100,500,1000,2000,5000]:
        reg_sum = 0
        for i in range(4):
            df = df_list[i]
            truth = truth_list[i]
            metric_result = []
            for mmd in mmd_list:
                df_here = df[df.run.str.endswith("mmd"+str(mmd))]
                mmd_value = getnum(df_here, c1, c2)
                metric_result.append(mmd_value)
            index_here = np.argmax(metric_result)
            metric_highest = truth[index_here]
            regret = np.max(truth) - metric_highest
            reg_sum += regret
        regret = reg_sum/4.0
        temp.append(regret)
    result.append(temp)

print(tabulate(result, headers=['c1/c2',10,50,100,500,1000,2000,5000]))


    
    




        






    