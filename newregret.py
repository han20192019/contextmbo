import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import numpy as np
from tabulate import tabulate

from scipy.stats import kendalltau
from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
from design_bench.datasets.continuous.hopper_controller_dataset import HopperControllerDataset
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
    df_prediction = df[df.tag.str.endswith("score/score_mean")]
    prediction_value = float(np.mean(df_prediction.value))
    print(prediction_value)
    return prediction_value

def getnum(df, c1, c2):
    df_prediction = df[df.tag.str.endswith("solver/prediction/mean")]
    df_prediction = df_prediction[df_prediction.step==51]
    prediction_value = float(np.mean(df_prediction.value))
    #print(prediction_value)

    df_prediction_error = df[df.tag.str.endswith("validate/mse/mean")]
    df_prediction_error = df_prediction_error[df_prediction_error.step==299]
    df_prediction_error_value = float(np.mean(np.sqrt(df_prediction_error.value)))
    #print(df_prediction_error_value)

    df_mmd_loss = df[df.tag.str.endswith("validate/mmd_L2/mean")]
    df_mmd_loss = df_mmd_loss[df_mmd_loss.step==299]
    df_mmd_loss_value = float(np.mean(np.sqrt(df_mmd_loss.value)))
    #print(df_mmd_loss_value)
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

#super, hopper, ant
id_list = [["HZC7MpmMRz6DLdEmRz7msQ", "W14zN9HdQkCnvI4oWnZycA", "ed14ZiQxRkG607bZPa2rwg", "lGHbjJ0nQgS8uZoweeIxZg", "LHwzmfkoQGqaxs8X2td8CA"],["SvwafLZrRwyGyvxjaPCY1Q", "cKpoV6IWR9u0RmBrNeQQzw", "SnBcyrUXSjq0Y0YmQ1ahgA", "dtXhG5YRRMmLU0oPa5mzkg", "2Zi6udLdQEiNW0L4Y7Et7g"],["KCajn90sRqqq26GX2NKlMQ", "jOxLdM8MT9qxqOREyarKog", "yKq8Wx0CRYqTjNjRVAoGDQ", "AjmxHWc4SmCRE7dcOojw6Q"]]
seed_list = [[1,2,3,4,5],[1,2,3,4,5],[2,3,4,5]]
df_list = []
for id_num in id_list:
    temp = []
    for idd in id_num:
        experiment = tb.data.experimental.ExperimentFromDev(idd)
        df = experiment.get_scalars()
        temp.append(df)
    df_list.append(temp)

mmd_list = [0, 0.1, 10, 100, 1000, 5000, 10000]


task_list = ["super","hopper","ant"]
truth_list = [[0.442548, 0.427424, 0.444815, 0.419334, 0.398897, 0.40951, 0.388292], [0.74169, 1.31685, 0.812978, 1.14601, 0.499653, 0.493314, 0.523442], [0.685792, 0.857839, 0.936095, 0.948368, 0.876337, 0.917816, 0.904951]]
result = []

for c1 in [10,50,100,500,1000,2000,5000,10000,20000,50000]:
    temp = [c1]
    for c2 in [10,50,100,500,1000,2000,5000]:
        reg_sum = 0
        for i in range(3):
            task = task_list[i]
            truth = truth_list[i]
            metric_result = []
            df_temp = df_list[i]
            for mmd in mmd_list:
                s = 0
                s_num = 0
                for seed in seed_list[i]:
                    name = task+"rep64seed"+str(seed)+"mmd"+str(mmd)
                    df = df_temp[s_num]
                    df_here = df[df.run.str.endswith(name)]
                    #print(df_here)
                    mmd_value = getnum(df_here, c1, c2)
                    s += mmd_value
                    s_num += 1
                s = s/len(seed_list)
                metric_result.append(s)
            #print(metric_result)
            index_here = np.argmax(metric_result)
            metric_highest = truth[index_here]
            regret = np.max(truth) - metric_highest
            reg_sum += regret
        regret = reg_sum/3.0
        print(regret)
        temp.append(regret)
    result.append(temp)



print(tabulate(result, headers=['c1/c2',10,50,100,500,1000,2000,5000]))
