import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import numpy as np

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

def getnum(df, c1, c2, norm):
    df_prediction = df[df.tag.str.endswith("solver/prediction/mean")]
    prediction_value = float(np.mean(df_prediction.value))

    df_prediction_error = df[df.tag.str.endswith("validate/mse/mean")]
    df_prediction_error = df_prediction_error[df_prediction_error.step==299]
    df_prediction_error_value = float(np.sqrt(np.mean(df_prediction_error.value)))

    df_mmd_loss = df[df.tag.str.endswith("validate/mmd/mean")]
    df_mmd_loss = df_mmd_loss[df_mmd_loss.step==299]
    df_mmd_loss_value = float(np.sqrt(np.mean(df_mmd_loss.value)))
    
    print("!!!")
    print(prediction_value)
    print(c1*df_prediction_error_value)
    print(c2*df_mmd_loss_value)
    print("!!!")
    
    
    return prediction_value-c1*df_prediction_error_value-c2*df_mmd_loss_value

def score(arr1, arr2):
    corr, _ = kendalltau(arr1, arr2)
    return corr

def getlist(df):
    df_prediction = df[df.tag.str.endswith("solver/prediction/mean")]
    prediction_value = float(np.mean(df_prediction.value))

    df_prediction_error = df[df.tag.str.endswith("validate/mse/mean")]
    df_prediction_error = df_prediction_error[df_prediction_error.step==299]
    df_prediction_error_value = float(np.sqrt(np.mean(df_prediction_error.value)))

    df_mmd_loss = df[df.tag.str.endswith("validate/mmd/mean")]
    df_mmd_loss = df_mmd_loss[df_mmd_loss.step==299]
    df_mmd_loss_value = float(np.sqrt(np.mean(df_mmd_loss.value)))
    
    return prediction_value, df_prediction_error_value, df_mmd_loss_value

ant_experiment_id = "jsa9OVIoTyyAlDjWv2ZbCQ"
kitty_experiment_id = "M3gblL6oTLGH1udiR53YFQ"
conductor_experiment_id = "1KynIKW8Qr2mBL3rCr7GUQ"
ant_norm = [111.6503758783241, 113.06675129193381, 128.7481167033942, 133.32987696547826, 154.12838915425644, 152.81089562611913, 180.35215145846115, 245.71503236977478, 327.6172781904193, 461.62039078339984, 624.362343777582, 694.3674782902551]
super_norm = [93.85806423638498, 89.5465799456882, 87.34797726296222, 85.86365953749944, 92.21180787314759, 95.98301957131872, 154.70445893769124, 239.62842175003948, 451.16946136109016, 1174.6312275176704, 1361.1276216953654, 1660.40634542224]
kitty_norm = [115.5884534541942, 125.65872170257641, 124.87369441041548, 125.71582940610196, 136.93282728752214, 147.01824977060724, 182.6496616495976, 247.14004588475854, 306.941156775256, 461.0420210135832, 618.9618731750891, 742.6304087800985]

experiment = tb.data.experimental.ExperimentFromDev(ant_experiment_id)
df = experiment.get_scalars()
best_score = -100
best_comb = [0,0]
best_result = []

mmd_list = [0, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 500, 2000, 5000, 10000]

alist = []
blist = []
clist = []
for mmd in mmd_list:
    df_here = df[df.run.str.endswith("mmd"+str(mmd))]
    a,b,c = getlist(df_here)
    alist.append(a)
    blist.append(b)
    clist.append(c)

plt.figure(figsize=(16, 6))
sns.pointplot(x = mmd_list, y = alist).set_title("ant prediction value")
plt.savefig("antpredictionvalue.jpg")

plt.figure(figsize=(16, 6))
sns.pointplot(x = mmd_list, y = blist).set_title("ant prediction error")
plt.savefig("antpredictionerror.jpg")

plt.figure(figsize=(16, 6))
sns.pointplot(x = mmd_list, y = clist).set_title("ant mmd loss")
plt.savefig("antmmdloss.jpg")


truth = []
for mmd in mmd_list:
    df_here = df[df.run.str.endswith("mmd"+str(mmd))]
    truth.append(gettruth(df_here))
truth_best_index = np.argmax(truth)
print(truth_best_index)

for c1 in range(500, 1000, 100):
    for c2 in range(500, 1000, 100):
        metric_result = []
        count = 0
        for mmd in mmd_list:
            df_here = df[df.run.str.endswith("mmd"+str(mmd))]
            mmd_value = getnum(df_here, c1, c2, kitty_norm[count])
            metric_result.append(mmd_value)
            count = count+1
        score_here = score(metric_result, truth)
        print(score_here)
        print([c1,c2])
        index_here = np.argmax(metric_result)
        print(index_here)
        if index_here == truth_best_index:
            print("find")
            print([c1,c2])
            break
        if score_here>best_score:
            best_score = score_here
            best_comb = [c1,c2]
            best_result = metric_result
print(best_score)
print(best_comb)
print(best_result)
print(truth)
print(mmd_list)


plt.figure(figsize=(16, 6))
sns.pointplot(x = mmd_list, y = metric_result)
plt.savefig("antmetric.jpg")
plt.figure(figsize=(16, 6))
sns.pointplot(x = mmd_list, y = truth)
plt.savefig("anttruth.jpg")


        






    