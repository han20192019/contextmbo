import tensorflow as tf
import numpy as np
import os
import click
import json
from tensorboard.plugins import projector
import random
s_list = [1,2,3,4,5]
d_list = [0,0.01,0.05,0.1,0.5,1,10,100,500,2000,5000,10000]
result = []
for m in d_list:
    sum_prod = 0
    count = 0
    for seed in s_list:
        count=count+1
        model_dir = "/nfs/kun2/users/hanqi2019/1223ant/seed"+str(seed)+"/ant"+"rep64"+"seed"+str(seed)+"mmd"+str(m)
        print(model_dir)
        trainer = tf.keras.models.load_model(model_dir)
        rep_model = trainer.rep_model
        forward_model = trainer.forward_model
        product = 1
        #for layer in forward_model.layers:
            #print(layer.name, layer)

        l = [forward_model.layers[1], forward_model.layers[3], forward_model.layers[5]]
        for layer in l:
            #print("svd")
            weight1 = layer.weights[0]
            #print(weight1)
            s = tf.linalg.svd(weight1)[0]
            s = s.numpy()
            max_s = np.max(s)
            #print(max_s)
            product = product*max_s
        print("!!!")
        print(product)
        sum_prod = sum_prod+product
        print(count)
    result.append(sum_prod/count)
print("average")
print(result)