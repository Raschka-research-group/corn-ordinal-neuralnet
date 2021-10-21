import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

df = pd.read_csv("aes.csv",index_col = 0)
train, validate, test =np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.85*len(df))])
print(len(train))
print(len(validate))
print(len(test))

train.to_csv ('aes_train.csv')
validate.to_csv('aes_valid.csv')
test.to_csv('aes_test.csv')