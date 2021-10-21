import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

df = pd.read_csv("beauty_new.csv")
df['beauty_scores'] = df['beauty_scores'].str.split(',').apply(lambda x: [int(i, 16) for i in x])
df_m = df
df_m['beauty_scores'] = [np.array(x).mean() for x in df.beauty_scores.values]
df_m['beauty_scores'] = df_m['beauty_scores'].round().astype(int)
df_m['beauty_scores'] = df_m['beauty_scores'] - 1
df_m['#flickr_photo_id'] = df_m['#flickr_photo_id'].astype(str) + ".jpg"

df_m = df_m.set_index('#flickr_photo_id')

df_m = df_m.drop(["Unnamed: 0"], axis=1)
df_m.index.name = None
print(df_m)
df_m.to_csv ('aes.csv')
