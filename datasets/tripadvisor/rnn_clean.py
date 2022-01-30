import torch
import torch.nn.functional as F
import torchtext
import time
import random
import pandas as pd
random.seed(123)
df = pd.read_csv("tripadvisor_hotel_reviews.csv")
df.columns = ['TEXT_COLUMN_NAME', 'LABEL_COLUMN_NAME']
# df = df.drop(columns=['id'])
print(df.columns)
def sampling_k_elements(group, k=1400):
    if len(group) < k:
        return group
    return group.sample(k)

balanced = df.groupby('LABEL_COLUMN_NAME').apply(sampling_k_elements).reset_index(drop=True)

balanced.to_csv('tripadvisor_balanced.csv', index=None)