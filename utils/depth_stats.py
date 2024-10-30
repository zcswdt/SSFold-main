from params import *
import torch
import numpy as np


data = torch.load(train_dataset_path)

maxes = []
mins = []
for i, e in enumerate(data):
    print("{}/{}".format(i, len(data)))
    depth = e.obs[1]
    maxes.append(depth.max())
    mins.append(depth.min())

print("Max:")
print(np.max(maxes))
print("Min:")
print(np.min(mins))

