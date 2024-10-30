import torch
import numpy as np

data = torch.load('./data/square_cloth_buf.pt')


min_depth = 100000
max_depth = 0

for episode in data:
  for i in range(len(episode)):
    depth = episode[i]['obs']['depth']
    depth = depth[depth > 0]
    n_depth = episode[i]['nobs']['depth']
    n_depth = n_depth[n_depth > 0]
    min_depth = min(min_depth, np.min(depth), np.min(n_depth))
    print(min_depth)
    max_depth = max(max_depth, np.max(depth), np.max(n_depth))
    print(max_depth)
#
# for i, e in enumerate(data):
#     print("{}/{}".format(i, len(data)))
#     depth = e["obs"]["depth"]
#     maxes.append(depth.max())
#     mins.append(depth.min())

print("Max:")
print(max_depth)
print("Min:")
print(min_depth)

