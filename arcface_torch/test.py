from torch import nn
import torch
import csv
from collections import defaultdict

identity_dict = defaultdict(list)

# idx = 0
current = 0
num_images = 0
with open('WebFace260M_dataset.csv', 'r') as f:
    r = csv.reader(f)
    for idx, row in enumerate(r):
        identity_dict[row[0]].append(row[1])
        now = int(row[1])
        num_images = num_images + 1
        if current != now:
            current = now
            if (((current % 1000) == 0) or (current > 990000)):
                print(current)
        # idx = idx + 1
            if current + 1 > 100: break

# print(num_images)
print(identity_dict)

for index, (key, value) in enumerate(identity_dict.items()):
    for v in value:
        print(v, index)