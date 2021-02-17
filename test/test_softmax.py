import math

import torch
from torch.nn import functional as F

num = torch.tensor([-math.inf, -math.inf, 0.2])
scores = F.softmax(num, dim=-1)
v = torch.tensor([0.2, 0.3, 0.5])
print(scores*v)

