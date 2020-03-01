from IPython import embed
import torch

x = torch.zeros(4)
x[1] = 1
x[2] = 2
x[3] = 3
x.requires_grad = True
x_clone = x.clone()
x_clone = x[[1, 0, 3, 2]]
for _ in range(10):
    pred_x = (1 - x).sum()
    pred_x_clone = (1 - x_clone).sum()
    # x.retain_grad()
    # x_clone.retain_grad()
    pred_x.backward()
    # pred_x_clone.backward()
    embed()