import torch
import torch.nn.functional as F
import math

N, H, W, C = 16, 128, 128, 256

for v in range(1, 100):
    v = v / 100.0
    mean = 0.3989 * torch.ones(N, C, H, W)
    std = math.sqrt(v)
    x = torch.normal(mean=mean, std=std).cuda()

    A_1 = x
    print(A_1.var(unbiased=False).item())
    del A_1

    A_2 = F.interpolate(x, size=(2 * H, 2 * W), mode="bilinear", align_corners=False)
    print(A_2.var(unbiased=False).item())
    del A_2

    A_4 = F.interpolate(x, size=(4 * H, 4 * W), mode="bilinear", align_corners=False)
    print(A_4.var(unbiased=False).item())
    del A_4

    A_8 = F.interpolate(x, size=(8 * H, 8 * W), mode="bilinear", align_corners=False)
    print(A_8.var(unbiased=False).item())
    del A_8

    print("")
    del x
