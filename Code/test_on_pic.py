import torch
import numpy as np
from custom_models import get_upsampling_weight


def test_get_upsampling_weight(picture):
    src = picture
    x = src.transpose(2, 0, 1)
    x = x[np.newaxis, :, :, :]
    x = torch.from_numpy(x).float()
    x = torch.autograd.Variable(x)

    in_channels = 3
    out_channels = 3
    kernel_size = 4

    m = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=2, bias=False)
    m.weight.data = get_upsampling_weight(
        in_channels, out_channels, kernel_size)

    y = m(x)

    y = y.data.numpy()
    y = y[0]
    y = y.transpose(1, 2, 0)
    dst = y.astype(np.uint8)

    assert abs(src.shape[0] * 2 - dst.shape[0]) <= 2
    assert abs(src.shape[1] * 2 - dst.shape[1]) <= 2

    return src, dst

