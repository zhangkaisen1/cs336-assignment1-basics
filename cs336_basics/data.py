import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    x: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]

    # 随机选择 batch_size 个起始位置
    # 因为 Data Loader 的目标不是把数据按顺序读一遍，而是从超长 token 序列中不断采样训练样本。
    # 每个起始位置 i 必须满足：0 <= i <= n - context_length - 1
    starts = torch.randint(
        low=0,
        high=n - context_length, 
        size=(batch_size,),
    )

    # 根据每个起始位置，截取长度为 context_length 的序列
    inputs = torch.stack([
        torch.from_numpy(x[i:i + context_length])
        for i in starts
    ])

    # target 比 input 向右移动一位
    targets = torch.stack([
        torch.from_numpy(x[i + 1:i + context_length + 1])
        for i in starts
    ])

    # 移动到指定 device
    inputs = inputs.to(device)
    targets = targets.to(device)

    return inputs, targets