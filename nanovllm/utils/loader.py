"""模型权重加载器：从 safetensors 文件加载，支持融合参数映射。"""

import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载：直接 copy。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """从 safetensors 加载模型权重。

    融合参数处理流程（以 QKV 为例）：
    1. 读取 "q_proj.weight" 等独立权重
    2. 通过 packed_modules_mapping 找到目标融合参数 "qkv_proj"
    3. 调用该参数的 weight_loader(param, weight, shard_id="q")
    4. weight_loader 负责 TP 分片提取和正确位置写入
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查是否为融合参数（如 q_proj → qkv_proj）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # v = (target_param_name, shard_id)
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 非融合参数：直接查找并加载
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
