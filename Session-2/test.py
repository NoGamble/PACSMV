import torch

# 检查MPS是否可用
print(f"MPS可用: {torch.backends.mps.is_available()}")  
# 输出 True 表示支持

# 检查PyTorch是否已编译MPS支持
print(f"MPS已构建: {torch.backends.mps.is_built()}")   