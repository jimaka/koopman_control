import torch

# 1. 设定模型路径
model_path = "/home/elane/intelligent_shipping_ws/data_process/checkpoints_horizontal/final_model.pt"

# 2. 加载模型文件 (map_location='cpu' 确保在没有 GPU 的机器上也能打开)
checkpoint = torch.load(model_path, map_location='cpu')

# 3. 访问字典中的 scaler 部分
scaler_info = checkpoint.get('scaler')

if scaler_info is not None:
    x_mean = scaler_info.get('x_mean')
    print("x_mean 内容如下:")
    print(x_mean)
    x_std = scaler_info.get('x_std')
    print("x_std 内容如下:")
    print(x_std)
else:
    print("该模型文件中没有保存 scaler 信息（可能 cfg.normalize_data 为 False）。")