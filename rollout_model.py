# """
# 使用训练好的Koopman模型在特定飞行数据上进行滚转预测，并与真实值比较。

# 此脚本加载train_koopman.py保存的检查点，重建KoopmanModel，
# 使用保存的标准化器对输入进行标准化，并使用飞行的控制输入执行
# 潜在空间滚转：z_{t+1} = A z_t + B u_t。
# 解码后的状态被反标准化回原始尺度，并与飞行的真实状态逐分量绘制对比。

# 使用示例：
# - python rollout_model.py --ckpt checkpoints_horizontal/best_model.pt --flight 0
# - python rollout_model.py --ckpt path.pt --flight 5 --limit-steps 200 --save plots/rollout_f5.png
# - 默认模式是教师强制（单步）。使用--open-loop进行开环预测。

# 注意：
# - 本脚本专为水平面船舶运动设计（6维状态，4维控制）
# - 支持两种滚转模式：教师强制（一步预测）和开环（多步预测）
# """

# import argparse
# import os
# from typing import Tuple, Optional, Dict, Any

# import numpy as np
# import torch
# import matplotlib.pyplot as plt

# from pelican_torch_dataset import PelicanHorizontalTransitionDataset
# from koopman import HorizontalKoopmanModel, DeepKoopmanModel


# class StandardScaler:
#     """标准化器（零均值，单位方差）"""
    
#     def __init__(self, x_mean, x_std, u_mean, u_std, device: torch.device):
#         """
#         初始化标准化器
        
#         Args:
#             x_mean: 状态均值 (state_dim,)
#             x_std: 状态标准差 (state_dim,)
#             u_mean: 控制均值 (control_dim,)
#             u_std: 控制标准差 (control_dim,)
#             device: 计算设备
#         """
#         self.x_mean = torch.tensor(x_mean, dtype=torch.float32, device=device)
#         self.x_std = torch.tensor(x_std, dtype=torch.float32, device=device)
#         self.u_mean = torch.tensor(u_mean, dtype=torch.float32, device=device)
#         self.u_std = torch.tensor(u_std, dtype=torch.float32, device=device)
        
#         # 确保形状正确
#         if self.x_mean.dim() == 2:
#             self.x_mean = self.x_mean.squeeze(0)
#         if self.x_std.dim() == 2:
#             self.x_std = self.x_std.squeeze(0)
#         if self.u_mean.dim() == 2:
#             self.u_mean = self.u_mean.squeeze(0)
#         if self.u_std.dim() == 2:
#             self.u_std = self.u_std.squeeze(0)
        
#     def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
#         """标准化状态：x_norm = (x - mean) / std"""
#         return (x - self.x_mean) / self.x_std
    
#     def normalize_u(self, u: torch.Tensor) -> torch.Tensor:
#         """标准化控制：u_norm = (u - mean) / std"""
#         return (u - self.u_mean) / self.u_std
    
#     def denormalize_x(self, x_norm: torch.Tensor) -> torch.Tensor:
#         """反标准化状态：x = x_norm * std + mean"""
#         return x_norm * self.x_std + self.x_mean
    
#     def denormalize_u(self, u_norm: torch.Tensor) -> torch.Tensor:
#         """反标准化控制：u = u_norm * std + mean"""
#         return u_norm * self.u_std + self.u_mean


# def load_flight_data(npz_path: str, flight_idx: int) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     从数据集中加载特定飞行的状态和控制序列
    
#     Args:
#         npz_path: 数据集文件路径
#         flight_idx: 飞行索引
        
#     Returns:
#         states: 状态序列 (L, 6) [x, y, yaw, u, v, r]
#         controls: 控制序列 (L, 4) [port_throttle, port_angle, starboard_throttle, starboard_angle]
#     """
#     dataset = PelicanHorizontalTransitionDataset(
#         npz_path, 
#         return_flight_index=False,
#         use_normalized=False
#     )
    
#     # 获取所有飞行数据
#     flight_data = dataset._flights
    
#     if flight_idx < 0 or flight_idx >= len(flight_data):
#         raise ValueError(f"飞行索引 {flight_idx} 超出范围 [0, {len(flight_data)-1}]")
    
#     flight = flight_data[flight_idx]
    
#     # 提取状态序列
#     states_np = dataset._build_state_vector(flight)  # (L, 6)
    
#     # 提取控制序列
#     if "Thrusters_CMD" in flight:
#         controls_np = flight["Thrusters_CMD"].T  # (L, 4)
#     elif "Motors_CMD" in flight:
#         controls_np = flight["Motors_CMD"].T  # (L, 4)
#     else:
#         controls_np = flight["Motors"].T  # (L, 4)
    
#     # 确保序列长度一致
#     L = min(len(states_np), len(controls_np))
#     states_np = states_np[:L]
#     controls_np = controls_np[:L]
    
#     print(f"加载飞行 {flight_idx}:")
#     print(f"  状态序列形状: {states_np.shape}")
#     print(f"  控制序列形状: {controls_np.shape}")
#     print(f"  序列长度: {L}")
    
#     return states_np, controls_np


# def teacher_forced_rollout(model, scaler, states_seq, controls_seq):
#     """
#     教师强制滚转：每一步使用真实状态作为输入
    
#     Args:
#         model: Koopman模型
#         scaler: 标准化器
#         states_seq: 真实状态序列 (L, 6)
#         controls_seq: 控制序列 (L, 4)
        
#     Returns:
#         pred_states: 预测状态序列 (L, 6)
#     """
#     device = next(model.parameters()).device
#     L = len(states_seq)
    
#     # 初始化预测序列
#     pred_states = np.zeros_like(states_seq)
    
#     with torch.no_grad():
#         for t in range(L - 1):
#             # 获取当前状态和控制
#             x_t = torch.tensor(states_seq[t], device=device, dtype=torch.float32)
#             u_t = torch.tensor(controls_seq[t], device=device, dtype=torch.float32)
            
#             # 标准化
#             x_t_norm = scaler.normalize_x(x_t)
#             u_t_norm = scaler.normalize_u(u_t)
            
#             # 前向传播（单步预测）
#             _, _, x_tp1_hat_norm, _ = model(x_t_norm.unsqueeze(0), u_t_norm.unsqueeze(0))
            
#             # 反标准化并分离梯度
#             x_tp1_hat = scaler.denormalize_x(x_tp1_hat_norm.squeeze(0)).detach()
            
#             # 保存预测结果
#             pred_states[t + 1] = x_tp1_hat.cpu().numpy()
    
#     # 第一个状态使用真实值（或使用模型重构）
#     x0 = torch.tensor(states_seq[0], device=device, dtype=torch.float32)
#     x0_norm = scaler.normalize_x(x0)
#     z0 = model.encode(x0_norm.unsqueeze(0))
#     x0_recon_norm = model.reconstruct_state(z0)
#     x0_recon = scaler.denormalize_x(x0_recon_norm.squeeze(0)).detach()
#     pred_states[0] = x0_recon.cpu().numpy()
    
#     return pred_states


# def open_loop_rollout(model, scaler, initial_state, controls_seq):
#     """
#     开环滚转：只使用初始状态和控制序列进行多步预测
    
#     Args:
#         model: Koopman模型
#         scaler: 标准化器
#         initial_state: 初始状态 (6,)
#         controls_seq: 控制序列 (L-1, 4)
        
#     Returns:
#         pred_states: 预测状态序列 (L, 6)
#     """
#     device = next(model.parameters()).device
    
#     # 将数据转换为tensor
#     x0 = torch.tensor(initial_state, device=device, dtype=torch.float32)
#     u_seq = torch.tensor(controls_seq, device=device, dtype=torch.float32)
    
#     # 标准化
#     x0_norm = scaler.normalize_x(x0)
#     u_seq_norm = scaler.normalize_u(u_seq)
    
#     # 使用模型的predict_sequence方法
#     with torch.no_grad():
#         # 添加批次维度
#         x0_norm_batch = x0_norm.unsqueeze(0)  # (1, 6)
#         u_seq_norm_batch = u_seq_norm.unsqueeze(1)  # (L-1, 1, 4)
        
#         # 预测序列
#         pred_states_norm = model.predict_sequence(x0_norm_batch, u_seq_norm_batch)  # (L, 1, 6)
#         pred_states_norm = pred_states_norm.squeeze(1)  # (L, 6)
        
#         # 反标准化并分离梯度
#         pred_states = scaler.denormalize_x(pred_states_norm).detach()
    
#     return pred_states.cpu().numpy()


# def compute_prediction_errors(gt_states, pred_states, state_names=None):
#     """
#     计算预测误差统计
    
#     Args:
#         gt_states: 真实状态序列 (L, 6)
#         pred_states: 预测状态序列 (L, 6)
#         state_names: 状态分量名称
        
#     Returns:
#         Dict containing error statistics
#     """
#     if state_names is None:
#         state_names = ['x', 'y', 'yaw', 'u', 'v', 'r']
    
#     errors = gt_states - pred_states
    
#     # 绝对误差
#     mae = np.mean(np.abs(errors), axis=0)
#     mse = np.mean(errors**2, axis=0)
#     rmse = np.sqrt(mse)
    
#     # 相对误差（百分比）
#     gt_range = np.max(gt_states, axis=0) - np.min(gt_states, axis=0)
#     gt_range[gt_range < 1e-6] = 1.0  # 避免除零
#     relative_mae = mae / gt_range * 100
    
#     # 相关系数
#     corr_coeffs = np.array([
#         np.corrcoef(gt_states[:, i], pred_states[:, i])[0, 1]
#         for i in range(gt_states.shape[1])
#     ])
    
#     # 总体统计
#     overall_mae = np.mean(mae)
#     overall_rmse = np.sqrt(np.mean(mse))
#     overall_corr = np.mean(corr_coeffs)
    
#     stats = {
#         'state_names': state_names,
#         'mae': mae,
#         'mse': mse,
#         'rmse': rmse,
#         'relative_mae': relative_mae,
#         'correlation': corr_coeffs,
#         'overall_mae': overall_mae,
#         'overall_rmse': overall_rmse,
#         'overall_correlation': overall_corr
#     }
    
#     return stats


# def print_error_statistics(stats):
#     """打印误差统计信息"""
#     print("\n" + "="*60)
#     print("预测误差统计")
#     print("="*60)
    
#     print(f"{'状态':<10} {'MAE':<12} {'RMSE':<12} {'相对MAE(%)':<15} {'相关系数':<10}")
#     print("-"*60)
    
#     for i, name in enumerate(stats['state_names']):
#         print(f"{name:<10} {stats['mae'][i]:<12.6f} {stats['rmse'][i]:<12.6f} "
#               f"{stats['relative_mae'][i]:<15.2f} {stats['correlation'][i]:<10.4f}")
    
#     print("-"*60)
#     print(f"{'总体':<10} {stats['overall_mae']:<12.6f} {stats['overall_rmse']:<12.6f} "
#           f"{'N/A':<15} {stats['overall_correlation']:<10.4f}")
#     print("="*60)


# def plot_rollout_comparison(gt_states, pred_states, flight_idx, mode, 
#                            state_names=None, save_path=None):
#     """
#     绘制真实状态与预测状态的对比图
    
#     Args:
#         gt_states: 真实状态序列 (L, 6)
#         pred_states: 预测状态序列 (L, 6)
#         flight_idx: 飞行索引
#         mode: 滚转模式 ('teacher-forced' 或 'open-loop')
#         state_names: 状态分量名称
#         save_path: 保存图片的路径
#     """
#     if state_names is None:
#         state_names = ['x', 'y', 'yaw', 'u', 'v', 'r']
    
#     L = len(gt_states)
#     time = np.arange(L)
    
#     # 创建子图
#     fig, axes = plt.subplots(3, 2, figsize=(15, 12))
#     axes = axes.flatten()
    
#     for i, (ax, name) in enumerate(zip(axes, state_names)):
#         # 绘制真实值和预测值
#         ax.plot(time, gt_states[:, i], 'b-', linewidth=1.5, label='真实值', alpha=0.7)
#         ax.plot(time, pred_states[:, i], 'r--', linewidth=1.5, label='预测值', alpha=0.7)
        
#         # 计算并显示误差
#         error = gt_states[:, i] - pred_states[:, i]
#         rmse = np.sqrt(np.mean(error**2))
#         corr = np.corrcoef(gt_states[:, i], pred_states[:, i])[0, 1]
        
#         # 在子图中添加误差信息
#         ax.text(0.02, 0.95, f'RMSE: {rmse:.4f}\nCorr: {corr:.4f}', 
#                 transform=ax.transAxes, verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
#         # 设置标题和标签
#         ax.set_title(f'{name} (状态[{i}])')
#         ax.set_xlabel('时间步')
#         ax.set_ylabel(name)
#         ax.grid(True, alpha=0.3)
#         ax.legend(loc='upper right')
    
#     # 添加总标题
#     mode_str = '教师强制' if mode == 'teacher-forced' else '开环'
#     fig.suptitle(f'飞行 {flight_idx} - {mode_str}滚转预测对比 (序列长度: {L})', 
#                 fontsize=16, fontweight='bold')
    
#     plt.tight_layout()
    
#     if save_path:
#         os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"对比图已保存至: {save_path}")
    
#     plt.show()
    
    
# def plot_prediction_errors(gt_states, pred_states, state_names=None, save_path=None):
#     """
#     绘制预测误差的时间序列
    
#     Args:
#         gt_states: 真实状态序列 (L, 6)
#         pred_states: 预测状态序列 (L, 6)
#         state_names: 状态分量名称
#         save_path: 保存图片的路径
#     """
#     if state_names is None:
#         state_names = ['x', 'y', 'yaw', 'u', 'v', 'r']
    
#     L = len(gt_states)
#     time = np.arange(L)
#     errors = gt_states - pred_states
    
#     # 创建子图
#     fig, axes = plt.subplots(3, 2, figsize=(15, 12))
#     axes = axes.flatten()
    
#     for i, (ax, name) in enumerate(zip(axes, state_names)):
#         # 绘制误差
#         ax.plot(time, errors[:, i], 'g-', linewidth=1.5, alpha=0.7)
#         ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
        
#         # 计算误差统计
#         mae = np.mean(np.abs(errors[:, i]))
#         std = np.std(errors[:, i])
        
#         # 在子图中添加统计信息
#         ax.text(0.02, 0.95, f'MAE: {mae:.4f}\nStd: {std:.4f}', 
#                 transform=ax.transAxes, verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
#         # 设置标题和标签
#         ax.set_title(f'{name} 预测误差')
#         ax.set_xlabel('时间步')
#         ax.set_ylabel('误差')
#         ax.grid(True, alpha=0.3)
    
#     fig.suptitle('预测误差时间序列', fontsize=16, fontweight='bold')
#     plt.tight_layout()
    
#     if save_path:
#         error_plot_path = save_path.replace('.png', '_errors.png')
#         plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
#         print(f"误差图已保存至: {error_plot_path}")
    
#     plt.show()


# def load_model_and_scaler(checkpoint_path, device):
#     """
#     从检查点加载模型和标准化器
    
#     Args:
#         checkpoint_path: 检查点文件路径
#         device: 计算设备
        
#     Returns:
#         model: 加载的Koopman模型
#         scaler: 标准化器
#         config: 模型配置
#     """
#     print(f"正在加载检查点: {checkpoint_path}")
    
#     # 修复：使用 weights_only=True 以避免警告
#     try:
#         checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#     except TypeError:
#         # 如果旧版本不支持 weights_only 参数
#         checkpoint = torch.load(checkpoint_path, map_location=device)
    
#     # 获取配置
#     config = checkpoint.get('config', {})
    
#     # 模型参数 - 确保所有参数都正确处理
#     model_type = config.get('model_type', 'horizontal')
#     state_dim = config.get('state_dim', 6)
#     control_dim = config.get('control_dim', 4)
#     latent_dim = config.get('latent_dim', 16)
    
#     # 修复：将元组转换为列表
#     enc_hidden = config.get('enc_hidden', [64, 64])
#     if isinstance(enc_hidden, tuple):
#         enc_hidden = list(enc_hidden)
    
#     dec_hidden = config.get('dec_hidden', [64, 64])
#     if isinstance(dec_hidden, tuple):
#         dec_hidden = list(dec_hidden)
    
#     use_skip = config.get('use_skip', True)
    
#     # 创建模型
#     print(f"创建 {model_type} 模型...")
#     print(f"配置参数: state_dim={state_dim}, control_dim={control_dim}, latent_dim={latent_dim}")
#     print(f"编码器隐藏层: {enc_hidden}")
#     print(f"解码器隐藏层: {dec_hidden}")
#     print(f"使用跳连接: {use_skip}")
    
#     if model_type == 'horizontal':
#         model = HorizontalKoopmanModel(
#             state_dim=state_dim,
#             control_dim=control_dim,
#             latent_dim=latent_dim,
#             enc_hidden=enc_hidden,
#             dec_hidden=dec_hidden,
#             use_skip=use_skip
#         )
#     elif model_type == 'deep':
#         model = DeepKoopmanModel(
#             state_dim=state_dim,
#             control_dim=control_dim,
#             latent_dim=latent_dim,
#             enc_hidden=enc_hidden,
#             dec_hidden=dec_hidden
#         )
#     else:
#         raise ValueError(f"未知的模型类型: {model_type}")
    
#     # 加载模型权重
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
#     model.eval()
    
#     # 统计模型参数
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"模型加载完成:")
#     print(f"  总参数量: {total_params:,}")
#     print(f"  可训练参数量: {trainable_params:,}")
    
#     # 加载标准化器 - 处理可能的各种格式
#     scaler_dict = checkpoint.get('scaler', {})
#     if scaler_dict and scaler_dict.get('x_mean') is not None:
#         try:
#             x_mean = np.array(scaler_dict.get('x_mean', [0.0]))
#             x_std = np.array(scaler_dict.get('x_std', [1.0]))
#             u_mean = np.array(scaler_dict.get('u_mean', [0.0]))
#             u_std = np.array(scaler_dict.get('u_std', [1.0]))
            
#             # 确保形状正确
#             if x_mean.ndim == 2:
#                 x_mean = x_mean.squeeze(0)
#             if x_std.ndim == 2:
#                 x_std = x_std.squeeze(0)
#             if u_mean.ndim == 2:
#                 u_mean = u_mean.squeeze(0)
#             if u_std.ndim == 2:
#                 u_std = u_std.squeeze(0)
            
#             scaler = StandardScaler(x_mean, x_std, u_mean, u_std, device)
#             print("标准化器加载完成")
            
#             # 打印标准化器信息
#             print(f"状态均值: {x_mean}")
#             print(f"状态标准差: {x_std}")
#             print(f"控制均值: {u_mean}")
#             print(f"控制标准差: {u_std}")
            
#         except Exception as e:
#             print(f"加载标准化器失败: {e}")
#             print("使用单位标准化器")
#             x_mean = np.zeros(state_dim)
#             x_std = np.ones(state_dim)
#             u_mean = np.zeros(control_dim)
#             u_std = np.ones(control_dim)
#             scaler = StandardScaler(x_mean, x_std, u_mean, u_std, device)
#     else:
#         print("警告: 检查点中没有标准化器参数，使用单位标准化器")
#         # 创建单位标准化器（不进行标准化）
#         x_mean = np.zeros(state_dim)
#         x_std = np.ones(state_dim)
#         u_mean = np.zeros(control_dim)
#         u_std = np.ones(control_dim)
#         scaler = StandardScaler(x_mean, x_std, u_mean, u_std, device)
    
#     # 打印检查点中的其他信息
#     print(f"训练轮数: {checkpoint.get('epoch', '未知')}")
#     print(f"全局步数: {checkpoint.get('global_step', '未知')}")
#     if 'val_loss' in checkpoint:
#         print(f"验证损失: {checkpoint['val_loss']:.6f}")
#     if 'test_loss' in checkpoint:
#         print(f"测试损失: {checkpoint['test_loss']:.6f}")
    
#     return model, scaler, config


# def main():
#     parser = argparse.ArgumentParser(description='使用训练好的Koopman模型进行滚转预测')
    
#     # 必需参数
#     parser.add_argument('--ckpt', type=str, required=True,
#                        help='训练好的模型检查点路径 (.pt 文件)')
#     parser.add_argument('--flight', type=int, required=True,
#                        help='要测试的飞行索引 [0..N-1]')
    
#     # 可选参数
#     parser.add_argument('--npz-path', type=str, default='pelican_dataset_horizontal.npz',
#                        help='水平面数据集路径 (默认: pelican_dataset_horizontal.npz)')
#     parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
#                        help='计算设备: cpu 或 cuda (默认: 自动选择)')
#     parser.add_argument('--mode', type=str, choices=['teacher-forced', 'open-loop'], default='teacher-forced',
#                        help='滚转模式: teacher-forced(教师强制) 或 open-loop(开环) (默认: teacher-forced)')
#     parser.add_argument('--limit-steps', type=int, default=0,
#                        help='限制滚转步数 (如果>0，只使用前N步)')
#     parser.add_argument('--save-plot', type=str, default='',
#                        help='保存对比图的路径 (如果不提供，则显示而不保存)')
#     parser.add_argument('--show-errors', action='store_true',
#                        help='显示预测误差图')
#     parser.add_argument('--print-details', action='store_true',
#                        help='打印详细预测信息')
    
#     args = parser.parse_args()
    
#     # 设置设备
#     device = torch.device(args.device)
#     print(f"使用设备: {device}")
    
#     # 加载模型和标准化器
#     model, scaler, config = load_model_and_scaler(args.ckpt, device)
    
#     # 加载飞行数据
#     print(f"\n加载飞行 {args.flight} 的数据...")
#     try:
#         states_seq, controls_seq = load_flight_data(args.npz_path, args.flight)
#     except Exception as e:
#         print(f"加载飞行数据失败: {e}")
#         print("尝试直接加载数据集文件...")
#         # 备选方案：直接加载整个数据集
#         dataset = PelicanHorizontalTransitionDataset(args.npz_path, return_flight_index=False, use_normalized=False)
#         flight_data = dataset._flights
#         if args.flight >= len(flight_data):
#             raise ValueError(f"飞行索引 {args.flight} 超出范围 [0, {len(flight_data)-1}]")
#         flight = flight_data[args.flight]
#         states_seq = dataset._build_state_vector(flight)
#         controls_seq = flight["Thrusters_CMD"].T if "Thrusters_CMD" in flight else flight["Motors"].T
        
#         # 确保长度一致
#         L = min(len(states_seq), len(controls_seq))
#         states_seq = states_seq[:L]
#         controls_seq = controls_seq[:L]
    
#     # 限制步数（如果指定）
#     if args.limit_steps > 0:
#         L_limit = min(args.limit_steps, len(states_seq))
#         states_seq = states_seq[:L_limit]
#         controls_seq = controls_seq[:L_limit]
#         print(f"限制序列长度为: {L_limit}")
    
#     print(f"状态序列形状: {states_seq.shape}")
#     print(f"控制序列形状: {controls_seq.shape}")
    
#     # 执行滚转预测
#     print(f"\n执行 {args.mode} 滚转预测...")
    
#     if args.mode == 'teacher-forced':
#         # 教师强制模式：每一步使用真实状态作为输入
#         pred_states = teacher_forced_rollout(model, scaler, states_seq, controls_seq)
#     else:
#         # 开环模式：只使用初始状态和控制序列
#         initial_state = states_seq[0]
#         pred_states = open_loop_rollout(model, scaler, initial_state, controls_seq)
    
#     print("滚转预测完成")
    
#     # 计算并打印误差统计
#     state_names = ['x', 'y', 'yaw', 'surge', 'sway', 'yaw_rate']
#     stats = compute_prediction_errors(states_seq, pred_states, state_names)
#     print_error_statistics(stats)
    
#     # 计算谱半径（稳定性指标）
#     with torch.no_grad():
#         spectral_radius = model.spectral_radius().item()
#     print(f"模型谱半径 (稳定性指标): {spectral_radius:.4f}")
#     if spectral_radius > 1.0:
#         print("警告: 谱半径 > 1.0，模型可能不稳定")
    
#     # 绘制对比图
#     save_path = args.save_plot
#     if not save_path and args.mode == 'open-loop':
#         # 为开环预测自动生成文件名
#         save_path = f"rollout_openloop_flight{args.flight}_steps{len(states_seq)}.png"
    
#     plot_rollout_comparison(states_seq, pred_states, args.flight, args.mode, 
#                            state_names, save_path)
    
#     # 如果需要，绘制误差图
#     if args.show_errors:
#         plot_prediction_errors(states_seq, pred_states, state_names, save_path)
    
#     # 打印详细预测信息（如果需要）
#     if args.print_details:
#         print("\n详细预测信息:")
#         print("-"*60)
#         for t in range(0, len(states_seq), max(1, len(states_seq)//10)):
#             print(f"时间步 {t}:")
#             print(f"  真实状态: {states_seq[t]}")
#             print(f"  预测状态: {pred_states[t]}")
#             print(f"  绝对误差: {np.abs(states_seq[t] - pred_states[t])}")
#             if t < len(controls_seq):
#                 print(f"  控制输入: {controls_seq[t]}")
#             print()


# if __name__ == '__main__':
#     main()








"""
Roll out a trained Koopman model on a specific flight data and compare with ground truth.

This script loads a checkpoint saved by train_koopman.py, reconstructs the KoopmanModel,
normalizes inputs using the saved scaler, and performs latent-space rollout:
z_{t+1} = A z_t + B u_t using the flight's control inputs.
The decoded states are inverse-transformed back to the original scale and
plotted against the flight's ground-truth states for each component.

Usage examples:
- python rollout_model.py --ckpt checkpoints_horizontal/best_model.pt --flight 0
- python rollout_model.py --ckpt path.pt --flight 5 --limit-steps 200 --save plots/rollout_f5.png
- Default mode is teacher-forced (single-step). Use --open-loop for free-running.

Note:
- This script is designed for horizontal plane ship motion (6D state, 4D control)
- Supports two rollout modes: teacher-forced (one-step) and open-loop (multi-step)
"""

import argparse
import os
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from pelican_torch_dataset import PelicanHorizontalTransitionDataset
from koopman import HorizontalKoopmanModel, DeepKoopmanModel


class StandardScaler:
    """Standard scaler (zero mean, unit variance)"""
    
    def __init__(self, x_mean, x_std, u_mean, u_std, device: torch.device):
        """
        Initialize scaler
        
        Args:
            x_mean: State mean (state_dim,)
            x_std: State standard deviation (state_dim,)
            u_mean: Control mean (control_dim,)
            u_std: Control standard deviation (control_dim,)
            device: Computing device
        """
        self.x_mean = torch.tensor(x_mean, dtype=torch.float32, device=device)
        self.x_std = torch.tensor(x_std, dtype=torch.float32, device=device)
        self.u_mean = torch.tensor(u_mean, dtype=torch.float32, device=device)
        self.u_std = torch.tensor(u_std, dtype=torch.float32, device=device)
        
        # Ensure correct shape
        if self.x_mean.dim() == 2:
            self.x_mean = self.x_mean.squeeze(0)
        if self.x_std.dim() == 2:
            self.x_std = self.x_std.squeeze(0)
        if self.u_mean.dim() == 2:
            self.u_mean = self.u_mean.squeeze(0)
        if self.u_std.dim() == 2:
            self.u_std = self.u_std.squeeze(0)
        
    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize state: x_norm = (x - mean) / std"""
        return (x - self.x_mean) / self.x_std
    
    def normalize_u(self, u: torch.Tensor) -> torch.Tensor:
        """Normalize control: u_norm = (u - mean) / std"""
        return (u - self.u_mean) / self.u_std
    
    def denormalize_x(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize state: x = x_norm * std + mean"""
        return x_norm * self.x_std + self.x_mean
    
    def denormalize_u(self, u_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize control: u = u_norm * std + mean"""
        return u_norm * self.u_std + self.u_mean


def load_flight_data(npz_path: str, flight_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load state and control sequences for a specific flight from dataset
    
    Args:
        npz_path: Dataset file path
        flight_idx: Flight index
        
    Returns:
        states: State sequence (L, 6) [x, y, yaw, u, v, r]
        controls: Control sequence (L, 4) [port_throttle, port_angle, starboard_throttle, starboard_angle]
    """
    dataset = PelicanHorizontalTransitionDataset(
        npz_path, 
        return_flight_index=False,
        use_normalized=False
    )
    
    # Get all flight data
    flight_data = dataset._flights
    
    if flight_idx < 0 or flight_idx >= len(flight_data):
        raise ValueError(f"Flight index {flight_idx} out of range [0, {len(flight_data)-1}]")
    
    flight = flight_data[flight_idx]
    
    # Extract state sequence
    states_np = dataset._build_state_vector(flight)  # (L, 6)
    
    # Extract control sequence
    if "Thrusters_CMD" in flight:
        controls_np = flight["Thrusters_CMD"].T  # (L, 4)
    elif "Motors_CMD" in flight:
        controls_np = flight["Motors_CMD"].T  # (L, 4)
    else:
        controls_np = flight["Motors"].T  # (L, 4)
    
    # Ensure consistent sequence length
    L = min(len(states_np), len(controls_np))
    states_np = states_np[:L]
    controls_np = controls_np[:L]
    
    print(f"Loading flight {flight_idx}:")
    print(f"  State sequence shape: {states_np.shape}")
    print(f"  Control sequence shape: {controls_np.shape}")
    print(f"  Sequence length: {L}")
    
    return states_np, controls_np


def teacher_forced_rollout(model, scaler, states_seq, controls_seq):
    """
    Teacher-forced rollout: Use true state as input at each step
    
    Args:
        model: Koopman model
        scaler: Standard scaler
        states_seq: True state sequence (L, 6)
        controls_seq: Control sequence (L, 4)
        
    Returns:
        pred_states: Predicted state sequence (L, 6)
    """
    device = next(model.parameters()).device
    L = len(states_seq)
    
    # Initialize prediction sequence
    pred_states = np.zeros_like(states_seq)
    
    with torch.no_grad():
        for t in range(L - 1):
            # Get current state and control
            x_t = torch.tensor(states_seq[t], device=device, dtype=torch.float32)
            u_t = torch.tensor(controls_seq[t], device=device, dtype=torch.float32)
            
            # Normalize
            x_t_norm = scaler.normalize_x(x_t)
            u_t_norm = scaler.normalize_u(u_t)
            
            # Forward pass (single-step prediction)
            _, _, x_tp1_hat_norm, _ = model(x_t_norm.unsqueeze(0), u_t_norm.unsqueeze(0))
            
            # Denormalize and detach gradients
            x_tp1_hat = scaler.denormalize_x(x_tp1_hat_norm.squeeze(0)).detach()
            
            # Save prediction result
            pred_states[t + 1] = x_tp1_hat.cpu().numpy()
    
    # First state uses true value (or model reconstruction)
    x0 = torch.tensor(states_seq[0], device=device, dtype=torch.float32)
    x0_norm = scaler.normalize_x(x0)
    z0 = model.encode(x0_norm.unsqueeze(0))
    x0_recon_norm = model.reconstruct_state(z0)
    x0_recon = scaler.denormalize_x(x0_recon_norm.squeeze(0)).detach()
    pred_states[0] = x0_recon.cpu().numpy()
    
    return pred_states


def open_loop_rollout(model, scaler, initial_state, controls_seq):
    """
    Open-loop rollout: Use only initial state and control sequence for multi-step prediction
    
    Args:
        model: Koopman model
        scaler: Standard scaler
        initial_state: Initial state (6,)
        controls_seq: Control sequence (L-1, 4)
        
    Returns:
        pred_states: Predicted state sequence (L, 6)
    """
    device = next(model.parameters()).device
    
    # Convert data to tensor
    x0 = torch.tensor(initial_state, device=device, dtype=torch.float32)
    u_seq = torch.tensor(controls_seq, device=device, dtype=torch.float32)
    
    # Normalize
    x0_norm = scaler.normalize_x(x0)
    u_seq_norm = scaler.normalize_u(u_seq)
    
    # Use model's predict_sequence method
    with torch.no_grad():
        # Add batch dimension
        x0_norm_batch = x0_norm.unsqueeze(0)  # (1, 6)
        u_seq_norm_batch = u_seq_norm.unsqueeze(1)  # (L-1, 1, 4)
        
        # Predict sequence
        pred_states_norm = model.predict_sequence(x0_norm_batch, u_seq_norm_batch)  # (L, 1, 6)
        pred_states_norm = pred_states_norm.squeeze(1)  # (L, 6)
        
        # Denormalize and detach gradients
        pred_states = scaler.denormalize_x(pred_states_norm).detach()
    
    return pred_states.cpu().numpy()


def compute_prediction_errors(gt_states, pred_states, state_names=None):
    """
    Compute prediction error statistics
    
    Args:
        gt_states: Ground truth state sequence (L, 6)
        pred_states: Predicted state sequence (L, 6)
        state_names: State component names
        
    Returns:
        Dict containing error statistics
    """
    if state_names is None:
        state_names = ['x', 'y', 'yaw', 'u', 'v', 'r']
    
    errors = gt_states - pred_states
    
    # Absolute errors
    mae = np.mean(np.abs(errors), axis=0)
    mse = np.mean(errors**2, axis=0)
    rmse = np.sqrt(mse)
    
    # Relative errors (percentage)
    gt_range = np.max(gt_states, axis=0) - np.min(gt_states, axis=0)
    gt_range[gt_range < 1e-6] = 1.0  # Avoid division by zero
    relative_mae = mae / gt_range * 100
    
    # Correlation coefficients
    corr_coeffs = np.array([
        np.corrcoef(gt_states[:, i], pred_states[:, i])[0, 1]
        for i in range(gt_states.shape[1])
    ])
    
    # Overall statistics
    overall_mae = np.mean(mae)
    overall_rmse = np.sqrt(np.mean(mse))
    overall_corr = np.mean(corr_coeffs)
    
    stats = {
        'state_names': state_names,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'relative_mae': relative_mae,
        'correlation': corr_coeffs,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'overall_correlation': overall_corr
    }
    
    return stats


def print_error_statistics(stats):
    """Print error statistics"""
    print("\n" + "="*70)
    print("Prediction Error Statistics")
    print("="*70)
    
    print(f"{'State':<10} {'MAE':<12} {'RMSE':<12} {'Rel.MAE(%)':<15} {'Correlation':<10}")
    print("-"*70)
    
    for i, name in enumerate(stats['state_names']):
        print(f"{name:<10} {stats['mae'][i]:<12.6f} {stats['rmse'][i]:<12.6f} "
              f"{stats['relative_mae'][i]:<15.2f} {stats['correlation'][i]:<10.4f}")
    
    print("-"*70)
    print(f"{'Overall':<10} {stats['overall_mae']:<12.6f} {stats['overall_rmse']:<12.6f} "
          f"{'N/A':<15} {stats['overall_correlation']:<10.4f}")
    print("="*70)


def plot_rollout_comparison(gt_states, pred_states, flight_idx, mode, 
                           state_names=None, save_path=None):
    """
    Plot comparison between ground truth and predicted states
    
    Args:
        gt_states: Ground truth state sequence (L, 6)
        pred_states: Predicted state sequence (L, 6)
        flight_idx: Flight index
        mode: Rollout mode ('teacher-forced' or 'open-loop')
        state_names: State component names
        save_path: Path to save the plot
    """
    if state_names is None:
        state_names = ['x [m]', 'y [m]', 'yaw [rad]', 'surge [m/s]', 'sway [m/s]', 'yaw_rate [rad/s]']
    
    L = len(gt_states)
    time = np.arange(L)
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        # Plot ground truth and prediction
        ax.plot(time, gt_states[:, i], 'b-', linewidth=1.5, label='Ground Truth', alpha=0.7)
        ax.plot(time, pred_states[:, i], 'r--', linewidth=1.5, label='Prediction', alpha=0.7)
        
        # Calculate and display error
        error = gt_states[:, i] - pred_states[:, i]
        rmse = np.sqrt(np.mean(error**2))
        corr = np.corrcoef(gt_states[:, i], pred_states[:, i])[0, 1]
        
        # Add error information to subplot
        ax.text(0.02, 0.95, f'RMSE: {rmse:.4f}\nCorr: {corr:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Set title and labels
        ax.set_title(f'{name}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(name.split(' [')[0])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    # Add main title
    mode_str = 'Teacher-Forced' if mode == 'teacher-forced' else 'Open-Loop'
    fig.suptitle(f'Flight {flight_idx} - {mode_str} Rollout Comparison (Sequence Length: {L})', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()
    
    
def plot_prediction_errors(gt_states, pred_states, state_names=None, save_path=None):
    """
    Plot prediction errors over time
    
    Args:
        gt_states: Ground truth state sequence (L, 6)
        pred_states: Predicted state sequence (L, 6)
        state_names: State component names
        save_path: Path to save the plot
    """
    if state_names is None:
        state_names = ['x [m]', 'y [m]', 'yaw [rad]', 'surge [m/s]', 'sway [m/s]', 'yaw_rate [rad/s]']
    
    L = len(gt_states)
    time = np.arange(L)
    errors = gt_states - pred_states
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        # Plot error
        ax.plot(time, errors[:, i], 'g-', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
        
        # Calculate error statistics
        mae = np.mean(np.abs(errors[:, i]))
        std = np.std(errors[:, i])
        
        # Add statistics to subplot
        ax.text(0.02, 0.95, f'MAE: {mae:.4f}\nStd: {std:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Set title and labels
        ax.set_title(f'{name} Prediction Error')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Error')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Prediction Error Time Series', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        error_plot_path = save_path.replace('.png', '_errors.png')
        plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
        print(f"Error plot saved to: {error_plot_path}")
    
    plt.show()


def load_model_and_scaler(checkpoint_path, device):
    """
    Load model and scaler from checkpoint
    
    Args:
        checkpoint_path: Checkpoint file path
        device: Computing device
        
    Returns:
        model: Loaded Koopman model
        scaler: Standard scaler
        config: Model configuration
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Fix: Use weights_only=True to avoid warning
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # If older version doesn't support weights_only parameter
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get configuration
    config = checkpoint.get('config', {})
    
    # Model parameters - ensure all parameters are correctly processed
    model_type = config.get('model_type', 'horizontal')
    state_dim = config.get('state_dim', 6)
    control_dim = config.get('control_dim', 4)
    latent_dim = config.get('latent_dim', 16)
    
    # Fix: Convert tuple to list
    enc_hidden = config.get('enc_hidden', [64, 64])
    if isinstance(enc_hidden, tuple):
        enc_hidden = list(enc_hidden)
    
    dec_hidden = config.get('dec_hidden', [64, 64])
    if isinstance(dec_hidden, tuple):
        dec_hidden = list(dec_hidden)
    
    use_skip = config.get('use_skip', True)
    
    # Create model
    print(f"Creating {model_type} model...")
    print(f"Configuration: state_dim={state_dim}, control_dim={control_dim}, latent_dim={latent_dim}")
    print(f"Encoder hidden layers: {enc_hidden}")
    print(f"Decoder hidden layers: {dec_hidden}")
    print(f"Use skip connection: {use_skip}")
    
    if model_type == 'horizontal':
        model = HorizontalKoopmanModel(
            state_dim=state_dim,
            control_dim=control_dim,
            latent_dim=latent_dim,
            enc_hidden=enc_hidden,
            dec_hidden=dec_hidden,
            use_skip=use_skip
        )
    elif model_type == 'deep':
        model = DeepKoopmanModel(
            state_dim=state_dim,
            control_dim=control_dim,
            latent_dim=latent_dim,
            enc_hidden=enc_hidden,
            dec_hidden=dec_hidden
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Load scaler - handle various possible formats
    scaler_dict = checkpoint.get('scaler', {})
    if scaler_dict and scaler_dict.get('x_mean') is not None:
        try:
            x_mean = np.array(scaler_dict.get('x_mean', [0.0]))
            x_std = np.array(scaler_dict.get('x_std', [1.0]))
            u_mean = np.array(scaler_dict.get('u_mean', [0.0]))
            u_std = np.array(scaler_dict.get('u_std', [1.0]))
            
            # Ensure correct shape
            if x_mean.ndim == 2:
                x_mean = x_mean.squeeze(0)
            if x_std.ndim == 2:
                x_std = x_std.squeeze(0)
            if u_mean.ndim == 2:
                u_mean = u_mean.squeeze(0)
            if u_std.ndim == 2:
                u_std = u_std.squeeze(0)
            
            scaler = StandardScaler(x_mean, x_std, u_mean, u_std, device)
            print("Scaler loaded successfully")
            
            # Print scaler information
            print(f"State mean: {x_mean}")
            print(f"State std: {x_std}")
            print(f"Control mean: {u_mean}")
            print(f"Control std: {u_std}")
            
        except Exception as e:
            print(f"Failed to load scaler: {e}")
            print("Using identity scaler")
            x_mean = np.zeros(state_dim)
            x_std = np.ones(state_dim)
            u_mean = np.zeros(control_dim)
            u_std = np.ones(control_dim)
            scaler = StandardScaler(x_mean, x_std, u_mean, u_std, device)
    else:
        print("Warning: No scaler parameters found in checkpoint, using identity scaler")
        # Create identity scaler (no normalization)
        x_mean = np.zeros(state_dim)
        x_std = np.ones(state_dim)
        u_mean = np.zeros(control_dim)
        u_std = np.ones(control_dim)
        scaler = StandardScaler(x_mean, x_std, u_mean, u_std, device)
    
    # Print other information from checkpoint
    print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Global step: {checkpoint.get('global_step', 'unknown')}")
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    if 'test_loss' in checkpoint:
        print(f"Test loss: {checkpoint['test_loss']:.6f}")
    
    return model, scaler, config


def main():
    parser = argparse.ArgumentParser(description='Roll out trained Koopman model for prediction')
    
    # Required arguments
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--flight', type=int, required=True,
                       help='Flight index to test [0..N-1]')
    
    # Optional arguments
    parser.add_argument('--npz-path', type=str, default='pelican_dataset_horizontal.npz',
                       help='Horizontal plane dataset path (default: pelican_dataset_horizontal.npz)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Computing device: cpu or cuda (default: auto-select)')
    parser.add_argument('--mode', type=str, choices=['teacher-forced', 'open-loop'], default='teacher-forced',
                       help='Rollout mode: teacher-forced or open-loop (default: teacher-forced)')
    parser.add_argument('--limit-steps', type=int, default=0,
                       help='Limit rollout steps (if >0, use only first N steps)')
    parser.add_argument('--save-plot', type=str, default='',
                       help='Path to save comparison plot (if not provided, show without saving)')
    parser.add_argument('--show-errors', action='store_true',
                       help='Show prediction error plot')
    parser.add_argument('--print-details', action='store_true',
                       help='Print detailed prediction information')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model and scaler
    model, scaler, config = load_model_and_scaler(args.ckpt, device)
    
    # Load flight data
    print(f"\nLoading data for flight {args.flight}...")
    try:
        states_seq, controls_seq = load_flight_data(args.npz_path, args.flight)
    except Exception as e:
        print(f"Failed to load flight data: {e}")
        print("Trying to load dataset file directly...")
        # Alternative: load entire dataset
        dataset = PelicanHorizontalTransitionDataset(args.npz_path, return_flight_index=False, use_normalized=False)
        flight_data = dataset._flights
        if args.flight >= len(flight_data):
            raise ValueError(f"Flight index {args.flight} out of range [0, {len(flight_data)-1}]")
        flight = flight_data[args.flight]
        states_seq = dataset._build_state_vector(flight)
        controls_seq = flight["Thrusters_CMD"].T if "Thrusters_CMD" in flight else flight["Motors"].T
        
        # Ensure consistent length
        L = min(len(states_seq), len(controls_seq))
        states_seq = states_seq[:L]
        controls_seq = controls_seq[:L]
    
    # Limit steps (if specified)
    if args.limit_steps > 0:
        L_limit = min(args.limit_steps, len(states_seq))
        states_seq = states_seq[:L_limit]
        controls_seq = controls_seq[:L_limit]
        print(f"Limited sequence length to: {L_limit}")
    
    print(f"State sequence shape: {states_seq.shape}")
    print(f"Control sequence shape: {controls_seq.shape}")
    
    # Execute rollout prediction
    print(f"\nExecuting {args.mode} rollout...")
    
    if args.mode == 'teacher-forced':
        # Teacher-forced mode: use true state as input at each step
        pred_states = teacher_forced_rollout(model, scaler, states_seq, controls_seq)
    else:
        # Open-loop mode: use only initial state and control sequence
        initial_state = states_seq[0]
        pred_states = open_loop_rollout(model, scaler, initial_state, controls_seq)
    
    print("Rollout prediction completed")
    
    # Calculate and print error statistics
    state_names = ['x', 'y', 'yaw', 'surge', 'sway', 'yaw_rate']
    stats = compute_prediction_errors(states_seq, pred_states, state_names)
    print_error_statistics(stats)
    
    # Calculate spectral radius (stability metric)
    with torch.no_grad():
        spectral_radius = model.spectral_radius().item()
    print(f"Model spectral radius (stability metric): {spectral_radius:.4f}")
    if spectral_radius > 1.0:
        print("Warning: Spectral radius > 1.0, model may be unstable")
    
    # Plot comparison
    save_path = args.save_plot
    if not save_path and args.mode == 'open-loop':
        # Auto-generate filename for open-loop prediction
        save_path = f"rollout_openloop_flight{args.flight}_steps{len(states_seq)}.png"
    
    # Update state names for plotting with units
    plot_state_names = ['x [m]', 'y [m]', 'yaw [rad]', 'surge [m/s]', 'sway [m/s]', 'yaw_rate [rad/s]']
    plot_rollout_comparison(states_seq, pred_states, args.flight, args.mode, 
                           plot_state_names, save_path)
    
    # If needed, plot error plot
    if args.show_errors:
        plot_prediction_errors(states_seq, pred_states, plot_state_names, save_path)
    
    # Print detailed prediction information (if needed)
    if args.print_details:
        print("\nDetailed prediction information:")
        print("-"*60)
        for t in range(0, len(states_seq), max(1, len(states_seq)//10)):
            print(f"Time step {t}:")
            print(f"  Ground truth state: {states_seq[t]}")
            print(f"  Predicted state: {pred_states[t]}")
            print(f"  Absolute error: {np.abs(states_seq[t] - pred_states[t])}")
            if t < len(controls_seq):
                print(f"  Control input: {controls_seq[t]}")
            print()


if __name__ == '__main__':
    main()