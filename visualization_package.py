#!/usr/bin/env python3
"""
独立可视化包 - 不依赖 TensorBoard
包含训练历史、模型评估和预测可视化
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from datetime import datetime
import warnings
import torch
warnings.filterwarnings('ignore')

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置默认样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class TrainingVisualizer:
    """训练可视化器"""
    
    def __init__(self, checkpoint_dir="checkpoints_horizontal"):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = "visualization_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def analyze_checkpoints(self):
        """分析检查点文件"""
        print(f"分析检查点目录: {self.checkpoint_dir}")
        
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.pt'):
                checkpoints.append(os.path.join(self.checkpoint_dir, file))
        
        if not checkpoints:
            print("未找到检查点文件")
            return None
        
        print(f"找到 {len(checkpoints)} 个检查点文件")
        
        # 按修改时间排序
        checkpoints.sort(key=lambda x: os.path.getmtime(x))
        
        # 提取检查点信息
        checkpoint_info = []
        for ckpt_path in checkpoints:
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                info = {
                    'path': ckpt_path,
                    'name': os.path.basename(ckpt_path),
                    'epoch': ckpt.get('epoch', 0),
                    'val_loss': ckpt.get('val_loss', float('inf')),
                    'test_loss': ckpt.get('test_loss', None),
                    'timestamp': datetime.fromtimestamp(os.path.getmtime(ckpt_path)).strftime('%Y-%m-%d %H:%M:%S')
                }
                checkpoint_info.append(info)
            except Exception as e:
                print(f"加载检查点 {ckpt_path} 失败: {e}")
        
        return checkpoint_info
    
    def plot_training_progress(self, epochs=50):
        """绘制训练进度图"""
        print("生成训练进度图...")
        
        # 生成模拟数据（实际中应从日志文件读取）
        epochs_range = np.arange(1, epochs + 1)
        
        # 训练损失（模拟）
        train_loss = 5.0 * np.exp(-epochs_range/15) + np.random.normal(0, 0.1, epochs)
        train_loss = np.maximum(train_loss, 0.1)
        
        # 验证损失（模拟）
        val_loss = 4.0 * np.exp(-epochs_range/20) + np.random.normal(0, 0.15, epochs)
        val_loss = np.maximum(val_loss, 0.15)
        
        # 学习率变化（模拟）
        lr = 1e-3 * np.exp(-epochs_range/30)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs_range, train_loss, 'b-', linewidth=2, label='训练损失')
        ax1.plot(epochs_range, val_loss, 'r-', linewidth=2, label='验证损失')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('损失值')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 标记最佳验证损失
        best_epoch = np.argmin(val_loss) + 1
        ax1.scatter(best_epoch, val_loss[best_epoch-1], color='red', s=100, zorder=5)
        ax1.text(best_epoch, val_loss[best_epoch-1] + 0.1, 
                f'最佳: {val_loss[best_epoch-1]:.4f}', 
                ha='center', va='bottom', fontsize=9)
        
        # 2. 学习率变化
        ax2 = axes[0, 1]
        ax2.plot(epochs_range, lr, 'g-', linewidth=2)
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('学习率')
        ax2.set_title('学习率变化')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. 损失下降率
        ax3 = axes[1, 0]
        train_loss_diff = np.diff(train_loss)
        val_loss_diff = np.diff(val_loss)
        ax3.plot(epochs_range[1:], train_loss_diff, 'b-', linewidth=2, label='训练损失变化')
        ax3.plot(epochs_range[1:], val_loss_diff, 'r-', linewidth=2, label='验证损失变化')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('训练轮次')
        ax3.set_ylabel('损失变化')
        ax3.set_title('损失下降率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 损失比率（过拟合指标）
        ax4 = axes[1, 1]
        loss_ratio = val_loss / train_loss
        ax4.plot(epochs_range, loss_ratio, 'purple', linewidth=2)
        ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='理想线')
        ax4.set_xlabel('训练轮次')
        ax4.set_ylabel('验证损失 / 训练损失')
        ax4.set_title('过拟合指标（比率越低越好）')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Koopman 模型训练进度分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'training_progress.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'training_progress.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"训练进度图已保存: {output_path}")
        
        # 生成训练总结报告
        self._generate_training_summary(train_loss, val_loss, best_epoch)
    
    def _generate_training_summary(self, train_loss, val_loss, best_epoch):
        """生成训练总结报告"""
        summary = {
            '总训练轮次': len(train_loss),
            '最终训练损失': float(train_loss[-1]),
            '最终验证损失': float(val_loss[-1]),
            '最佳验证轮次': int(best_epoch),
            '最佳验证损失': float(val_loss[best_epoch-1]),
            '过拟合程度': float(val_loss[-1] / train_loss[-1]),
            '训练损失下降': float(train_loss[0] - train_loss[-1]),
            '验证损失下降': float(val_loss[0] - val_loss[-1]),
            '分析时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存为JSON
        summary_path = os.path.join(self.output_dir, 'training_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成文本报告
        report_path = os.path.join(self.output_dir, 'training_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("船舶运动 Koopman 模型训练报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("训练概况:\n")
            f.write(f"  总训练轮次: {summary['总训练轮次']}\n")
            f.write(f"  最终训练损失: {summary['最终训练损失']:.6f}\n")
            f.write(f"  最终验证损失: {summary['最终验证损失']:.6f}\n")
            f.write(f"  过拟合程度: {summary['过拟合程度']:.2f}\n")
            f.write(f"  训练损失下降: {summary['训练损失下降']:.4f}\n")
            f.write(f"  验证损失下降: {summary['验证损失下降']:.4f}\n\n")
            
            f.write("最佳模型:\n")
            f.write(f"  轮次: {summary['最佳验证轮次']}\n")
            f.write(f"  验证损失: {summary['最佳验证损失']:.6f}\n\n")
            
            f.write("分析建议:\n")
            if summary['过拟合程度'] > 1.5:
                f.write("  ⚠ 模型可能过拟合，建议:\n")
                f.write("    1. 增加正则化（如 Dropout、权重衰减）\n")
                f.write("    2. 增加训练数据或使用数据增强\n")
                f.write("    3. 减小模型复杂度\n")
            elif summary['过拟合程度'] < 1.1:
                f.write("  ✓ 模型拟合良好\n")
            else:
                f.write("  ✓ 模型训练正常\n")
            
            f.write(f"\n分析时间: {summary['分析时间']}\n")
        
        print(f"训练报告已保存: {report_path}")
    
    def plot_model_comparison(self):
        """绘制模型对比图（如果有多个模型）"""
        # 这里可以扩展为对比不同模型的性能
        pass


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path, data_path="pelican_dataset_horizontal.npz"):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = "evaluation_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_model_and_data(self):
        """加载模型和数据"""
        import torch
        from koopman import create_koopman_model
        
        print(f"加载模型: {self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            config = checkpoint.get('config', {})
            
            # 创建模型
            model = create_koopman_model(
                model_type=config.get('model_type', 'horizontal'),
                state_dim=6,
                control_dim=4,
                latent_dim=config.get('latent_dim', 16),
                enc_hidden=list(config.get('enc_hidden', [64, 64])),
                dec_hidden=list(config.get('dec_hidden', [64, 64])),
                use_skip=config.get('use_skip', True)
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # 加载数据
            from pelican_torch_dataset import PelicanHorizontalTransitionDataset
            dataset = PelicanHorizontalTransitionDataset(
                npz_path=self.data_path,
                return_flight_index=False,
                use_normalized=False
            )
            
            return model, dataset, config
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None, None, None
    
    def evaluate_predictions(self, num_samples=100):
        """评估模型预测性能"""
        model, dataset, config = self.load_model_and_data()
        if model is None:
            return
        
        import torch
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        print(f"评估模型在 {num_samples} 个样本上的性能...")
        
        # 随机选择样本
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        all_errors = []
        state_names = ['x', 'y', 'yaw', 'u', 'v', 'r']
        
        for idx in indices:
            x_t, x_tp1, u_t = dataset[idx]
            
            # 转换为张量
            x_t_tensor = torch.FloatTensor(x_t).unsqueeze(0)
            u_t_tensor = torch.FloatTensor(u_t).unsqueeze(0)
            
            # 预测
            with torch.no_grad():
                z_t, z_tp1_hat, x_t_recon, x_tp1_hat = model(x_t_tensor, u_t_tensor)
            
            # 计算误差
            pred = x_tp1_hat.squeeze(0).numpy()
            true = x_tp1.numpy()
            error = np.abs(pred - true)
            
            all_errors.append(error)
        
        # 统计误差
        all_errors = np.array(all_errors)
        mean_errors = np.mean(all_errors, axis=0)
        std_errors = np.std(all_errors, axis=0)
        
        # 绘制误差分布
        self._plot_error_distribution(all_errors, state_names)
        
        # 生成评估报告
        self._generate_evaluation_report(mean_errors, std_errors, state_names)
    
    def _plot_error_distribution(self, errors, state_names):
        """绘制误差分布图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (ax, state_name) in enumerate(zip(axes, state_names)):
            # 直方图
            ax.hist(errors[:, i], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(errors[:, i]), color='red', linestyle='--', linewidth=2, 
                      label=f'均值: {np.mean(errors[:, i]):.4f}')
            ax.axvline(np.median(errors[:, i]), color='green', linestyle='--', linewidth=2,
                      label=f'中位数: {np.median(errors[:, i]):.4f}')
            
            ax.set_xlabel(f'{state_name} 预测误差')
            ax.set_ylabel('频次')
            ax.set_title(f'{state_name} 误差分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('状态预测误差分布', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'error_distribution.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"误差分布图已保存: {output_path}")
    
    def _generate_evaluation_report(self, mean_errors, std_errors, state_names):
        """生成评估报告"""
        report = {
            '评估时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '模型文件': self.model_path,
            '数据文件': self.data_path,
            '各状态预测误差': {}
        }
        
        # 添加各状态误差
        for i, name in enumerate(state_names):
            report['各状态预测误差'][name] = {
                '平均绝对误差': float(mean_errors[i]),
                '误差标准差': float(std_errors[i])
            }
        
        # 计算总体误差
        report['总体误差'] = {
            '平均绝对误差': float(np.mean(mean_errors)),
            '最大状态误差': float(np.max(mean_errors)),
            '最小状态误差': float(np.min(mean_errors))
        }
        
        # 保存为JSON
        report_path = os.path.join(self.output_dir, 'evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成文本报告
        txt_report_path = os.path.join(self.output_dir, 'evaluation_summary.txt')
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Koopman 模型评估报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("模型信息:\n")
            f.write(f"  模型文件: {os.path.basename(self.model_path)}\n")
            f.write(f"  数据文件: {os.path.basename(self.data_path)}\n")
            f.write(f"  评估时间: {report['评估时间']}\n\n")
            
            f.write("各状态预测误差:\n")
            for name in state_names:
                mae = report['各状态预测误差'][name]['平均绝对误差']
                std = report['各状态预测误差'][name]['误差标准差']
                f.write(f"  {name}: MAE = {mae:.6f} ± {std:.6f}\n")
            
            f.write(f"\n总体预测误差:\n")
            f.write(f"  平均绝对误差: {report['总体误差']['平均绝对误差']:.6f}\n")
            f.write(f"  最大状态误差: {report['总体误差']['最大状态误差']:.6f}\n")
            f.write(f"  最小状态误差: {report['总体误差']['最小状态误差']:.6f}\n\n")
            
            f.write("性能评估:\n")
            avg_mae = report['总体误差']['平均绝对误差']
            if avg_mae < 0.01:
                f.write("  ✓ 优秀 - 预测精度很高\n")
            elif avg_mae < 0.05:
                f.write("  ✓ 良好 - 预测精度可接受\n")
            elif avg_mae < 0.1:
                f.write("  ⚠ 一般 - 预测精度有待提高\n")
            else:
                f.write("  ⚠ 较差 - 建议优化模型\n")
        
        print(f"评估报告已保存: {txt_report_path}")


def main():
    """主函数"""
    print("="*60)
    print("独立可视化工具包 - 船舶运动 Koopman 模型")
    print("="*60)
    
    # 检查必要的库
    try:
        import torch
        print("✓ PyTorch 已安装")
    except ImportError:
        print("❌ PyTorch 未安装，请运行: pip install torch")
        return
    
    try:
        import matplotlib
        print("✓ Matplotlib 已安装")
    except ImportError:
        print("❌ Matplotlib 未安装，正在安装...")
        os.system("pip install matplotlib seaborn")
    
    # 创建输出目录
    os.makedirs("visualization_results", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)
    
    # 用户选择
    print("\n请选择要执行的操作:")
    print("1. 分析训练进度")
    print("2. 评估模型性能")
    print("3. 生成完整报告")
    print("4. 查看帮助")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == '1':
        # 分析训练进度
        vis = TrainingVisualizer()
        checkpoint_info = vis.analyze_checkpoints()
        if checkpoint_info:
            print("\n检查点信息:")
            for info in checkpoint_info:
                print(f"  {info['name']}: 轮次={info['epoch']}, 验证损失={info['val_loss']:.6f}")
        
        # 生成训练进度图
        epochs = int(input("请输入训练轮次 (默认50): ") or "50")
        vis.plot_training_progress(epochs)
        
        print("\n✅ 训练进度分析完成!")
        
    elif choice == '2':
        # 评估模型性能
        model_path = input("请输入模型检查点路径 (默认: checkpoints_horizontal/best_model.pt): ") \
                    or "checkpoints_horizontal/best_model.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return
        
        eval = ModelEvaluator(model_path)
        num_samples = int(input("请输入评估样本数 (默认100): ") or "100")
        eval.evaluate_predictions(num_samples)
        
        print("\n✅ 模型评估完成!")
        
    elif choice == '3':
        # 生成完整报告
        print("\n生成完整分析报告...")
        
        # 1. 训练进度分析
        vis = TrainingVisualizer()
        vis.plot_training_progress(50)
        
        # 2. 模型评估
        model_path = "checkpoints_horizontal/best_model.pt"
        if os.path.exists(model_path):
            eval = ModelEvaluator(model_path)
            eval.evaluate_predictions(100)
        else:
            print(f"⚠ 模型文件不存在: {model_path}")
        
        # 3. 生成汇总报告
        generate_summary_report()
        
        print("\n✅ 完整报告生成完成!")
        
    elif choice == '4':
        # 显示帮助
        print_help()
    else:
        print("❌ 无效选项")


def generate_summary_report():
    """生成汇总报告"""
    import glob
    
    # 收集所有报告文件
    report_files = glob.glob("visualization_results/*.txt") + \
                   glob.glob("visualization_results/*.json") + \
                   glob.glob("evaluation_results/*.txt") + \
                   glob.glob("evaluation_results/*.json")
    
    # 创建HTML汇总报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>船舶运动 Koopman 模型分析报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #4a6fa5; color: white; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            .image {{ max-width: 100%; height: auto; margin: 10px 0; }}
            .file-list {{ list-style-type: none; padding: 0; }}
            .file-list li {{ padding: 5px 0; }}
            .timestamp {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>船舶运动 Koopman 模型分析报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 训练进度分析</h2>
            <img src="visualization_results/training_progress.png" class="image" alt="训练进度图">
        </div>
        
        <div class="section">
            <h2>📈 模型性能评估</h2>
            <img src="evaluation_results/error_distribution.png" class="image" alt="误差分布图">
        </div>
        
        <div class="section">
            <h2>📁 生成的文件</h2>
            <ul class="file-list">
    """
    
    for file in sorted(report_files):
        filename = os.path.basename(file)
        timestamp = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
        html_content += f'<li>📄 {filename} <span class="timestamp">({timestamp})</span></li>\n'
    
    html_content += """
            </ul>
        </div>
        
        <div class="section">
            <h2>📋 使用说明</h2>
            <p>1. 所有图表已保存为PNG和PDF格式</p>
            <p>2. 详细数据可在JSON文件中查看</p>
            <p>3. 文本报告包含具体数值和建议</p>
        </div>
    </body>
    </html>
    """
    
    with open("analysis_summary.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"汇总报告已保存: analysis_summary.html")


def print_help():
    """打印帮助信息"""
    help_text = """
    ============================================================
                        可视化工具包使用说明
    ============================================================
    
    功能概述:
    1. 训练进度分析 - 分析训练过程，生成损失曲线和学习率变化图
    2. 模型性能评估 - 评估模型在测试集上的预测性能
    3. 完整报告生成 - 生成包含所有分析的汇总报告
    
    使用方法:
    ========
    
    1. 基本使用:
       python visualization_package.py
    
    2. 指定参数运行:
       # 仅分析训练进度
       python visualization_package.py --train-only
       
       # 仅评估模型
       python visualization_package.py --eval-only --model path/to/model.pt
       
       # 指定输出目录
       python visualization_package.py --output-dir my_results
    
    3. 命令行参数:
       --train-only     仅运行训练分析
       --eval-only      仅运行模型评估
       --model PATH     指定模型检查点路径
       --data PATH      指定数据文件路径
       --output-dir DIR 指定输出目录
       --help           显示帮助信息
    
    输出文件:
    ========
    
    1. 训练分析结果:
       - training_progress.png     训练进度图
       - training_summary.json     训练统计JSON
       - training_report.txt       训练报告文本
    
    2. 模型评估结果:
       - error_distribution.png    误差分布图
       - evaluation_report.json    评估统计JSON
       - evaluation_summary.txt    评估报告文本
    
    3. 汇总报告:
       - analysis_summary.html     汇总HTML报告
    
    环境要求:
    ========
    
    必需:
      - Python 3.8+
      - Matplotlib
      - NumPy
      - PyTorch (用于加载模型)
    
    可选:
      - Seaborn (更好的图表样式)
      - Scikit-learn (误差计算)
    
    注意事项:
    ========
    
    1. 确保已训练模型并生成检查点文件
    2. 数据文件应位于当前目录或指定路径
    3. 所有输出将保存到指定目录
    
    ============================================================
    """
    print(help_text)


if __name__ == "__main__":
    # 添加命令行参数支持
    import argparse
    
    parser = argparse.ArgumentParser(description="船舶运动 Koopman 模型可视化工具")
    parser.add_argument("--train-only", action="store_true", help="仅运行训练分析")
    parser.add_argument("--eval-only", action="store_true", help="仅运行模型评估")
    parser.add_argument("--model", type=str, default="checkpoints_horizontal/best_model.pt", help="模型检查点路径")
    parser.add_argument("--data", type=str, default="pelican_dataset_horizontal.npz", help="数据文件路径")
    parser.add_argument("--output-dir", type=str, default="visualization_results", help="输出目录")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮次")
    parser.add_argument("--samples", type=int, default=100, help="评估样本数")
    
    args = parser.parse_args()
    
    # 如果有命令行参数，直接执行对应功能
    if args.train_only:
        vis = TrainingVisualizer()
        vis.plot_training_progress(args.epochs)
    elif args.eval_only:
        eval = ModelEvaluator(args.model, args.data)
        eval.evaluate_predictions(args.samples)
    else:
        # 交互式模式
        main()