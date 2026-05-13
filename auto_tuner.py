import os
import json
import subprocess
import re
import time
from PIL import Image
import google.generativeai as genai
from pydantic import BaseModel, Field

# ==========================================
# 0. 基础环境配置 (已根据你的 start_docker_images_koopman_1.sh 脚本完全对齐)
# ==========================================
CONTAINER_NAME = "koopman_latest_sm120_martin"
CONTAINER_WORKDIR = "/home/elane/intelligent_shipping_ws/data_process"

# 配置 Gemini API Key
genai.configure(api_key=os.environ.get("AIzaSyAEryVR_uGzKXbL0Ygeqo0pXfZvqmbL9OU"))

# ==========================================
# 1. 定义 Gemini 结构化决策输出格式
# ==========================================
class TuningDecision(BaseModel):
    analysis: str = Field(description="对当前轮次生成的图片（航迹、速度曲线、误差趋势）和日志进行深度的视觉和数值诊断分析")
    is_feasible: bool = Field(description="核心判断：根据图表质量和误差指标，当前代码是否可行（已达标、无发散/震荡现象）？可行填 True，仍需优化填 False")
    optimized_koopman_code: str = Field(description="如果 is_feasible 为 False，请提供全新优化重写后的 koopman.py 源码；如果为 True，请留空")
    optimized_train_code: str = Field(description="如果 is_feasible 为 False，请提供全新优化重写后的 train_multistep_voyage.py 源码；如果为 True，请留空")

# ==========================================
# 2. 核心控制管道 (重点优化：日志流式可见)
# ==========================================
def run_container_pipeline():
    """在宿主机通过 docker exec 驱动容器运行训练与评估"""
    print(f"\n🚀 [Docker] 正在驱动容器 {CONTAINER_NAME} 启动工作区内的模型训练...")
    print("💡 实时训练日志流如下（若画面不动请稍等，模型正在前向计算）：\n" + "-"*50)
    
    # 1. 运行训练：移除了 capture_output=True，让 Epoch 进度直接打印在屏幕上
    train_cmd = [
        "docker", "exec", "-w", CONTAINER_WORKDIR, CONTAINER_NAME,
        "python3", "train_multistep_voyage.py", "--epochs", "15"
    ]
    subprocess.run(train_cmd) # 流式实时输出
    
    print("-"*50 + f"\n📊 [Docker] 正在驱动容器运行评估并生成图表 (test_and_plot.py)...")
    
    # 2. 运行作图与评估脚本
    test_cmd = [
        "docker", "exec", "-w", CONTAINER_WORKDIR, CONTAINER_NAME,
        "python3", "test_and_plot.py"
    ]
    # 这里需要捕获输出，用来提取数字误差指标投喂给 Gemini
    result = subprocess.run(test_cmd, capture_output=True, text=True, encoding='utf-8')
    stdout = result.stdout
    stderr = result.stderr
    
    # 将作图脚本的输出和可能存在的报错打印出来，方便排查
    print("\n--- [test_and_plot.py] 控制台实时回显开始 ---")
    print(stdout)
    if stderr:
        print("⚠️ 容器内标准错误输出 (Stderr):")
        print(stderr)
    print("--- [test_and_plot.py] 控制台实时回显结束 ---\n")
    
    # 3. 正则解析基础数字指标
    vel_error, acc_error = 999.0, 999.0
    match_vel = re.search(r"平均速度误差\s*:\s*([\d\.]+)", stdout)
    match_acc = re.search(r"平均加速度误差\s*:\s*([\d\.]+)", stdout)
    if match_vel: vel_error = float(match_vel.group(1))
    if match_acc: acc_error = float(match_acc.group(1))
    
    return vel_error, acc_error, stdout

def read_local_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    return ""

def write_local_file(path, content):
    with open(path, 'w', encoding='utf-8') as f: f.write(content)

# ==========================================
# 3. 自动化迭代主循环
# ==========================================
def main():
    MAX_ITERATIONS = 8  # 最大允许自主迭代轮数
    
    model = genai.GenerativeModel(
        model_name='gemini-1.5-pro',
        generation_config={"response_mime_type": "application/json", "response_schema": TuningDecision}
    )
    
    print("⚡ 船舶动力学 Koopman 智能闭环调优引擎已就位...")
    
    for iteration in range(MAX_ITERATIONS):
        print(f"\n====================== 🔄 自动化寻优第 {iteration+1}/{MAX_ITERATIONS} 轮 ======================")
        
        # Step 3.1: 驱动 Docker 跑数与绘图
        vel_err, acc_err, output_log = run_container_pipeline()
        
        # Step 3.2: 捕获由于 Volume 挂载自动同步到宿主机的图片
        images_payload = []
        plot_paths = [
            "test_analysis/trajectory_comparison.png", 
            "test_analysis/u_prediction.png",          
            "test_analysis/velocity_error_curve.png"   
        ]
        
        for path in plot_paths:
            if os.path.exists(path):
                images_payload.append(Image.open(path))
        
        if not images_payload:
            print("⚠️ 警告：宿主机未能在 test_analysis/ 目录下检测到图片！请确认容器内是否成功生成。")

        # Step 3.3: 读取宿主机当前的代码
        current_koopman = read_local_file("koopman.py")
        current_train = read_local_file("train_multistep_voyage.py")
        
        # Step 3.4: 构建多模态专家 Prompt
        prompt = f"""
你是一个精通船舶控制工程、水动力学辨识以及 Koopman 算子理论的顶尖 AI 算法科学家。
我们为你附加了当前轮次测试生成的【多步推演可视化航迹图】、【速度拟合图】和【误差累积曲线图】。

【当前系统可观测状态】
- 平均速度误差: {vel_err} m/s
- 平均加速度误差: {acc_err} m/s^2
- 测试脚本打印的完整控制台日志:
{output_log}

【当前宿主机对应的源码】
--- koopman.py ---
{current_koopman}

--- train_multistep_voyage.py ---
{current_train}

【你的核心任务】
1. 视觉诊断：通过附加的图片，观察红色预测曲线（PI-Koopman）相较于绿色真实曲线（GT），是否在打舵转弯时出现了发散（Drift）、相位滞后（Phase Lag）、高频震荡（Oscillation）等物理层面的拟合缺陷？
2. 可行性决策：
   - 如果红线与绿线几乎完美重合，时序误差没有随着步数大幅累积，且数字指标达标，请在 JSON 中设置 `is_feasible = true`。
   - 如果发现明显的拟合缺陷或代码中存在明显的 Loss 缺失 Bug（例如漏掉了直接对速度序列状态 pred_seq 的约束），请设置 `is_feasible = false`。
3. 代码重构：当不可行时，顺着你的诊断结论对 `koopman.py` 或 `train_multistep_voyage.py` 进行针对性修改，提供更严谨的改进方案。
"""

        print("🧠 正在将最新代码、日志以及可视化图表打包提交给 Gemini 进行视觉诊断...")
        api_inputs = [prompt] + images_payload
        
        try:
            response = model.generate_content(api_inputs)
            decision = json.loads(response.text)
            
            print(f"\n🔍 Gemini 专家视觉审计结论：\n{decision['analysis']}\n")
            print(f"🎯 是否通过可行性评估: 【{decision['is_feasible']}】")
            
            if decision['is_feasible']:
                print("🎉 成功！智能体判定当前代码完全可行且拟合性能达标。自动化寻优结束。")
                break
            
            # 执行代码复写覆盖
            print("💾 正在自动将优化后的新代码注入工作区...")
            if decision['optimized_koopman_code'].strip():
                write_local_file("koopman.py.bak", current_koopman)
                write_local_file("koopman.py", decision['optimized_koopman_code'])
            if decision['optimized_train_code'].strip():
                write_local_file("train_multistep_voyage.py.bak", current_train)
                write_local_file("train_multistep_voyage.py", decision['optimized_train_code'])
                
            print("⏳ 准备进入下一轮 Docker 编译与训练循环...")
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ 迭代过程中发生异常: {e}，5秒后重试...")
            time.sleep(5)

if __name__ == "__main__":
    main()