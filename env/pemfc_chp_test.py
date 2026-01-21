'''
Author: Jiale Cheng &&cjl2646289864@gmail.com
Date: 2025-12-11 22:09:38
LastEditors: Jiale Cheng &&cjl2646289864@gmail.com
LastEditTime: 2026-01-21 07:35:07
FilePath: \PC\env_2\pemfc_chp_test.py
Description: 单纯作为测试脚本使用，加载训练好的模型进行推理，并绘图分析结果
Copyright (c) 2026 by cjl2646289864@gmail.com, All Rights Reserved. 
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pemfc_chp_train import PEMFCCHPEnv  # 从环境文件导入类
os.system('cls' if os.name == 'nt' else 'clear')
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_test():
    # 1. 初始化环境
    test_env = DummyVecEnv([lambda: PEMFCCHPEnv()])
    
    # 2. 添加 VecMonitor 包装（与训练环境保持一致）
    from stable_baselines3.common.vec_env import VecMonitor
    test_env = VecMonitor(test_env)
    
    # 3. 加载归一化统计量 (非常重要，否则模型推理会乱套)
    if os.path.exists("vec_normalize.pkl"):
        test_env = VecNormalize.load("vec_normalize.pkl", test_env)
        test_env.training = False      # 关闭更新
        test_env.norm_reward = False   # 测试不缩放奖励
    
    # 4. 加载训练好的模型
    model_path = "./logs/best_model.zip" if os.path.exists("./logs/best_model.zip") else "ppo_pemfc_final.zip"
    model = PPO.load(model_path, device="cpu")

    # 5. 运行一个完整的回合
    obs = test_env.reset()
    done = False
    history = []

    print(f">>> 正在加载模型 {model_path} 进行测试...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        history.append(info[0]) # 获取环境内部数据

    return history

def plot_analysis():
    # 读取结果
    history = np.load("test_results.npy", allow_pickle=True)
    # 数据提取
    P_gen_final = np.array([x['P_gen_W'] for x in history])     # kW生成的电功率与SOC和热水箱交互后剩下的电功率
    P_load = np.array([x['P_need_W'] for x in history])         # kW 电负载需求
    P_raw_W = np.array([x['P_raw_W'] for x in history])     # kW PEMFC 原始发电功率
    
    Q_gen = np.array([x['Q_gen_kW'] for x in history])          # kW PEMFC 实际产热
    Q_heater = np.array([x['Q_heater'] for x in history])       # kW 加热棒产热
    Q_load = np.array([x['Q_need_kW'] for x in history])        # kW 热负载需求
    Q_loss = np.array([x['Q_loss'] for x in history])           # kW 热损失
    Q_gen_used = np.array([x['Q_gen_used'] for x in history])   # kW 实际被用掉的 PEMFC 产热
    
    
    SOC = np.array([x['SOC'] for x in history])
    SOH = np.array([x['SOH'] for x in history])
    T_tank_C = np.array([x['T_tank_C'] for x in history])
    
    reward_list = np.array([x['reward'] for x in history])
    
    t = np.arange(len(P_gen_final))
    
    # ===============================
    # 2. 电功率指标
    # ===============================
    # 电跟踪率（截断，防止>1和除0）
    eta_e_t = P_gen_final / (P_load)
    eta_e_avg = np.mean(eta_e_t)

    # ===============================
    # 3. 热功率指标
    # ===============================
    # 实际供热（实际被用掉的 PEMFC 产热 + 电加热）
    Q_supply = Q_gen_used + Q_heater

    eta_h_t = Q_supply / (Q_load)
    eta_h_avg = np.mean(eta_h_t)
    # ===============================
    # 4. SOC 指标
    # ===============================
    SOC_mean = np.mean(SOC)
    SOC_min = np.min(SOC)
    SOC_max = np.max(SOC)
    # ===============================
    # 5. SOH 指标
    # ===============================
    SOH_init = SOH[0]
    SOH_final = SOH[-1]
    SOH_decay = SOH_init - SOH_final
    SOH_decay_rate = SOH_decay / len(SOH)
    # ===============================
    # 6. 水箱温度指标
    # ===============================
    T_mean = np.mean(T_tank_C)
    T_min = np.min(T_tank_C)
    T_max = np.max(T_tank_C)
    # ===============================
    # 7. 整体效率（基于“有用能量”）
    # η = (有效供电 + 有效供热) / (总发电 + 总供热)
    # 长时间尺度我觉得可以忽略热水箱和SOC温度变化
    # ===============================
    E_e_useful = np.sum(np.minimum(P_gen_final, P_load))
    E_h_useful = np.sum(np.minimum(Q_supply, Q_load))

    E_e_gen = np.sum(P_gen_final)
    E_h_gen = np.sum(Q_supply)

    eta_overall = (E_e_useful + E_h_useful) / (E_e_gen + E_h_gen )
    # ===============================
    # 8. 打印关键统计指标（论文用）
    # ===============================
    print("========== Performance Summary ==========")
    print(f"Average Electric Tracking Rate η_e: {eta_e_avg:.3f}")
    print(f"Average Thermal  Tracking Rate η_h: {eta_h_avg:.3f}")
    print(f"Overall Energy Efficiency η: {eta_overall:.3f}")
    print(f"SOC: mean={SOC_mean:.3f}, min={SOC_min:.3f}, max={SOC_max:.3f}")
    print(f"SOH decay: {SOH_decay:.5f}, decay rate per step: {SOH_decay_rate:.5e}")
    print(f"T_tank: mean={T_mean:.2f} °C, min={T_min:.2f}, max={T_max:.2f}")
    # =====================================================
    # 9. 绘图
    # =====================================================
    # ---------- 电功率 ----------
    plt.figure(figsize=(10, 4))
    plt.plot(t, P_load, label="Electric Load", linestyle="--")
    plt.plot(t, P_gen_final, label="PEMFC Net Power")
    plt.xlabel("Time step")
    plt.ylabel("Power (kW)")
    plt.title("Electrical Power")
    plt.legend()
    plt.grid(True)
    plt.savefig("电功率曲线.png", dpi=300)
    # ---------- 热功率 ----------
    plt.figure(figsize=(10, 4))
    plt.plot(t, Q_load+ Q_loss, label="need Load", linestyle="--")
    plt.plot(t, Q_heater+Q_gen, label="Heater Power")
    plt.xlabel("Time step")
    plt.ylabel("Power (kW)")
    plt.title("Thermal Power")
    plt.legend()
    plt.grid(True)
    plt.savefig("热功率曲线.png", dpi=300)
    '''
    # ---------- 电跟踪率 ----------
    plt.figure(figsize=(10, 4))
    plt.plot(t, eta_e_t, label="Electric Tracking Rate")
    plt.xlabel("Time step")
    plt.ylabel("η_e")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig("电功率跟踪率曲线.png", dpi=300)

    # ---------- 热跟踪率 ----------
    plt.figure(figsize=(10, 4))
    plt.plot(t, eta_h_t, label="Thermal Tracking Rate")
    plt.xlabel("Time step")
    plt.ylabel("η_h")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig("热功率跟踪率曲线.png", dpi=300)
    '''
    # ---------- SOC ----------
    plt.figure(figsize=(10, 4))
    plt.plot(t, SOC)
    plt.xlabel("Time step")
    plt.ylabel("SOC")
    plt.title("Battery SOC")
    plt.grid(True)
    plt.savefig("电池SOC曲线.png", dpi=300)

    # ---------- SOH ----------
    plt.figure(figsize=(10, 4))
    plt.plot(t, SOH)
    plt.xlabel("Time step")
    plt.ylabel("SOH")
    plt.title("PEMFC Stack SOH")
    plt.grid(True)
    plt.savefig("PEMFC堆寿命SOH曲线.png", dpi=300)

    # ---------- 水箱温度 ----------
    plt.figure(figsize=(10, 4))
    plt.plot(t, T_tank_C)
    plt.xlabel("Time step")
    plt.ylabel("Temperature (°C)")
    plt.title("Water Tank Temperature")
    plt.grid(True)
    plt.savefig("水箱温度曲线.png", dpi=300)
    '''
    # ---------- reward ----------
    plt.figure(figsize=(10, 4))
    plt.plot(t, reward_list)
    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.title("Reward per Step")
    plt.grid(True)
    plt.savefig("每步奖励reward曲线.png", dpi=300)

    # ---------- loss ----------
    plt.figure(figsize=(10, 4))
    plt.plot(t, Q_loss)
    plt.xlabel("Time step")
    plt.ylabel("Heat Loss (kW)")
    plt.title("Heat Loss per Step")
    plt.grid(True)
    plt.savefig("每步热损失曲线.png", dpi=300)
    
    '''
    plt.show()
    print(">>> 测试结束")
    

if __name__ == "__main__":
    results = run_test()
    #保存结果
    #np.save("test_results.npy", results)
    plot_analysis()