'''
Author: Jiale Cheng &&cjl2646289864@gmail.com
Date: 2025-12-11 22:09:38
LastEditors: Jiale Cheng &&cjl2646289864@gmail.com
LastEditTime: 2026-01-24 20:59:42
FilePath: \PC\env\pemfc_chp_train.py
Description: 单纯作为 PEMFC-CHP 强化学习环境的定义与训练脚本
Copyright (c) 2026 by cjl2646289864@gmail.com, All Rights Reserved. 
'''
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pandas as pd
import os
os.system('cls' if os.name == 'nt' else 'clear')

class PEMFCCHPEnv(gym.Env):
    """
    PEMFC-CHP 强化学习环境
    """

    def __init__(self):
        super().__init__()

        # -----------------------
        # 参数
        # -----------------------
        self.p = {

            # -------- 系统规模 --------
            'N_stacks': 15,
            'J_max': 1.876,
            'J_min': 0.0,

            # -------- PEMFC --------
            'N_cell': 30,
            'A_fc': 200.0,
            'T_op': 343.15,
            'P_H2_in_bar': 3.0,
            'P_O2_in_bar': 3.0,
            'F': 96487,
            'R': 8.314,
            'atm_to_pa': 101325,
            'LHV_H2': 241820.0,

            # -------- 热系统 --------
            'M_water': 100.0,
            'cp_water_liq': 4184.0,
            'R_air_thermal': 1.5,
            'T_amb': 298.15,
            'eff_heating_rod': 0.95,
            # -------- 电池 --------
            'C_batt_Ah': 100.0,
            'V_batt': 48.0,
            'eta_batt': 0.95,# 这是充放电效率的乘积
            'SOC_init': 0.5,

            # -------- 初始 --------
            'T_tank_init': 348.15,

            # -------- 电化学 --------
            'V_act_c1': -0.944872,
            'V_act_c2': 0.002631,
            'V_act_c3': 5.303807e-05,
            'V_act_c4': -1.418930e-04,
            'R_internal': 0.021698,

            # -------- 时间 --------
            'dt': 300,

            # -------- Reward --------
            'wP': 5.0,              # ↓ 原 10 → 稍微降，避免电功率独裁
            
            'SOC_target': 0.5,
            'SOC_band': 0.05,       # ↑ 原 0.1 → 给 RL 更宽容区间
            'T_tank_target': 273.15+60,#暂时定为80℃
            'T_band': 2.5,         # ↑ 原 10 → 减少热侧震荡

            # -------- 论文中的 SOH 关键参数 --------
            'kappa_1': 13.79e-6,  # 启停单次电压降 (V)
            'kappa_2': 0.04185e-6, # 变载电压降 (V/kW)
            'kappa_3': 8.662e-6,   # 低功率运行电压降 (V/h)
            'kappa_4': 10.0e-6,    # 高功率运行电压降 (V/h)
            'P_fc_low': 5000.0,    # 低功率阈值 5kW
            'P_fc_high': 40000.0,  # 高功率阈值 40kW

            'Ea_base': 31700,      # 电池活化能基准
            'R_gas': 8.314,        # 气体常数
            'T_batt': 298.15,      # 假设电池工作温度 25℃
            'alpha_c': 31630,      # 前因子
            'z_power': 0.55,       # 时间幂律因子
            'V_nom': 1.1,          # 燃料电池单体初始标称电压
            
            # -------- 数据 --------
            'load_profiles': "processed_PEMFC_load.csv",
        }

        '''
        description: 加载负荷数据
        return {*}
        '''        
        path = os.path.join(os.path.dirname(__file__), self.p['load_profiles'])
        if os.path.exists(path):
            df = pd.read_csv(path)
            self.load_profiles = pd.DataFrame({
                'P_e': df.iloc[:, 1],
                'Q_th': df.iloc[:, 2]
            })
        else:
            self.load_profiles = pd.DataFrame({
                'P_e': np.ones(1000) * 10.0,
                'Q_th': np.ones(1000) * 5.0
            })

        '''
        description: 强化学习动作空间设置
        return {*}
        '''
        self.action_space = spaces.Box(
            low=np.array([self.p['J_min']] * self.p['N_stacks'] + [0.0]),
            high=np.array([self.p['J_max']] * self.p['N_stacks'] + [1.0]),
            dtype=np.float64
        )

        obs_dim = 4 + self.p['N_stacks'] + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        self.n_steps_total = len(self.load_profiles)
        self.reset()

    '''
    description: 环境初始化
    param {*} self:
    param {*} seed:随机种子，这里无所谓，因为环境是确定性的
    param {*} options:这里也无所谓
    return {*}
    '''    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.state_T_tank = 60+273.15
        self.state_SOC = 0.5
        self.state_j_prev = np.zeros(self.p['N_stacks'])
        self.alpha_prev = 0.5
        self.state_fc_V_loss = 0.0     # 累积电压降 (V)
        self.state_fc_SOH = 1.0        # 燃料电池 SOH (从1.0开始)
        self.state_P_fc_prev = 0.0     # 记录上一时刻燃料电池总功率
        return self._get_obs(), {}

    '''
    description: 获取观测值
    param {*} self:
    return {*}
    '''    
    def _get_obs(self):
        idx = min(self.current_step, self.n_steps_total - 1)
        return np.concatenate([
            np.array([
                self.load_profiles['P_e'].iloc[idx],
                self.load_profiles['Q_th'].iloc[idx],
                self.state_T_tank,
                self.state_SOC
            ], dtype=np.float64),
            self.state_j_prev.astype(np.float64),
            np.array([self.alpha_prev], dtype=np.float64)
        ])

    '''
    description: 解析强化学习动作
    param {*} self:
    param {*} action:强化学习直接输出的动作
    return {*}
    '''    
    def _parse_action(self, action):
        p = self.p
        j_vec = np.clip(action[:p['N_stacks']], p['J_min'], p['J_max'])
        raw_alpha = action[-1]
        return j_vec, raw_alpha

    '''
    description: 基于论文1的SOH退化模型
    return {
        delta_SOH_fc: 本步燃料电池 SOH 变化量
    }
    '''    
    def _update_PEMFC_SOH(self, P_fc_now):
        p = self.p
        dt_hr = p['dt'] / 3600.0

        # 1. 计算这一步产生的电压降 (论文核心式子)
        delta_V = 0.0
        
        # 启停损耗
        if self.state_P_fc_prev <= 10.0 and P_fc_now > 10.0:
            delta_V += p['kappa_1']
        
        # 变载损耗
        delta_P_kw = abs(P_fc_now - self.state_P_fc_prev) / 1000.0
        delta_V += p['kappa_2'] * delta_P_kw
        
        # 稳态运行损耗
        if P_fc_now > 0 and P_fc_now < p['P_fc_low']:
            delta_V += p['kappa_3'] * dt_hr
        elif P_fc_now > p['P_fc_high']:
            delta_V += p['kappa_4'] * dt_hr

        # 2. 将电压降映射为 SOH 的减少
        # 假设燃料电池单体初始电压 1.1V，当总下降达到初始电压的 10% (即 0.11V) 时寿命终结
        V_EOL = 0.11  # 寿命终点阈值
        delta_SOH_fc = delta_V / V_EOL
        
        # 更新累积状态
        self.state_fc_V_loss += delta_V
        self.state_fc_SOH -= delta_SOH_fc
        
        return delta_SOH_fc
    '''
    description: 训练过程中获取当前负载
    param {*} self:
    return {
        P_need: 当前电功率需求 (W)
        Q_need: 当前热功率需求 (W)
    }
    '''    
    def _get_load(self):
        idx = min(self.current_step, self.n_steps_total - 1)
        P_need = self.load_profiles['P_e'].iloc[idx] * 1000
        Q_need = self.load_profiles['Q_th'].iloc[idx] * 1000
        return P_need, Q_need

    '''
    description: 训练过程中运行的 PEMFC 物理模型
    param {*} self:
    param {*} j_vec:电流密度
    return {
        P_net: PEMFC 系统净电功率输出 (W)
        Q_cool: PEMFC 系统产热 (W)
        mol_H2_rate: 氢气消耗速率 (mol/s)
    }
    '''    
    def _run_pemfc(self, j_vec):
        sim = self._run_system_physics(j_vec)
        return sim['P_net'], sim['Q_cool'], sim['mol_H2_rate'], sim.get('E_h_W', 0.0)

    '''
    description: 更新SOC唯一入口
    param {*} self:
    param {*} SOC_old:上次的SOC
    param {*} P_charge:PEMFC给电池充电功率（正充电，负吸电）
    param {*} P_heater:电池给电加热棒放电功率
    return {
        SOC_new:更新后的SOC
    }
    '''    
    def _update_SOC(self, SOC_old, P_batt_charge_from_fc, P_batt_discharge_output):
        """
        SOC 更新（唯一入口，基于能量守恒）

        参数说明：
        - SOC_old: step 开始时的 SOC (0..1)
        - P_batt_charge_from_fc: 燃料电池向电池的充电功率（W，正）
        - P_batt_discharge_output: 电池对外输出的放电功率（W，正，包含给加热棒和负载的输出功率）

        能量守恒：
            E_new = E_old + eta_ch * P_charge * dt - (P_dis_out / eta_dis) * dt
        然后 SOC = E_new / E_capacity
        """

        p = self.p

        # 电池容量 (Wh)
        capacity_Wh = p['C_batt_Ah'] * p['V_batt']
        dt_hr = p['dt'] / 3600.0

        # 将总充/放效率分拆为充放两端（p['eta_batt'] 原来为乘积）
        eta_total = p.get('eta_batt', 1.0)
        # 均分到充电/放电端（简单且常用的做法）
        eta_ch = math.sqrt(max(eta_total, 0.0))
        eta_dis = eta_ch

        # 旧能量 (Wh)
        E_old_Wh = SOC_old * capacity_Wh

        # 充电端实际写入电池的能量 (Wh)
        E_ch_Wh = eta_ch * max(P_batt_charge_from_fc, 0.0) * dt_hr

        # 放电端从电池内部抽取的能量 (Wh)（注意 P_batt_discharge_output 是对外输出）
        E_dis_internal_Wh = (max(P_batt_discharge_output, 0.0) / max(eta_dis, 1e-8)) * dt_hr

        E_new_Wh = E_old_Wh + E_ch_Wh - E_dis_internal_Wh

        SOC_new = E_new_Wh / max(capacity_Wh, 1e-9)
        return np.clip(SOC_new, 0.0, 1.0)


    '''
    description: 热管理（水箱 + 电加热棒）
    param {*} self:
    param {*} Q_gen: PEMFC产生的热量
    param {*} Q_need: 负载需要的热量
    param {*} P_need: 负载需要的电功率
    param {*} P_gen_raw: PEMFC原始发电功率
    param {*} alpha: 控制参数（这里已经被优化掉了）   
    return {
        P_heater_fc: 电加热棒耗电——燃料电池
        P_heater_batt: 电加热棒耗电——电池
        Q_heater: 电加热棒产热
        P_charge: 给电池充电功率
        Q_loss: 环境散热
        Q_gen_used: 燃料电池产热被利用的部分
    }
    '''    
    def _thermal_management(self, Q_gen, Q_need, P_need, P_gen_raw, alpha):
        p = self.p

        # ============================================================
        # 1. 当前水箱可用热量 & 热损失
        # ============================================================
        Q_tank_avail = (
            p['M_water'] * p['cp_water_liq']
            * (self.state_T_tank - p['T_tank_target']-p ['T_band'])
        ) / p['dt']

        # 热损失应为瞬时热功率（W）：(T_tank - T_amb)/R (K / (K/W) = W)
        # 之前乘以 dt 会把单位变成能量 (J)，与其它量（W）混用会导致单位错误。
        Q_loss = (self.state_T_tank - p['T_amb']) / p['R_air_thermal']

        # 正值表示缺热
        Q_deficit = Q_need - (  Q_tank_avail) + Q_loss
        # Q_deficit =max(Q_deficit-Q_gen,0)  
        Q_gen_used = min(Q_deficit, Q_gen)
        Q_deficit -= Q_gen_used

        # ============================================================
        # 2. PEMFC 电功率富余
        # ============================================================
        P_surplus_fc = max(P_gen_raw - P_need, 0.0)

        # ============================================================
        # 3. SOC → 功率换算
        # ============================================================
        # 将 1.0 SOC 对应到在一个时间步长内的可转换功率 (W)
        # capacity_Wh = C_batt_Ah * V_batt
        # soc_to_w = capacity_Wh / (dt/3600) = capacity_Wh * 3600 / dt
        soc_to_w = (
            p['C_batt_Ah'] * p['V_batt']
            / (p['dt'] / 3600.0)
        )

        SOC_target = p['SOC_target']
        SOC_band = p['SOC_band']
        SOC_lower = SOC_target - SOC_band
        SOC_upper = SOC_target + SOC_band
        SOC_emergency = max(SOC_lower - 0.05, 0.0)

        # ============================================================
        # 4. SOC 主动恢复（只计算充电功率）
        # ============================================================
        P_charge = 0.0 #燃料电池给SOC充电
        if P_surplus_fc > 0:#电池缺电，功率有余，先充电
            P_charge = min(
                (SOC_target - self.state_SOC) * soc_to_w,#充到Target需要的电    
                P_surplus_fc
            )
            P_surplus_fc -= P_charge
        # ============================================================
        # 5. 电池最大可放电功率
        # ============================================================
        if Q_deficit > 0 and self.state_T_tank < p['T_tank_target']:
            P_batt_max = max(
                (self.state_SOC - SOC_emergency) * soc_to_w,
                0.0
            )
        else:
            P_batt_max = max(
                (self.state_SOC - SOC_lower) * soc_to_w,
                0.0
            )
        
        # ============================================================
        # 6. 电加热棒功率
        # ============================================================
        if Q_deficit > 0:
            P_heater_need = Q_deficit / p['eff_heating_rod']
            P_heater_max = P_surplus_fc + P_batt_max
            P_heater = min(P_heater_need, P_heater_max)
            P_heater_fc = min(P_heater, P_surplus_fc)
            P_heater_batt = P_heater - P_heater_fc
        else:
            P_heater_fc = 0.0
            P_heater_batt = 0.0
            P_heater = 0.0

        Q_heater = P_heater * p['eff_heating_rod']

        # ============================================================
        # 7. 返回（只返回“决策量”）
        # ============================================================
        return P_heater_fc,P_heater_batt, Q_heater, P_charge, Q_loss,Q_gen_used



    '''
    description: 计算奖励函数
    param {*} self:
    param {*} P_gen: 与热、SOC等交互后剩余的电功率
    param {*} P_need: 负载需要的电功率
    param {*} next_SOC: 下一时刻的电池SOC
    param {*} next_T: 下一时刻的水箱温度
    param {*} P_heater: 电加热棒耗电功率
    param {*} delta_SOH: 电池SOH变化量
    return {
        reward: 本步奖励值
    }
    '''    
    def _compute_reward(self, P_gen, P_need, next_SOC, next_T,  P_heater, delta_SOH):
        p = self.p
        reward = 0.0
        
        # Q_loss = (self.state_T_tank - p['T_amb']) / p['R_air_thermal']

        
        T_err = abs(next_T - p['T_tank_target'])
        P_err = abs(P_gen - P_need)
        SOC_err = abs(next_SOC - self.p['SOC_target'])
        
        # 不能让去辅助补热的电能没有奖励反馈
        reward += P_heater / 5000 *p['wP']
        
        #给 PEMFC 一个“追负载奖励”
        reward += 200.0 * np.exp(-0.002 * P_err)#x=100 y=73.57        
        reward += 400.0 * np.exp(-10 * SOC_err)#x=0.1 y=54.1
        reward += 500.0 * np.exp(-0.1 * T_err)#x=5  y=190.2
        
        T_high = p['T_tank_target'] + p['T_band']
        T_low  = p['T_tank_target'] - p['T_band']
        if next_T < T_low:
            reward -= 50.0 * (T_low - next_T)
        elif next_T > T_high:
            reward -= 50.0 * (next_T - T_high)
        else:
            reward += 50.0   # 在安全区间内给小正奖励
        
        
        if abs(next_SOC-p['SOC_target'])>0.1:#如果SOC差大于0.1
            reward -= 30.0*25 * abs(next_SOC-p['SOC_target'])#0.2*25*300=1500
        # 2. 论文核心：健康损耗惩罚 (负奖励)
        # 将电压降和 SOH 降转化为惩罚，权重需要根据量级微调
        reward -= delta_SOH * 1e7  # 电池 SOH 变化量
        
        return reward


    '''
    description: 环境执行一步动作
    param {*} self:
    param {*} action: 强化学习输出的动作
    return {*}
    '''    
    def step(self, action):
        p=self.p
        # ============================================================
        # 0. 解析动作
        # ============================================================
        j_vec, alpha = self._parse_action(action)

        # ============================================================
        # 1. 读取当前状态（明确区分 old / next）
        # ============================================================
        T_tank_old = self.state_T_tank
        SOC_old = self.state_SOC

        # ============================================================
        # 2. 负载与 PEMFC 计算
        # ============================================================
        P_need, Q_need = self._get_load()
        P_gen_raw, Q_gen, mol_H2, E_h_W = self._run_pemfc(j_vec)

        # ============================================================
        # 3. 热管理（只做功率/热量分配，不更新状态）
        # ============================================================
        (
            P_heater_fc,    #电加热棒耗电——燃料电池
            P_heater_batt,  # 电加热棒耗电——电池
            Q_heater,       # 电加热棒产热
            P_charge,       # 给电池充电功率
            Q_loss, Q_gen_used          # 环境散热

        ) = self._thermal_management(
            Q_gen=Q_gen,
            Q_need=Q_need,
            P_need=P_need,
            P_gen_raw=P_gen_raw,
            alpha=alpha
        )

        # ============================================================
        # 4. 电功率收支（系统对外净输出）
        # ============================================================
        P_gen = P_gen_raw - P_heater_fc - P_charge

        # ============================================================
        # 5. 电池给负载/加热棒补电（区分输出与能量上限）
        # ============================================================
        SOC_to_fc_output = 0.0
        if P_gen < P_need:  # 如果缺电
            P_deficit = P_need - P_gen  # 缺这么多

            # 计算电池可用的最小 SOC 下限（与 _thermal_management 保持一致）
            SOC_target = p['SOC_target']
            SOC_band = p['SOC_band']
            SOC_lower = SOC_target - SOC_band
            SOC_emergency = max(SOC_lower - 0.05, 0.0)
            SOC_min = SOC_emergency

            # 电池容量和放电效率
            capacity_Wh = p['C_batt_Ah'] * p['V_batt']
            eta_total = p.get('eta_batt', 1.0)
            eta_dis = math.sqrt(max(eta_total, 0.0))

            # 可用能量 (Wh)
            available_energy_Wh = max(self.state_SOC - SOC_min, 0.0) * capacity_Wh

            # 在当前时间步内，电池能够对外提供的最大输出功率（W，已考虑放电效率）
            max_output_power_W = available_energy_Wh * eta_dis * 3600.0 / p['dt']

            # 实际从电池对外输出给负载的功率（W）
            SOC_to_fc_output = min(max_output_power_W, P_deficit)
            P_gen += SOC_to_fc_output
        
        # ============================================================
        # 6. 更新水箱温度（只在这里更新）
        # ============================================================
        Q_net =  Q_gen_used + Q_heater - Q_need - Q_loss
        next_T = T_tank_old + (
            Q_net * self.p['dt']
            / (self.p['M_water'] * self.p['cp_water_liq'])
        )
        next_T = np.clip(next_T, 273.15, 373.15)
        # ============================================================
        # 7. 更新 SOC（只在这里更新）
        # ============================================================
        # 将 P_charge（来自 PEMFC 的充电功率）和电池对外放电合并传入 SOC 更新
        next_SOC = self._update_SOC(
            SOC_old=SOC_old,
            P_batt_charge_from_fc=P_charge,
            P_batt_discharge_output=(P_heater_batt + SOC_to_fc_output)
        )

        # ============================================================
        # 8. SOH 退化更新（用原始 PEMFC 输出）
        # ============================================================
        delta_SOH = self._update_PEMFC_SOH(P_gen_raw)

        # ============================================================
        # 9. 计算奖励（用 next 状态）
        # ============================================================
        reward = self._compute_reward(
            P_gen=P_gen,
            P_need=P_need,
            next_SOC=next_SOC,
            next_T=next_T,
            P_heater=P_heater_fc + P_heater_batt,
            delta_SOH=delta_SOH
        )

        # ============================================================
        # 10. 状态持久化（统一在末尾）
        # ============================================================
        self.state_T_tank = next_T
        self.state_SOC = next_SOC
        self.state_P_fc_prev = P_gen_raw
        self.state_j_prev = j_vec.copy()
        self.alpha_prev = alpha
        self.current_step += 1

        terminated = (self.current_step >= self.n_steps_total)

        # ============================================================
        # 11. info 记录（物理量一致）
        # ============================================================
        info = {
            'SOC': next_SOC,
            'T_tank_C': next_T - 273.15,
            'SOH': self.state_fc_SOH,

            # 同时提供 W 与 KW 单位的键，便于外部使用者选择
            'P_gen_W': P_gen,
            'P_gen_KW': P_gen / 1000.0,
            'P_gen_KWh': (P_gen / 1000.0) * (p['dt'] / 3600.0),
            'P_need_W': P_need,
            'P_need_KW': P_need / 1000.0,
            'P_need_KWh': (P_need / 1000.0) * (p['dt'] / 3600.0),
            'P_raw_W': P_gen_raw,
            'P_raw_KW': P_gen_raw / 1000.0,
            'P_raw_KWh': (P_gen_raw / 1000.0) * (p['dt'] / 3600.0),
            
            'H2_rate_mol_s': mol_H2,
            'H2_energy_W': E_h_W,
            'H2_energy_KW': E_h_W / 1000.0,
            'H2_energy_KWh': (E_h_W / 1000.0) * (p['dt'] / 3600.0),

            'Q_gen_W': Q_gen,
            'Q_gen_KW': Q_gen / 1000.0,
            'Q_gen_KWh': (Q_gen / 1000.0) * (p['dt'] / 3600.0),
            'Q_need_W': Q_need,
            'Q_need_KW': Q_need / 1000.0,
            'Q_need_KWh': (Q_need / 1000.0) * (p['dt'] / 3600.0),
            'Q_heater_W': Q_heater,
            'Q_heater_KW': Q_heater / 1000.0,
            'Q_heater_KWh': (Q_heater / 1000.0) * (p['dt'] / 3600.0),
            'Q_loss_W': Q_loss,
            'Q_loss_KW': Q_loss / 1000.0,
            'Q_loss_KWh': (Q_loss / 1000.0) * (p['dt'] / 3600.0),
            'Q_gen_used_W': Q_gen_used,
            'Q_gen_used_KW': Q_gen_used / 1000.0,
            'Q_gen_used_KWh': (Q_gen_used / 1000.0) * (p['dt'] / 3600.0),
            'reward': reward,
        }

        return self._get_obs(), reward, terminated, False, info


    '''
    description: 实际物理模型
    param {*} self:
    param {*} j_vec: 燃料电池电流密度向量
    return {
        P_net: PEMFC 系统净电功率输出 (W)
        Q_cool: PEMFC 系统产热 (W)
        mol_H2_rate: 氢气消耗速率 (mol/s)
    }
    '''    
    def _run_system_physics(self, j_vec):
        p = self.p
        P_tot, Q_tot, H2_tot = 0.0, 0.0, 0.0
        # 考虑衰减：电压基准随累积电压降减小
        V_base = p['V_nom'] - (self.state_fc_V_loss / p['N_cell'])#

        # 与选用热值一致的“热中性电压”(thermoneutral voltage)
        # E_tn = ΔH / (2F). 当使用 LHV 时约为 1.25V；使用 HHV 时约为 1.48V。
        h2_heat = p.get('LHV_H2', p.get('HHV_H2', 0.0))  # J/mol
        E_tn = h2_heat / (2.0 * p['F']) if h2_heat > 0 else 0.0

        for i, j in enumerate(j_vec):
            if j <= 1e-4: continue
            I = j * p['A_fc']
            # V_cell 现在基于受衰减影响的 V_base
            V_cell = max(0.45, V_base - j * p['R_internal']) #

            P_stack = p['N_cell'] * V_cell * I
            # 产热：Q = N_cell * I * (E_tn - V_cell)
            # 用 E_tn 替代常数 1.48，避免在 LHV/HHV 切换时能量守恒被破坏，从而导致“总效率>1”。
            Q_stack = p['N_cell'] * I * max(E_tn - V_cell, 0.0)
            mol_H2 = p['N_cell'] * I / (2 * p['F'])

            P_tot += P_stack
            Q_tot += Q_stack
            H2_tot += mol_H2

        # 将氢气摩尔流率转换为功率（W）
        E_h_W = H2_tot * h2_heat

        return {
            'P_net': P_tot,
            'Q_cool': Q_tot,
            'mol_H2_rate': H2_tot,
            'E_h_W': E_h_W,
        }

from stable_baselines3.common.callbacks import ( BaseCallback, EvalCallback, CheckpointCallback, CallbackList ) 
from datetime import datetime 
from datetime import datetime

class DetailedLogCallback(BaseCallback):
    def _on_step(self):
        infos = self.locals.get("infos", None)

        if infos and self.n_calls % 10 == 0:
            info = infos[0]

            # ==========================
            # 现实时间戳（wall-clock）
            # ==========================
            now = datetime.now()
            ts_str = now.strftime("%Y-%m-%d %H:%M:%S")

            # --- 记录为文本（TensorBoard 的 text 面板）---
            self.logger.record("time/wall", ts_str)

            log_keys = [
                'P_need_KW',
                'P_gen_KW',
                'Q_gen_KW',
                'Q_need_KW',
                'T_tank_C',
                'SOC',
            ]

            # ==========================
            # 数值型日志（曲线）
            # ==========================
            for k in log_keys:
                if k in info:
                    self.logger.record(f'custom/{k}', float(info[k]))

            '''            
            # ==========================
            # 终端实时打印（推荐）
            # ==========================
            msg = f"[{ts_str}] "
            msg += " | ".join(
                f"{k}={info[k]:.3f}" for k in log_keys if k in info
            )
            print(msg)
            '''

        return True
    
# =====================================================
# 训练脚本逻辑 (直接运行此文件开始训练)
# =====================================================
if __name__ == "__main__":
    import os
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, SubprocVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

    # --- 配置 ---
    LOG_DIR = "./logs/"
    os.makedirs(LOG_DIR, exist_ok=True)
    N_ENVS = min(max(2, os.cpu_count() - 2), 8)  # 自动检测CPU核心数
    print(f">>> 使用 {N_ENVS} 个并行环境 (多进程)")

    def make_env():
        def _init():
            return PEMFCCHPEnv()
        return _init

    # 1. 创建并包装环境 - 使用多进程
    train_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    train_env = VecMonitor(train_env, filename=os.path.join(LOG_DIR, "monitor.csv"))
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env()])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # 2. 设置回调
    eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_DIR, log_path=LOG_DIR,
                                 eval_freq=50000 // N_ENVS, deterministic=True)
    checkpoint_callback = CheckpointCallback(save_freq=100000 // N_ENVS, save_path=LOG_DIR, name_prefix="ppo_pemfc")
    
    # 3. 定义 PPO 模型
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        device="cpu",
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,  # 减小以适应多进程
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        tensorboard_log=LOG_DIR,
        seed=42
    )

    # 4. 开始训练
    print(">>> 训练开始 (多进程并行)...")
    model.learn(total_timesteps=1000000, callback=CallbackList([eval_callback, checkpoint_callback, DetailedLogCallback()]))

    # 5. 保存最终成果
    model.save("ppo_pemfc_final")
    train_env.save("vec_normalize.pkl") # 测试时必须加载此文件
    print(">>> 训练完成，模型与归一化参数已保存。")
    
    # 6. 关闭环境
    train_env.close()