import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据加载与预处理 (Data Loading & Prep)
# ==========================================
# 读取数据
df = pd.read_csv('aggregated.csv')

# 清洗数据：将 'err' 替换为 NaN
df = df.replace('err', np.nan)

# 定义需要处理的所有数值列
cols_to_clean = [
    'CPU_BIG_FREQ_KHz', 'CPU_MID_FREQ_KHz', 'CPU_LITTLE_FREQ_KHz', 'AVG_SOC_TEMP',
    'RougeMesuré', 'VertMesuré', 'BleuMesuré', 'Brightness',
    'GPU0_FREQ', 'GPU_1FREQ', 'GPU_MEM_AVG', 'TOTAL_DATA_WIFI_BYTES',
    'CPU_BIG_ENERGY_AVG_UWS', 'CPU_MID_ENERGY_AVG_UWS', 'CPU_LITTLE_ENERGY_AVG_UWS',
    'Display_ENERGY_AVG_UWS', 'GPU_ENERGY_AVG_UWS', 'WLANBT_ENERGY_AVG_UWS'
]

# 强制转换为数值类型并填充缺失值（使用中位数）
for col in cols_to_clean:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# 创建核心特征：频率与温度的交互项 (Interaction Terms)
# 这一步体现了模型对“热限频”等非线性物理特性的捕捉
df['Interact_Big'] = df['CPU_BIG_FREQ_KHz'] * df['AVG_SOC_TEMP']
df['Interact_Mid'] = df['CPU_MID_FREQ_KHz'] * df['AVG_SOC_TEMP']
df['Interact_Little'] = df['CPU_LITTLE_FREQ_KHz'] * df['AVG_SOC_TEMP']

# ==========================================
# 2. 模型训练与预测 (Modeling)
# ==========================================
# 定义 6 个主要组件的目标变量和特征
targets = {
    'CPU_Big': 'CPU_BIG_ENERGY_AVG_UWS',
    'CPU_Mid': 'CPU_MID_ENERGY_AVG_UWS',
    'CPU_Little': 'CPU_LITTLE_ENERGY_AVG_UWS',
    'Display': 'Display_ENERGY_AVG_UWS',
    'GPU': 'GPU_ENERGY_AVG_UWS',
    'WLAN_BT': 'WLANBT_ENERGY_AVG_UWS'
}

feature_sets = {
    'CPU_Big': ['CPU_BIG_FREQ_KHz', 'AVG_SOC_TEMP', 'Interact_Big'],
    'CPU_Mid': ['CPU_MID_FREQ_KHz', 'AVG_SOC_TEMP', 'Interact_Mid'],
    'CPU_Little': ['CPU_LITTLE_FREQ_KHz', 'AVG_SOC_TEMP', 'Interact_Little'],
    'Display': ['RougeMesuré', 'VertMesuré', 'BleuMesuré', 'Brightness'],
    'GPU': ['GPU0_FREQ', 'GPU_1FREQ', 'GPU_MEM_AVG'],
    'WLAN_BT': ['TOTAL_DATA_WIFI_BYTES']
}

# 训练每个组件的模型并保存预测结果
preds = pd.DataFrame(index=df.index)
for name, target_col in targets.items():
    X = df[feature_sets[name]]
    y = df[target_col]

    # 使用线性回归
    model = LinearRegression()
    model.fit(X, y)
    preds[name] = model.predict(X)

# 计算总功耗的实际值与预测值
df['Total_Actual'] = df[list(targets.values())].sum(axis=1)
df['Total_Pred'] = preds.sum(axis=1)

# 计算残差 (Residual) 和 相对误差 (Relative Error)
df['Residual'] = df['Total_Actual'] - df['Total_Pred']
df['Rel_Error_Pct'] = (df['Residual'] / df['Total_Actual']) * 100

# ==========================================
# 3. 专家级绘图 (Advanced Visualization)
# ==========================================
# 设置绘图风格
plt.style.use('seaborn-v0_8-ticks')
fig = plt.figure(figsize=(16, 12))
# 使用 GridSpec 布局：3行2列
gs = fig.add_gridspec(3, 2)

# --- 图 A：宏观拟合能力 (Main Comparison) ---
# 占据第一行整行
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, df['Total_Actual'], label='Actual Measured Power', color='#2C3E50', alpha=0.7, linewidth=1)
ax1.plot(df.index, df['Total_Pred'], label='Model Prediction', color='#E74C3C', alpha=0.7, linewidth=1, linestyle='--')
ax1.set_title('A. Overall Prediction Capability: Actual vs Predicted Power', fontsize=14, loc='left', fontweight='bold')
ax1.set_ylabel('Power ($\mu Ws$)', fontsize=12)
ax1.set_xlim(0, len(df))
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# --- 图 B：误差演变趋势 (Error Trend) ---
# 占据第二行整行
ax2 = fig.add_subplot(gs[1, :])
# 绘制灰色的瞬时误差散点
ax2.scatter(df.index, df['Rel_Error_Pct'], color='gray', alpha=0.2, s=10, label='Instantaneous Error (%)')
# 计算并绘制 50点 移动平均趋势线 (Error Trend)
df['Error_Trend'] = df['Rel_Error_Pct'].rolling(window=50, center=True).mean()
ax2.plot(df.index, df['Error_Trend'], color='#E74C3C', linewidth=2, label='Error Trend (50-point Moving Avg)')
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.set_title('B. Model Stability: Relative Error Evolution (%)', fontsize=14, loc='left', fontweight='bold')
ax2.set_ylabel('Relative Error (%)', fontsize=12)
ax2.set_xlabel('Time Sequence (t)', fontsize=12)
ax2.set_ylim(-50, 50)  # 限制Y轴范围以聚焦主体误差
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# --- 图 C：残差正态性检验 (Normality Check) ---
# 占据第三行左侧
ax3 = fig.add_subplot(gs[2, 0])
sns.histplot(df['Residual'], kde=True, color='#3498DB', ax=ax3, bins=50)
ax3.set_title('C. Residual Normality Check (Distribution)', fontsize=14, loc='left', fontweight='bold')
ax3.set_xlabel('Residual Error ($\mu Ws$)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.grid(True, alpha=0.3)

# --- 图 D：温度敏感性诊断 (Temperature Sensitivity) ---
# 占据第三行右侧
ax4 = fig.add_subplot(gs[2, 1])
# 绘制散点图，颜色映射到相对误差大小
sc = ax4.scatter(df['AVG_SOC_TEMP'], df['Residual'], c=df['Rel_Error_Pct'], cmap='coolwarm', alpha=0.6, s=20, vmin=-20,
                 vmax=20)
ax4.set_title('D. Temperature Sensitivity: Residuals vs Temp', fontsize=14, loc='left', fontweight='bold')
ax4.set_xlabel('SoC Temperature (Raw Unit)', fontsize=12)
ax4.set_ylabel('Residual Error ($\mu Ws$)', fontsize=12)
plt.colorbar(sc, ax=ax4, label='Relative Error (%)')
ax4.axhline(0, color='black', linestyle='--', linewidth=1)
ax4.grid(True, alpha=0.3)

# 保存图片
plt.tight_layout()
plt.savefig('improved_residual_analysis.png', dpi=300)
plt.show()

print("图表已生成：improved_residual_analysis.png")