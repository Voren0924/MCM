import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# 配置美赛专用学术配色
MCM_PALETTE = {
    'primary': '#2878B5',    # 深蓝：主数据点
    'secondary': '#9AC9DB',  # 浅蓝：辅助对比
    'accent_1': '#F8AC8C',   # 橙色：分布填充
    'accent_2': '#C82423',   # 深红：参考线/边框
    'soft_red': '#FF8884'    # 浅红：高亮
}

# 1. 数据清洗与对齐
def clean_data(file_agg, file_gt):
    df_a = pd.read_csv(file_agg)
    df_g = pd.read_csv(file_gt)
    # 处理数值格式差异（如欧洲格式的逗号小数点）
    for df in [df_a, df_g]:
        for col in df.columns:
            if df[col].dtype == object and col != 'ID':
                df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
    return pd.merge(df_a, df_g[['ID', 'sum_odpm']], on='ID').dropna()

data = clean_data('aggregated.csv', 'ground_truth.csv')

# 2. 机制建模参数 (Unit: uW, Frequency: GHz)
P_BASE = 490701
C_BIG, C_MID, C_LIT = 398918.9, 47064.8, 249848.5
W_R, W_G, W_B, GAMMA = 585.6, 588.8, 1265.5, 3423.4
C_GPU = 941247.6
P_FLOOR = 935848.0  # 物理钳位值 (Minimum Power Consumption)

# 3. 预测值计算 (基于物理功耗模型)
f_big, f_mid, f_lit = [data[f'CPU_{c}_FREQ_KHz'] / 1e6 for c in ['BIG', 'MID', 'LITTLE']]
f_gpu = data['GPU_1FREQ'] / 1e6
R, G, B, L = data['RougeMesuré'], data['VertMesuré'], data['BleuMesuré'], data['Brightness']

# 基础模型：多组分线性叠加与非线性频率平方项
raw_pred = (P_BASE + C_BIG*f_big**2 + C_MID*f_mid**2 + C_LIT*f_lit**2 +
            W_R*R + W_G*G + W_B*B + GAMMA*L + C_GPU*f_gpu**2)

# 应用物理钳位：体现硬件最低能耗限制
y_pred = np.maximum(P_FLOOR, raw_pred)
y_true = data['sum_odpm']
residuals = y_true - y_pred

# 4. 专业学术可视化
plt.rcParams['font.family'] = 'serif' # 使用衬线字体更显学术
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 图1: 拟合优度对比 (Goodness-of-Fit)
axes[0].scatter(y_pred/1e3, y_true/1e3, alpha=0.6, c=MCM_PALETTE['primary'], edgecolors='w', s=45)
axes[0].plot([y_true.min()/1e3, y_true.max()/1e3], [y_true.min()/1e3, y_true.max()/1e3],
            color=MCM_PALETTE['accent_2'], linestyle='--', lw=2, label='Perfect Prediction ($y=\hat{y}$)')
axes[0].set_title('A: Predicted vs. Measured Power', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Power (mW)', fontsize=12)
axes[0].set_ylabel('Measured Power (mW)', fontsize=12)
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.5)

# 图2: 残差散点图 (Heteroscedasticity Analysis)
axes[1].scatter(y_pred/1e3, residuals/1e3, alpha=0.6, c=MCM_PALETTE['primary'], edgecolors='w', s=45)
axes[1].axhline(0, color=MCM_PALETTE['secondary'], linestyle='-', lw=1.5)
axes[1].set_title('B: Residual Analysis', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Power (mW)', fontsize=12)
axes[1].set_ylabel('Residual Error (mW)', fontsize=12)
axes[1].grid(True, linestyle=':', alpha=0.5)

# 图3: 残差分布与核密度估计 (Normality of Errors)
sns.histplot(residuals/1e3, kde=True, color=MCM_PALETTE['accent_2'],
             edgecolor=MCM_PALETTE['accent_1'], alpha=0.7, ax=axes[2])
axes[2].set_title('C: Error Distribution', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Residual Error (mW)', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

# 性能指标输出
print(f"R-squared: {r2_score(y_true, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_true, y_pred)/1e3:.2f} mW")