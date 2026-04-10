import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 调色盘与学术配置
MCM_PALETTE = ['#2878B5', '#9AC9DB', '#F8AC8C', '#C82423', '#FF8884']
plt.rcParams['font.family'] = 'serif'

# 2. 数据处理：涵盖所有核心组件
df_gt = pd.read_csv('ground_truth.csv')
comp_map = {
    'CPU_BIG_ENERGY_AVG_UWS': 'CPU Big',
    'CPU_MID_ENERGY_AVG_UWS': 'CPU Mid',
    'CPU_LITTLE_ENERGY_AVG_UWS': 'CPU Little',
    'GPU_ENERGY_AVG_UWS': 'GPU',
    'Display_ENERGY_AVG_UWS': 'Display',
    'WLANBT_ENERGY_AVG_UWS': 'WLAN',
    'Camera_ENERGY_AVG_UWS': 'Camera',
    'Sensor_ENERGY_AVG_UWS': 'Sensor',
    'GPS_ENERGY_AVG_UWS': 'GPS'
}
total_col = 'sum_odpm'

# 清洗数据：处理欧洲格式的逗号小数点并转换为数值
all_needed = list(comp_map.keys()) + [total_col]
for col in all_needed:
    if df_gt[col].dtype == object:
        df_gt[col] = pd.to_numeric(df_gt[col].str.replace(',', '.'), errors='coerce')

df_clean = df_gt[all_needed].dropna()

# 3. 计算 VCR (方差贡献率) 和 MCR (均值贡献率)
var_total = df_clean[total_col].var()
mean_total = df_clean[total_col].mean()

results = []
for col, name in comp_map.items():
    vcr = (df_clean[col].var() / var_total) * 100
    mcr = (df_clean[col].mean() / mean_total) * 100
    results.append({'Component': name, 'VCR': vcr, 'MCR': mcr})

# 按方差贡献排序
vcr_df = pd.DataFrame(results).sort_values(by='VCR', ascending=False)
vcr_df['Cumulative_VCR'] = vcr_df['VCR'].cumsum()

# 4. 帕累托可视化
fig, ax1 = plt.subplots(figsize=(13, 7))

# 柱状图：展示各组件方差贡献
bars = ax1.bar(vcr_df['Component'], vcr_df['VCR'], color=MCM_PALETTE[0], alpha=0.85, label='Individual VCR (%)')
ax1.set_ylabel('Variance Contribution Ratio (VCR %)', fontsize=12, fontweight='bold', color=MCM_PALETTE[0])
ax1.set_ylim(0, vcr_df['VCR'].max() * 1.25)
plt.xticks(rotation=30, ha='right', fontsize=11)

# 折线图：展示累计方差贡献
ax2 = ax1.twinx()
ax2.plot(vcr_df['Component'], vcr_df['Cumulative_VCR'], color=MCM_PALETTE[3], marker='o', ms=8, lw=2.5, label='Cumulative VCR')
ax2.set_ylabel('Cumulative VCR (%)', fontsize=12, fontweight='bold', color=MCM_PALETTE[3])
ax2.set_ylim(0, 105) # 累计最高100%

# 数值标注 (只标注 VCR > 0.1% 的核心组件，保持图面整洁)
for i, bar in enumerate(bars):
    v = vcr_df.iloc[i]['VCR']
    m = vcr_df.iloc[i]['MCR']
    if v > 0.05:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'VCR:{v:.1f}%\n(MCR:{m:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.title('Global Pareto Analysis',
          fontsize=16, fontweight='bold', pad=30)
ax1.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

# 输出统计表用于论文附录
print(vcr_df[['Component', 'VCR', 'MCR']].to_string(index=False))