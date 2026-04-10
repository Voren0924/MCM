import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格 (学术风)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用通用字体
plt.rcParams['axes.unicode_minus'] = False


def plot_wlan_analysis():
    print("正在加载并清洗数据...")
    try:
        # 1. 加载数据
        agg = pd.read_csv('aggregated.csv')
        gt = pd.read_csv('ground_truth.csv')

        # 2. 清洗数值列
        def clean_numeric(df):
            for col in df.columns:
                if df[col].dtype == 'object' and col != 'ID':
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '.').replace('err', np.nan),
                        errors='coerce'
                    )
            return df

        agg = clean_numeric(agg)
        gt = clean_numeric(gt)
        df = pd.merge(agg, gt, on='ID', suffixes=('', '_gt'))

        # 提取 WLAN 功耗 (uW -> mW) 和 数据量 (Bytes -> MB)
        wlan_power_mw = df['WLANBT_ENERGY_AVG_UWS'] / 1000
        data_mb = df['TOTAL_DATA_WIFI_BYTES'] / (1024 * 1024)

        # 过滤无效数据
        mask = wlan_power_mw.notna() & data_mb.notna()
        wlan_power_mw = wlan_power_mw[mask]
        data_mb = data_mb[mask]

        print(f"有效数据点: {len(wlan_power_mw)} 个")

    except FileNotFoundError:
        print("错误：找不到 'aggregated.csv' 或 'ground_truth.csv'，请检查路径。")
        return

    # ==========================================
    # 图 1: WLAN 功耗分布直方图 (Distribution)
    # ==========================================
    plt.figure(figsize=(10, 6))

    # 绘制直方图和核密度估计 (KDE)
    sns.histplot(wlan_power_mw, bins=40, kde=True, color='#4A90E2', edgecolor='black', alpha=0.6)

    # 标注统计量
    mean_val = wlan_power_mw.mean()
    median_val = wlan_power_mw.median()

    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f} mW')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.1f} mW')

    plt.title('Distribution of WLAN Power Consumption', fontsize=14, fontweight='bold')
    plt.xlabel('Power Consumption (mW)', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('WLAN_Distribution.png', dpi=300)
    print("已生成图表: WLAN_Distribution.png")

    # ==========================================
    # 图 2: WLAN 功耗 vs. 数据吞吐量 (State-Based Model)
    # ==========================================
    plt.figure(figsize=(10, 6))

    # 设定阈值 (例如 25% 分位数)
    threshold_mb = data_mb.quantile(0.25)

    # 分组计算均值
    idle_mean = wlan_power_mw[data_mb < threshold_mb].mean()
    active_mean = wlan_power_mw[data_mb >= threshold_mb].mean()

    # 绘制散点
    plt.scatter(data_mb, wlan_power_mw, alpha=0.4, s=20, color='#888888', label='Data Points')

    # 绘制阈值分割线
    plt.axvline(x=threshold_mb, color='red', linestyle='--', linewidth=1.5,
                label=f'Threshold ({threshold_mb:.2f} MB)')

    # 绘制状态均值线 (State Levels)
    # State 0: Low Data
    plt.hlines(y=idle_mean, xmin=0, xmax=threshold_mb, colors='green', linewidth=3,
               label=f'Idle State (~{idle_mean:.0f} mW)')
    # State 1: Active Data
    plt.hlines(y=active_mean, xmin=threshold_mb, xmax=data_mb.max(), colors='orange', linewidth=3,
               label=f'Active State (~{active_mean:.0f} mW)')

    plt.title('WLAN Power State Model: Constant Base vs. Throughput', fontsize=14, fontweight='bold')
    plt.xlabel('Data Throughput (MB)', fontsize=12)
    plt.ylabel('Power Consumption (mW)', fontsize=12)

    # 添加注释
    plt.text(threshold_mb * 1.1, active_mean + 10, f'+{(active_mean - idle_mean) / idle_mean:.1%} Increase',
             color='orange', fontweight='bold')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('WLAN_State_Model.png', dpi=300)
    print("已生成图表: WLAN_State_Model.png")
    plt.show()

if __name__ == "__main__":
    plot_wlan_analysis()