import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
from sklearn.metrics import r2_score

# 1. 定义数据清洗函数 (处理 CSV 中的逗号小数位)
def clean_numeric(series):
    if series.dtype == object:
        return pd.to_numeric(series.str.replace(',', '.'), errors='coerce')
    return pd.to_numeric(series, errors='coerce')

# 2. 加载数据集
# 假设文件名分别为 aggregated.csv 和 ground_truth.csv
df_agg = pd.read_csv('aggregated.csv')
df_gt = pd.read_csv('ground_truth.csv')

# 3. 合并数据
data = pd.merge(df_agg, df_gt[['ID', 'sum_odpm']], on='ID')

# 4. 指定特征列并清洗
freq_cols = ['CPU_BIG_FREQ_KHz', 'CPU_MID_FREQ_KHz', 'CPU_LITTLE_FREQ_KHz', 'GPU_1FREQ']
disp_cols = ['RougeMesuré', 'VertMesuré', 'BleuMesuré', 'Brightness']
target_col = 'sum_odpm'

for col in freq_cols + disp_cols + [target_col]:
    data[col] = clean_numeric(data[col])

# 删除含有缺失值的行
data = data.dropna(subset=freq_cols + disp_cols + [target_col])

# 5. 特征工程：将频率转换为 GHz 并取平方 (以捕捉非线性功耗)
data['f_big_G2'] = (data['CPU_BIG_FREQ_KHz'] / 1e6) ** 2
data['f_mid_G2'] = (data['CPU_MID_FREQ_KHz'] / 1e6) ** 2
data['f_lit_G2'] = (data['CPU_LITTLE_FREQ_KHz'] / 1e6) ** 2
data['f_gpu_G2'] = (data['GPU_1FREQ'] / 1e6) ** 2

# 6. 准备回归矩阵 X 和目标向量 y
features = ['f_big_G2', 'f_mid_G2', 'f_lit_G2',
            'RougeMesuré', 'VertMesuré', 'BleuMesuré', 'Brightness',
            'f_gpu_G2']

X = data[features].values
y = data[target_col].values

# 在 X 中加入一列全 1，用于拟合截距 P_base
X_design = np.hstack([np.ones((X.shape[0], 1)), X])

# 7. 使用受限最小二乘法求解
# 约束条件：第一个系数 (P_base) 范围为 (-inf, inf)，其余系数必须 >= 0
lower_bounds = [-np.inf] + [0.0] * len(features)
upper_bounds = [np.inf] * (len(features) + 1)

res = lsq_linear(X_design, y, bounds=(lower_bounds, upper_bounds))
coeffs = res.x

# 8. 打印结果
labels = ['P_base'] + features
print("--- 最终模型系数 ---")
for label, val in zip(labels, coeffs):
    print(f"{label}: {val:.6f}")

# 9. 计算拟合优度 R2
y_pred = X_design @ coeffs
r2 = r2_score(y, y_pred)
print(f"\nR-squared: {r2:.4f}")