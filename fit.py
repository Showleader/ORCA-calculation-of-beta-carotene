import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# 定义文件路径
data_file = r'G:\BaiduNetdiskDownload\cityudessertation\PHY6505\codeandresult\distribution_data.txt'
output_file = r'G:\BaiduNetdiskDownload\cityudessertation\PHY6505\codeandresult\fit.txt'
plot_file = r'G:\BaiduNetdiskDownload\cityudessertation\PHY6505\codeandresult\fit.png'

# 加载数据，假设数据为两列，中间用空白分隔
data = np.loadtxt(data_file)
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]

# 找到 y 值最大的点对应的 X 坐标，作为分界线
x_divide = X[np.argmax(y), 0]
print("分界线 x 坐标为:", x_divide)

# 定义分段函数，将输入 x 与分界线进行比较
def if_then_else(x, output1, output2):
    # 如果 x 小于分界线，返回 output1，否则返回 output2
    return np.where(x < x_divide, output1, output2)

# 用 gplearn 包装自定义的分段函数
if_then_else_function = make_function(function=if_then_else, name='if_then_else', arity=3)

# 定义函数集合，此处包含基本运算和自定义的分段函数
function_set = ['add', 'sub', 'mul', 'div', if_then_else_function]

# 设置符号回归器参数
est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           function_set=function_set,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

# 拟合模型
est_gp.fit(X, y)

# 导出拟合得到的方程
equation = est_gp._program

# 将拟合方程写入文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("Fitted Equation using gplearn SymbolicRegressor:\n")
    f.write(str(equation))
    
print("Fitted equation has been written to:", output_file)

# 生成用于绘图的光滑 X 值
X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_pred = est_gp.predict(X_plot)

# 绘制原始数据点和拟合曲线
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Original Data', alpha=0.5)
plt.plot(X_plot, y_pred, color='red', linewidth=2, label='Fitted Curve')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Fitting using gplearn SymbolicRegressor with Piecewise Function")
plt.legend()
plt.grid(True)

# 保存图像到文件
plt.savefig(plot_file, dpi=300)
print("Fitted plot has been saved to:", plot_file)
plt.show()