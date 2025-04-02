import numpy as np



# 训练参数配置
TRAINING_CONFIG = {
    'n_samples': 1000,      # 训练样本数量
    'hidden_size': 128,     # 隐藏层大小
    'epochs': 10000,        # 训练轮数
    'learning_rate': 0.2,   # 学习率
}

# 数据生成参数
DATA_CONFIG = {
    'x_range': (-np.pi, np.pi),  # 训练数据x的范围
    'test_points': 200,          # 测试数据点数量
    'test_range': (-np.pi,np.pi),  # 测试数据x的范围
}

# 文件路径配置

PATH_CONFIG = {
    'model_dir': 'models',       # 模型保存目录
    'plot_dir': 'plots',         # 图表保存目录
    'model_prefix': 'sin_model', # 模型文件名前缀
}

# 可视化配置
PLOT_CONFIG = {
    'figsize': (10, 6),         # 图表大小
    'true_color': 'blue',       # 真实值线条颜色
    'pred_color': 'red',        # 预测值线条颜色
    'pred_style': '--',         # 预测值线条样式
    'dpi': 300,                 # 图片分辨率
} 