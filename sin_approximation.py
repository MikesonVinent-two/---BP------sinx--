import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math
from tqdm import tqdm
import pickle
import os
from datetime import datetime

# 生成训练数据
def generate_data(n_samples):
    x = np.random.uniform(-np.pi, np.pi, n_samples)
    y = np.sin(x)
    return x, y

# 定义神经网络类
class NeuralNetwork:
    def __init__(self, input_size, hidden_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros((1, 1))

    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2

    def backward(self, x, y, output, learning_rate):
        m = x.shape[0]
        
        # 输出层误差
        delta2 = (output - y)
        
        # 隐藏层误差
        delta1 = np.dot(delta2, self.w2.T) * self.tanh_derivative(self.z1)
        
        # 更新权重和偏置
        self.w2 -= learning_rate * np.dot(self.a1.T, delta2) / m
        self.b2 -= learning_rate * np.sum(delta2, axis=0, keepdims=True) / m
        self.w1 -= learning_rate * np.dot(x.T, delta1) / m
        self.b1 -= learning_rate * np.sum(delta1, axis=0, keepdims=True) / m

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

# 训练函数
def train_network():
    # 设置超参数
    n_samples = 1000
    hidden_size = 512
    epochs = 10000
    learning_rate = 0.1
    
    # 创建保存文件的目录
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 生成文件名的参数字符串
    params_str = f"samples{n_samples}_hidden{hidden_size}_lr{learning_rate}_epochs{epochs}"
    
    # 生成训练数据
    X_train, y_train = generate_data(n_samples)
    X_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    # 初始化神经网络
    nn = NeuralNetwork(1, hidden_size)

    # 训练循环
    for epoch in tqdm(range(epochs)):
        # 前向传播
        output = nn.forward(X_train)
        
        # 反向传播
        nn.backward(X_train, y_train, output, learning_rate)
        
        # 计算误差
        if epoch % 100 == 0:
            error = np.mean(np.abs(output - y_train))
            if error < 0.01:
                print(f"\nReached target error at epoch {epoch}")
                break

    # 保存模型
    model_filename = f"models/sin_model_{params_str}.pkl"
    nn.save_model(model_filename)
    print(f"模型已保存至: {model_filename}")
    
    return nn, params_str

# 可视化结果
def plot_results(nn, params_str):
    # 生成测试数据
    X_test = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
    y_pred = nn.forward(X_test)
    y_true = np.sin(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_true, label='True sin(x)', color='blue')
    plt.plot(X_test, y_pred, label='Neural Network', color='red', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.title('神经网络拟合正弦函数')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # 保存图像
    plt.savefig(f"plots/sin_plot_{params_str}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算误差
    error = np.mean(np.abs(y_pred - y_true))
    print(f'平均误差: {error:.4f}')
    
    # 保存训练结果数据
    results = {
        'error': error,
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'parameters': params_str
    }
    
    with open(f"models/training_results_{params_str}.txt", 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    # 训练网络
    trained_nn, params_str = train_network()
    
    # 显示结果
    plot_results(trained_nn, params_str) 