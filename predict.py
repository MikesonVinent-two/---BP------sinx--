import numpy as np
import matplotlib.pyplot as plt
from sin_approximation import NeuralNetwork
from config import PATH_CONFIG, DATA_CONFIG, PLOT_CONFIG
import argparse
import os

def load_and_predict(model_path):
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在")
        return

    # 加载模型
    try:
        nn = NeuralNetwork.load_model(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 生成测试数据
    X_test = np.linspace(DATA_CONFIG['test_range'][0], DATA_CONFIG['test_range'][1], DATA_CONFIG['test_points']).reshape(-1, 1)
    y_pred = nn.forward(X_test)
    y_true = np.sin(X_test)

    # 绘制结果
    plt.figure(figsize=PLOT_CONFIG['figsize'])
    plt.plot(X_test, y_true, label='真实sin(x)', color=PLOT_CONFIG['true_color'])
    plt.plot(X_test, y_pred, label='神经网络预测', color=PLOT_CONFIG['pred_color'], linestyle=PLOT_CONFIG['pred_style'])
    plt.legend()
    plt.grid(True)
    plt.title('神经网络预测结果')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # 计算误差
    error = np.mean(np.abs(y_pred - y_true))
    print(f'平均误差: {error:.4f}')

    # 进行单点预测
    while True:
        try:
            x = float(input("\n请输入要预测的x值（输入q退出）: "))
            if x.lower() == 'q':
                break
            x = np.array([[x]])
            y_pred = nn.forward(x)
            y_true = np.sin(x)
            print(f"x = {x[0][0]:.4f}")
            print(f"预测值: {y_pred[0][0]:.4f}")
            print(f"真实值: {y_true[0][0]:.4f}")
            print(f"误差: {abs(y_pred[0][0] - y_true[0][0]):.4f}")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            break

def list_available_models():
    """列出models目录下所有可用的模型文件"""
    model_dir = PATH_CONFIG['model_dir']
    if not os.path.exists(model_dir):
        print(f"错误：模型目录 {model_dir} 不存在")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not model_files:
        print(f"在 {model_dir} 目录下没有找到模型文件")
        return
    
    print("\n可用的模型文件：")
    for i, file in enumerate(model_files, 1):
        print(f"{i}. {file}")
    return model_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用训练好的神经网络模型预测正弦函数')
    parser.add_argument('--model', type=str, help='模型文件的完整路径')
    parser.add_argument('--list', action='store_true', help='列出所有可用的模型文件')
    args = parser.parse_args()

    if args.list:
        list_available_models()
    elif args.model:
        load_and_predict(args.model)
    else:
        # 如果没有指定参数，列出可用模型并让用户选择
        model_files = list_available_models()
        if model_files:
            try:
                choice = int(input("\n请选择要使用的模型文件编号（输入0退出）: "))
                if 1 <= choice <= len(model_files):
                    model_path = os.path.join(PATH_CONFIG['model_dir'], model_files[choice-1])
                    load_and_predict(model_path)
                elif choice != 0:
                    print("无效的选择")
            except ValueError:
                print("请输入有效的数字") 