import numpy as np
import math
import matplotlib.pyplot as plt


def fitness_function_1(x):
    dim = len(x)
    y = 0
    for i in range(0, dim):
        y += x[i]**2
    return y


def fitness_function_13(x):
    dim = len(x)
    conti_addition = 0
    for i in range(1, dim+1):
        conti_addition += x[i-1]**2

    conti_multiple = 1
    for i in range(1, dim+1):
        conti_multiple *= math.cos(x[i-1] / math.sqrt(i))
    return conti_addition / 4000 - conti_multiple + 1


def fitness_function_14(x):
    dim = len(x)
    continuous_addition = 0
    for i in range(dim):
        continuous_addition += (x[i]**2 - 10 * math.cos(2 * math.pi * x[i]) + 10)
    return continuous_addition


def fitness_function_21(x):
    dim = len(x)
    continuous_addition = 0
    for i in range(dim):
        continuous_addition += -1 * x[i] * math.sin(math.sqrt(abs(x[i])))
    return continuous_addition


def paint_image_2d():
    test_x = np.linspace(-search_range_num, search_range_num, num_sample, dtype=float)
    test_x = test_x.reshape((-1, 1))
    test_y = []

    for i in range(num_sample):
        test_y.append(fitness_function(test_x[i]))
    # 设置图像标题
    plt.title('三维曲面图')
    plt.plot(test_x, test_y)
    # 显示图像
    plt.show()
    print(test_y)


def paint_image_3d():
    test_x = np.linspace(-search_range_num, search_range_num, num_sample, dtype=float)
    test_y = np.linspace(-search_range_num, search_range_num, num_sample, dtype=float)
    test_x, test_y = np.meshgrid(test_x, test_y)

    test_z = np.ndarray(shape=(num_sample, num_sample), dtype=float)
    for i in range(num_sample):
        for j in range(num_sample):
            test_z[i][j] = fitness_function(np.array([test_x[i][j], test_y[i][j]]))

    # 创建一个三维坐标轴对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 使用add_subplot创建3D坐标轴

    # 绘制三维曲面图
    surf = ax.plot_surface(test_x, test_y, test_z, cmap='viridis')

    # 添加颜色条
    fig.colorbar(surf)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置图像标题
    plt.title('')

    # 显示图像
    plt.show()


def fitness_function(x):
    return fitness_function_21(x)


search_range_num = 500
num_sample = 100

Dn = 30
population_size = 90
group_size = 9
max_iterations = 30

# paint_image_3d()

