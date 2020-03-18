import paddle.fluid as fluid
import paddle as pd
import numpy as np

#一个线性网络
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
hidden = fluid.layers.fc(input=x, size=100, act='relu')
net = fluid.layers.fc(input=hidden, size=1, act=None)

#损失函数
#接着定义神经网络的损失函数，
# 这里同样使用了fluid.layers.data()这个接口，
# 这个可以理解为数据对应的结果，
# 上面name为x的fluid.layers.data()为属性数据。
# 这里使用了平方差损失函数(square_error_cost)，
# PaddlePaddle提供了很多的损失函数的接口，
# 比如交叉熵损失函数(cross_entropy)。
# 因为本项目是一个线性回归任务，所以我们使用的是平方差损失函数。
# 因为fluid.layers.square_error_cost()求的是一个Batch的损失值，
# 所以我们还要对他求一个平均值。

y = fluid.layers.data(name='y', shape=[1], dtype='float32')
cost = fluid.layers.square_error_cost(input=net, label=y)
avg_cost = fluid.layers.mean(cost)

# 复制一个主程序用于之后的预测
test_program = fluid.default_main_program().clone(for_test=True)

#定义优化方法
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01)
opts = optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

# 定义训练和测试数据
x_data = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).astype('float32')
y_data = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]]).astype('float32')
test_data = np.array([[6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).astype('float32')

for pass_id in range(100):
    train_cost = exe.run(program=fluid.default_main_program(),
                         feed={'x': x_data, 'y': y_data},
                         fetch_list=[avg_cost])
    print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0]))


result = exe.run(program=test_program,
                 feed={'x': test_data, 'y': np.array([[0.0]]).astype('float32')},
                 fetch_list=[net])
print("当x为6.0时，y为：%0.5f" % result[0][0][0])