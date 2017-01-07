'''
线性回归
'''
# -*- encoding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 准备一些数据
plt.ion() #打开交互
n_observations = 100
fig, ax = plt.subplots(1, 1)
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5,0.5,n_observations)
ax.scatter(xs,ys)
fig.show()
plt.draw()
