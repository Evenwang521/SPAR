import numpy as np
import matplotlib.pyplot as plt

def f(d):
    return 1-np.exp(-1000*(d-0.4)**2)

def ce(t):
    return 100 * 1/(1236.5/(np.power((t-15),0.618))-72.5)

def sigmoid(x): 
    return 1/(1+ np.exp(np.abs(x-0.15)))
# 测试函数
d_values = np.linspace(0, 1, 10000)
function_values = sigmoid(d_values)
# print(function_values)

# 可视化函数
plt.plot(d_values, function_values)
plt.xlabel('d')
plt.ylabel('f(d)')
plt.title('Function f(d)')
plt.grid(True)
plt.show()