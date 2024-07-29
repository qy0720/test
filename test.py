import numpy as np
import torch.nn as nn
from math import sin, cos, pi
import torch
import os
import scipy.io as sio
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
seq_time = 4
class lstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #out, _ = self.gru(x)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float()
        out, _ = self.lstm(x[:, :seq_time, :])
        out = self.linear(out[:, -1, :])
        return out
#数据归一化
def guiyihua(Yn,X_max,X_min):
    Yn_1 = (Yn - X_min) / (X_max - X_min)
    return Yn_1
#数据反归一化
def fanguiyihua(Yn, X_max, X_min):
    Yn_11 = Yn * (X_max - X_min) + X_min
    return Yn_11
def runge_kutta_step(f, xn, Yn, h):
    K1 = f(xn, Yn)
    K2 = f(xn + h / 2.0, Yn + K1 * h / 2.0)
    K3 = f(xn + h / 2.0, Yn + K2 * h / 2.0)
    K4 = f(xn + h, Yn + K3 * h)

    Yn1 = Yn + (K1 + 2 * K2 + 2 * K3 + K4) * h / 6.0
    return Yn1


# 例如二元方程组
def f(xn, Yn):  #在这个微分方程中，x为控制量，y为状态量
    y1, y2, y3, y4, y5, y6 = Yn

    x1, x2, x3 = xn
    DD = 0.5 * 1.225 * (1-0.00688*(y3*3.2808/1000))**4.256*(y4**2)*27.871*(0.0476-0.1462*x2+0.0491*(x2**2)+12.8046*(x2**3)-12.6985*(x2**4))

    LL = 0.5 * 1.225 * (1-0.00688*(y3*3.2808/1000))**4.256*(y4**2)*27.871*(0.0174+4.3329*x2-1.3048*(x2**2)+2.2442*(x2**3)-5.8517*(x2**4))

    f1 = y4 * cos(y6) *cos(y5)
    f2 = y4 * cos(y6) *sin(y5)
    f3 = y4 * sin(y6)
    f4 = (91130 * x1 * cos(x2)-DD)/9299 - 9.8 * sin(y6)
    f5 = (91130 * x1 * sin(x2)+LL) * sin(x3)/9299/y4/cos(y6)
    f6 = (91130 * x1 *sin(x2)+LL) * cos(x3)/9299/y4 - 9.8 * cos(y6)/y4

    return np.array([f1, f2, f3, f4, f5, f6])


# 主程序
#加载训练好的模型
model = torch.load('D:\\university\\models\\lastlstm1.pth')
model.to('cpu')
folder_path = "/mnt/bn/qy-dcar-valume/ZGC/data"
X_max = np.array([3200, 2400, 640.0266, 213.5, 0.98149, 0.40458])
X_min = np.array([-3959.99487, -6359.99968, 200, 122.10758, -0.00001, -0.21532])

for n in range(338, 401): # 设置要处理的编号范围
    file_name = f"traj{n}.mat"
    print(file_name)
    file_path = os.path.join(folder_path, file_name)
    data0 = sio.loadmat(file_path)
    # 处理data......
    data0 = data0['trajectory']
    x0 = data0[:,0]
    y0 = data0[:,1]
    z0 = data0[:,2]
    print(x0.shape)
    h = 0.001
    T = 0

    state1 = data0[0:4, 0:6]
    con1 = data0[0:4, 6:9]
    Yn = np.empty((1, 4, 6))
    
    Yn[0,0,:] = data0[0,0:6]

    Yn[0,1,:] = data0[1,0:6]
    Yn[0,2,:] = data0[2,0:6]
    Yn[0,3,:] = data0[3,0:6]
    print(Yn.shape)
    for i in range(75000):
        # T = 0
        Yn = guiyihua(Yn,X_max,X_min)
        xn = model(Yn)
        xn = xn[0]
        xn = xn.detach().numpy()
        Yn = fanguiyihua(Yn,X_max,X_min)
        Yn1 = runge_kutta_step(f, xn, Yn[-1,-1,:], h)
        
        state1 = np.append(state1, [Yn1], axis=0)
        con1 = np.append(con1, [xn], axis=0)
        YY = Yn[:, 1:, :]
       
        Yn[0,0:3,:] = YY
        Yn[0,3,:] = Yn1.reshape(1, 6)
        T = T+h

    print(state1.shape, con1.shape)
    trajectory1 = np.concatenate((state1, con1), axis=1)
    save_pathnpymat = 'D:\\university\\paper1\\netlstmstatecon1'
    filenamenpy = f"trajnet{n}.npy"
    full_pathnpy = save_pathnpymat + '/' + filenamenpy
    np.save(full_pathnpy, trajectory1)
    filenamemat = f"trajnet{n}.mat"
    full_pathmat = save_pathnpymat + '/' + filenamemat
    savemat(full_pathmat, {'trajectory': trajectory1})

    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot(state1[:,0], state1[:,1], state1[:,2], label='line 1')
    ax.plot(x0, y0, z0, label='line 2')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    save_path = 'D:\\university\\paper1\\figlstm1'
    figfilename = f"traj{n}.png"
    full_path = save_path + '/' + figfilename
    plt.savefig(full_path)
    # plt.show()
    plt.close()