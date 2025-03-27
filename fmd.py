from typing import Tuple

import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy.io import loadmat

from scipy.signal import correlate, correlation_lags, hilbert, lfilter, firwin, get_window

epsilon = 2.2204e-16


def FMD(fs: float,
        x: np.ndarray,
        filter_size: int,
        cut_num: int,
        mode_num: int,
        max_iter_num: int) -> np.ndarray:
    """特征模式分解 (Feature Mode Decomposition)

    参数:
        fs: 采样频率 (Hz)
        x: 输入信号 (1D数组)
        FilterSize: 滤波器长度
        CutNum: 频带分割数
        ModeNum: 目标模式数
        MaxIterNum: 最大迭代次数

    返回:
        Final_Mode: 分解后的模式矩阵 (每列为一个模式)
    """
    # 输入信号处理
    x = np.asarray(x).flatten()
    N = len(x)

    # 初始化频率边界
    freq_bound = np.linspace(0, 1 - 1 / cut_num, cut_num)

    # 生成初始滤波器组
    temp_filters = np.zeros((filter_size, cut_num))
    for n in range(cut_num):
        # window_array = get_window('hann', filter_size, fftbins=False)
        # window_array = matlab_hanning(filter_size)
        # 使用 firwin 生成滤波器系数，numtaps 设置为 filter_size
        cutoff = [freq_bound[n] + epsilon, freq_bound[n] + 1 / cut_num - epsilon]
        temp_filters[:, n] = firwin(
            filter_size,
            cutoff,
            window='hann',
            # window=tuple(window_array.tolist()),
            pass_zero=False
        )

    # 初始化结果存储结构
    result: list[dict] = [{} for _ in range(cut_num + 1)]
    result[0] = {
        'IterCount': None,
        'Iterations': None,
        'CorrMatrix': None,
        'ComparedModeNum': None,
        'StopNum': None
    }

    # 初始化临时信号
    temp_sig = np.tile(x, (cut_num, 1)).T

    itercount = 1
    while True:
        # 确定迭代次数
        iternum = max_iter_num - (cut_num - mode_num) * 2 if itercount == 1 else 2

        current_iter: dict[str, list] = {
            'Iterations': [[] for _ in range(cut_num)],
            'CorrMatrix': None,
            'ComparedModeNum': None,
            'StopNum': None
        }

        # 对每个频带执行MCKD
        for n in range(cut_num):
            f_init = temp_filters[:, n].copy()
            y_Iter, f_Iter, k_Iter, T_Iter = xxc_mckd(
                fs=fs,
                x=temp_sig[:, n],
                f_init=f_init,
                termIter=iternum,
                T=None,
                M=1,
                plotMode=False
            )

            # 存储迭代结果
            current_iter['Iterations'][n] = {
                'y': y_Iter[:, -1],
                'f': f_Iter[:, -1],
                'k': k_Iter[-1],
                'fft': np.abs(np.fft.fft(f_Iter, axis=0))[:filter_size // 2, :],
                'freq': (np.argmax(np.abs(np.fft.fft(f_Iter, axis=0))[:filter_size // 2, :], axis=0) - 1) * (
                        fs / filter_size),
                'T': T_Iter
            }

        # 更新信号和滤波器
        for n in range(cut_num):
            temp_sig[:, n] = current_iter['Iterations'][n]['y']
            temp_filters[:, n] = current_iter['Iterations'][n]['f']

        # 计算相关系数矩阵
        corr_matrix = np.abs(np.corrcoef(temp_sig, rowvar=False))
        corr_matrix = np.triu(corr_matrix, k=1)

        # 寻找最相关模式对
        I, J = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)

        # 计算CK值
        XI = temp_sig[:, I] - np.mean(temp_sig[:, I])
        XJ = temp_sig[:, J] - np.mean(temp_sig[:, J])
        T_I = current_iter['Iterations'][I]['T']
        T_J = current_iter['Iterations'][J]['T']
        KI = CK(XI, T_I, M=1)
        KJ = CK(XJ, T_J, M=1)

        # 确定删除的索引
        delete_idx = J if KI > KJ else I

        # 删除模式
        temp_sig = np.delete(temp_sig, delete_idx, axis=1)
        temp_filters = np.delete(temp_filters, delete_idx, axis=1)
        cut_num -= 1

        # 存储迭代信息
        result[itercount] = {
            'IterCount': itercount,
            'Iterations': current_iter['Iterations'],
            'CorrMatrix': corr_matrix,
            'ComparedModeNum': (I, J),
            'StopNum': delete_idx
        }

        # 终止条件
        if temp_sig.shape[1] == mode_num - 1:
            break

        itercount += 1

    # 提取最终模式
    Final_Mode = np.zeros((N, mode_num))
    for nn in range(mode_num):
        Final_Mode[:, nn] = result[itercount]['Iterations'][nn]['y']

    return Final_Mode


def xxc_mckd(fs: float,
             x: np.ndarray,
             f_init: np.ndarray,
             termIter: int = 30,
             T: int = None,
             M: int = 3,
             plotMode: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """改进的最大相关峰度解卷积 (MCKD) 算法

    参数:
        fs: 采样频率 (Hz)
        x: 输入信号 (1D数组)
        f_init: 初始滤波器系数 (1D数组)
        termIter: 最大迭代次数 (默认30)
        T: 初始周期 (样本数)，自动计算时设为None
        M: 移位阶数 (默认3)
        plotMode: 保留参数 (未实现)

    返回:
        y_final: 各迭代输出信号 (2D数组，每列为一次迭代)
        f_final: 各迭代滤波器系数 (2D数组，每列为一次迭代)
        ckIter: 各迭代相关峰度值 (1D数组)
        T_final: 最终估计周期
    """
    # 输入校验
    x = np.asarray(x).flatten()
    L = len(f_init)
    N = len(x)
    f = f_init.reshape(-1, 1)  # 转换为列向量

    # 自动计算初始周期
    if T is None:
        x_env = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
        T = TT(x_env, fs)
    T = int(round(T))
    T_final = T

    # 预分配三维矩阵 (优化内存布局)
    XmT = np.zeros((L, N, M + 1), order='F')

    # 初始化输出存储
    y_final = np.zeros((N, termIter))
    f_final = np.zeros((L, termIter))
    ckIter = np.zeros(termIter)

    # 主迭代循环
    for n in range(termIter):
        # 更新XmT矩阵 (向量化加速)
        for m in range(M + 1):
            shift = m * T
            # 处理第一行
            XmT[0, shift:, m] = x[:N - shift] if shift < N else 0
            # 滚动生成后续行
            for l_i in range(1, L):
                XmT[l_i, 1:, m] = XmT[l_i - 1, :-1, m]

        # 矩阵逆计算 (添加正则化)
        XTX = XmT[:, :, 0] @ XmT[:, :, 0].T
        Xinv = np.linalg.pinv(XTX + 1e-8 * np.eye(L))

        # 计算输出信号
        y = (f.T @ XmT[:, :, 0]).T

        # 生成时移信号矩阵yt
        yt = np.zeros((N, M + 1))
        yt[:, 0] = y.flatten()
        for m in range(1, M + 1):
            yt[m * T:, m] = y[:-m * T].flatten() if m * T < N else 0

        # 计算alpha矩阵 (向量化实现)
        alpha = np.zeros((N, M + 1))
        for m in range(M + 1):
            cols = [k for k in range(M + 1) if k != m]
            alpha[:, m] = np.prod(yt[:, cols], axis=1) ** 2 * yt[:, m]

        # 计算beta
        beta = np.prod(yt, axis=1)

        # 更新滤波器系数
        Xalpha = np.sum([XmT[:, :, m] @ alpha[:, m] for m in range(M + 1)], axis=0)
        numerator = np.sum(y ** 2)
        denominator = 2 * np.sum(beta ** 2) + 1e-12
        f = (numerator / denominator) * (Xinv @ Xalpha)
        f /= np.linalg.norm(f)  # 归一化

        # 记录当前迭代结果
        ckIter[n] = np.sum(beta ** 2) / (np.sum(y ** 2) ** (M + 1) + 1e-12)
        f_final[:, n] = f.flatten()
        y_final[:, n] = lfilter(f.flatten(), [1.0], x)

        # 动态更新周期估计
        if n % 5 == 4:
            y_env = np.abs(hilbert(y)) - np.mean(np.abs(hilbert(y)))
            T_new = TT(y_env, fs)
            if 10 < T_new < N // 2:
                T = int(round(T_new))
                T_final = T

    return y_final, f_final, ckIter, T_final


def TT(y, fs):
    """通过自相关函数估计信号的周期 (样本数)

    参数:
        y : 输入信号 (1D数组)
        fs : 采样频率 (用于确定最大滞后量)

    返回:
        T : 估计的周期 (样本数)
    """
    y = np.asarray(y).flatten()
    if len(y) < 2:
        return 0  # 无效输入处理

    # 计算最大滞后量（不超过信号长度）
    max_lag = min(fs, len(y) - 1)

    # 计算归一化自相关
    acf = correlate(y, y, mode='full')
    norm = np.dot(y, y)
    acf_normalized = acf / norm if norm != 0 else acf

    # 获取滞后值数组
    lags = correlation_lags(len(y), len(y), mode='full')

    # 提取非负滞后部分 (0到max_lag)
    mask = (lags >= 0) & (lags <= max_lag)
    valid_lags = lags[mask]
    valid_acf = acf_normalized[mask]

    # 寻找第一个过零点
    zero_cross = None
    for i in range(1, len(valid_acf)):
        prev = valid_acf[i - 1]
        curr = valid_acf[i]
        if (prev > 0 > curr) or (prev == 0 or curr == 0):
            zero_cross = i
            break

    if zero_cross is None:
        return 0

    # 提取过零点后的自相关段
    final_lags = valid_lags[zero_cross:]
    final_acf = valid_acf[zero_cross:]

    # 找到最大峰值的滞后值
    peak_pos = np.argmax(final_acf)
    T = final_lags[peak_pos]

    return int(T)


def CK(x, T, M=2):
    """计算相关峰度 (Correlated Kurtosis)

    参数:
        x : 输入信号 (1D数组)
        T : 时移周期
        M : 阶数 (默认=2)

    返回:
        ck : 相关峰度值
    """
    # 确保x是行向量 (转换为二维数组，形状为 (1, N))
    x = np.array(x).flatten()
    N = len(x)

    # 初始化时移矩阵 (M+1行, N列)
    x_shift = np.zeros((M + 1, N))
    x_shift[0, :] = x

    # 逐行填充时移信号
    for m in range(1, M + 1):
        # 从第T位置开始填充左移后的信号 (MATLAB索引转换为Python索引)
        x_shift[m, T:] = x_shift[m - 1, :-T]

    # 计算分子：时移信号逐点乘积的平方和
    numerator = np.sum(np.prod(x_shift, axis=0) ** 2)

    # 计算分母：原始信号能量的(M+1)次方
    denominator = np.sum(x ** 2) ** (M + 1)

    return numerator / denominator


def max_ij(X):
    """返回矩阵X中最大值的行索引、列索引和最大值"""
    # 按列找最大值及其行索引
    col_max_values = np.max(X, axis=0)
    row_indices = np.argmax(X, axis=0)

    # 找全局最大值及其列索引
    max_value = np.max(col_max_values)
    j = np.argmax(col_max_values)

    # 获取对应行索引
    i = row_indices[j]

    return i, j, max_value


def matlab_hanning(filter_size: int) -> np.ndarray:
    """
    生成与 MATLAB 完全一致的汉宁窗（首尾为 0）

    参数:
        filter_size (int): 窗口长度

    返回:
        np.ndarray: 窗口数组
    """
    # 生成从 0 到 N-1 的索引
    n = np.arange(filter_size)

    # MATLAB 的 hanning 公式
    window = 0.5 * (1 - np.cos(2 * np.pi * n / (filter_size)))
    #
    # # 强制首尾为 0 (MATLAB 的特殊处理)
    # window[0] = 0.0
    # window[-1] = 0.0

    return window


def main():
    np.random.seed(42)
    data = pandas.read_excel(r"data/原始信号.xlsx", header=None)

    fs = len(data)
    filter_size = 30
    cut_num = 7
    mod_num = 5
    max_iter_num = 20

    imfs = FMD(fs, data.values, filter_size, cut_num, mod_num, max_iter_num)

    matlab_data = loadmat(r"data/matlab_result.mat")
    matlab_modes = matlab_data['u']

    plt.figure(figsize=(12, 6))
    for i in range(imfs.shape[1]):
        plt.subplot(imfs.shape[1], 1, i + 1)
        plt.plot(matlab_modes[:, i], 'r--', label='MATLAB')
        plt.plot(imfs[:, i], 'b-', alpha=0.5, label='Python')
        plt.title(f'Mode {i + 1}')
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
