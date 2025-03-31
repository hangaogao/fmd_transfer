from typing import Tuple

import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.linalg import inv

from scipy.signal import hilbert, lfilter, firwin

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
        filter_size: 滤波器长度
        cut_num: 频带分割数
        mode_num: 目标模式数
        max_iter_num: 最大迭代次数

    返回:
        Final_Mode: 分解后的模式矩阵 (每列为一个模式)
    """
    x = np.asarray(x).flatten()
    N = len(x)

    # 初始化频率边界
    freq_bound = np.linspace(0, 1 - 1 / cut_num, cut_num)

    # 生成初始滤波器组
    temp_filters = np.zeros((filter_size, cut_num))
    for n in range(cut_num):
        # 使用 firwin 生成滤波器系数，numtaps 设置为 filter_size
        # matlab: w = window(@hanning,FilterSize); 计算长度为N+2 的汉宁窗，最后输出窗函数不包含 前后的0点，长度依然为N
        # 需要自定义窗口值 参考函数： fmd.matlab_hanning
        cutoff = [freq_bound[n] + epsilon, freq_bound[n] + 1 / cut_num - epsilon]
        w_a = firwin(
            filter_size + 2,
            cutoff,
            window="hann",
            pass_zero=False
        )
        temp_filters[:, n] = w_a[1:-1]

    # 初始化结果存储结构
    result: list[dict] = [{} for _ in range(cut_num + 1)]
    result[0] = {
        "IterCount": None,
        "Iterations": None,
        "CorrMatrix": None,
        "ComparedModeNum": None,
        "StopNum": None
    }

    # 初始化临时信号
    temp_sig = np.tile(x, (cut_num, 1)).T

    iter_count = 1
    while True:
        # 确定迭代次数
        iter_num = max_iter_num - (cut_num - mode_num) * 2 if iter_count == 1 else 2

        current_iter: dict[str, list] = {
            "Iterations": [[] for _ in range(cut_num)],
            "CorrMatrix": None,
            "ComparedModeNum": None,
            "StopNum": None
        }

        # 对每个频带执行MCKD
        for n in range(cut_num):
            f_init = temp_filters[:, n].copy()
            y_Iter, f_Iter, k_Iter, T_Iter = xxc_mckd(
                fs=fs,
                x=temp_sig[:, n],
                f_init=f_init,
                term_iter=iter_num,
                initial_t=None,
                initial_m=1,
                plot_mode=False
            )

            # 存储迭代结果
            current_iter["Iterations"][n] = {
                "y": y_Iter[:, -1],
                "f": f_Iter[:, -1],
                "k": k_Iter[-1],
                "fft": np.abs(np.fft.fft(f_Iter, axis=0))[:filter_size // 2, :],
                "freq": (np.argmax(np.abs(np.fft.fft(f_Iter, axis=0))[:filter_size // 2, :], axis=0) - 1) * (
                        fs / filter_size),
                "T": T_Iter
            }

        # 更新信号和滤波器
        for n in range(cut_num):
            temp_sig[:, n] = current_iter["Iterations"][n]["y"]
            temp_filters[:, n] = current_iter["Iterations"][n]["f"]

        # 计算相关系数矩阵
        corr_matrix = np.abs(np.corrcoef(temp_sig, rowvar=False))
        corr_matrix = np.triu(corr_matrix, k=1)

        # 寻找最相关模式对
        I, J = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)

        # 计算CK值
        XI = temp_sig[:, I] - np.mean(temp_sig[:, I])
        XJ = temp_sig[:, J] - np.mean(temp_sig[:, J])
        T_I = current_iter["Iterations"][I]["T"]
        T_J = current_iter["Iterations"][J]["T"]
        KI = CK(XI, T_I, m_a=1)
        KJ = CK(XJ, T_J, m_a=1)

        # 确定删除的索引
        delete_idx = J if KI > KJ else I

        # 删除模式
        temp_sig = np.delete(temp_sig, delete_idx, axis=1)
        temp_filters = np.delete(temp_filters, delete_idx, axis=1)
        cut_num -= 1

        # 存储迭代信息
        result[iter_count] = {
            "IterCount": iter_count,
            "Iterations": current_iter["Iterations"],
            "CorrMatrix": corr_matrix,
            "ComparedModeNum": (I, J),
            "StopNum": delete_idx
        }

        if temp_sig.shape[1] == mode_num - 1:
            break

        iter_count += 1

    # 提取最终模式
    Final_Mode = np.zeros((N, mode_num))
    for nn in range(mode_num):
        Final_Mode[:, nn] = result[iter_count]["Iterations"][nn]["y"]

    return Final_Mode


def xxc_mckd(fs: float,
             x: np.ndarray,
             f_init: np.ndarray,
             term_iter: int = 30,
             initial_t: int = None,
             initial_m: int = 3,
             plot_mode: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
        ck_iter: 各迭代相关峰度值 (1D数组)
        t_final: 最终估计周期
    """
    # 输入校验
    x = np.asarray(x).flatten()
    L = len(f_init)
    N = len(x)

    # 自动计算初始周期
    if initial_t is None:
        x_env = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
        initial_t = TT(x_env, fs)
    initial_t = int(round(initial_t))

    # 预分配XmT矩阵
    XmT = np.zeros((L, N, initial_m + 1))
    for m in range(initial_m + 1):
        for i in range(L):
            if i == 0:
                start = m * initial_t
                XmT[i, start:, m] = x[:N - start]
            else:
                XmT[i, 1:, m] = XmT[i - 1, :-1, m]

    Xinv = inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

    f = f_init.copy()
    ck_best = 0
    ck_iter = []
    t_final = initial_t
    f_final = np.zeros((L, term_iter))
    y_final = np.zeros((N, term_iter))

    for n in range(term_iter):
        # 计算输出信号
        y = (f.T @ XmT[:, :, 0]).flatten()

        # 生成yt矩阵
        f_final[:, n] = f.flatten()
        yt = np.zeros((N, initial_m + 1))
        yt[:, 0] = y
        for m in range(1, initial_m + 1):
            yt[initial_t:, m] = y[:-initial_t]

        # 计算alpha
        alpha = np.zeros((N, initial_m + 1))
        for m in range(initial_m + 1):
            cols = [k for k in range(initial_m + 1) if k != m]
            alpha[:, m] = np.prod(yt[:, cols], axis=1) ** 2 * yt[:, m]

        # 计算beta
        beta = np.prod(yt, axis=1)

        # 计算Xalpha
        Xalpha = np.zeros(L)
        for m in range(initial_m + 1):
            Xalpha += XmT[:, :, m] @ alpha[:, m]

        # 更新滤波器系数
        numerator = np.sum(y ** 2)
        denominator = 2 * np.sum(beta ** 2)
        f = (numerator / denominator) * Xinv @ Xalpha.reshape(-1, 1)
        f /= np.sqrt(np.sum(f ** 2))

        # 计算CK值
        ck = np.sum(beta ** 2) / (np.sum(y ** 2) ** (initial_m + 1))
        ck_iter.append(ck)
        if ck > ck_best:
            ck_best = ck

        # 更新周期T
        xyenvelope = np.abs(hilbert(y)) - np.mean(np.abs(hilbert(y)))
        initial_t = TT(xyenvelope, fs)
        initial_t = int(round(initial_t))
        t_final = initial_t

        # 重新计算XmT
        XmT = np.zeros((L, N, initial_m + 1))
        for m in range(initial_m + 1):
            for j in range(L):
                if j == 0:
                    start = m * initial_t
                    XmT[j, start:, m] = x[:N - start]
                else:
                    XmT[j, 1:, m] = XmT[j - 1, :-1, m]

        Xinv = inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

        y_final[:, n] = lfilter(f_final[:, n], 1, x)

    return y_final, f_final, np.array(ck_iter), t_final


def TT(y, fs):
    """通过自相关函数估计信号的周期 (样本数)

    参数:
        y : 输入信号 (1D数组)
        fs : 采样频率 (用于确定最大滞后量)

    返回:
        T : 估计的周期 (样本数)
    """
    M = fs
    n = len(y)
    max_lag = n - 1
    M = min(M, max_lag)  # 确保M不超过最大可能滞后

    # 计算自相关
    autocorr = np.correlate(y, y, mode='full')

    # 确定中间索引
    middle = n - 1
    start = middle - M
    end = middle + M + 1

    # 提取从滞后-M到M的部分
    NA = autocorr[start:end]

    # 归一化到零滞后的自相关值
    NA = NA / autocorr[middle]
    NA = np.insert(NA, 0, 0)
    NA = np.append(NA, 0)
    mid = np.ceil(len(NA) / 2)
    NA = NA[int(mid) - 1:]

    # 寻找第一个过零点
    sample1 = NA[0]
    zeroposi = None
    for lag in range(1, len(NA)):
        sample2 = NA[lag]

        if sample1 > 0 > sample2:
            zeroposi = lag
            break
        elif sample1 == 0 or sample2 == 0:
            zeroposi = lag
            break
        else:
            sample1 = sample2

    NA = NA[zeroposi:]
    max_position = np.argmax(NA)

    # zeroposi，max_position对应的是索引值，matlab的索引值是从1开始的
    T = zeroposi + 1 + max_position + 1

    return T


def CK(x, t_a, m_a=2):
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
    x_shift = np.zeros((m_a + 1, N))
    x_shift[0, :] = x

    # 逐行填充时移信号
    for m in range(1, m_a + 1):
        # 从第T位置开始填充左移后的信号 (MATLAB索引转换为Python索引)
        x_shift[m, t_a:] = x_shift[m - 1, :-t_a]

    # 计算分子：时移信号逐点乘积的平方和
    numerator = np.sum(np.prod(x_shift, axis=0) ** 2)

    # 计算分母：原始信号能量的(M+1)次方
    denominator = np.sum(x ** 2) ** (m_a + 1)

    return numerator / denominator


def max_ij(x):
    """返回矩阵X中最大值的行索引、列索引和最大值"""
    # 按列找最大值及其行索引
    col_max_values = np.max(x, axis=0)
    row_indices = np.argmax(x, axis=0)

    # 找全局最大值及其列索引
    max_value = np.max(col_max_values)
    j = np.argmax(col_max_values)

    # 获取对应行索引
    i = row_indices[j]

    return i, j, max_value


def matlab_hanning(filter_size: int) -> list[float]:
    """
    生成与 MATLAB 完全一致的汉宁窗（首尾为 0）

    参数:
        filter_size (int): 窗口长度

    返回:
        np.ndarray: 窗口数组
    """
    # 生成从 0 到 N-1 的索引
    N = filter_size + 2
    n = np.arange(N)

    # MATLAB 的 hanning 公式
    window = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    return window[1:-1].tolist()


def main():
    np.random.seed(42)
    data = pandas.read_excel(r"data/原始信号.xlsx", header=None)
    signal = data.values

    signal = np.sin(2 * np.pi * signal) + np.cos(0.3 * np.pi * signal)

    filter_size = 30
    cut_num = 7
    mod_num = 4
    max_iter_num = 20
    fs = len(signal)
    imfs = FMD(fs, signal, filter_size, cut_num, mod_num, max_iter_num)

    matlab_data = loadmat(r"data/matlab_result.mat")
    matlab_modes = matlab_data["u"]

    plt.figure(figsize=(12, 6))
    for i in range(imfs.shape[1]):
        plt.subplot(imfs.shape[1], 1, i + 1)
        plt.plot(matlab_modes[:, i], "r--", label="MATLAB")
        plt.plot(imfs[:, i], "b-", alpha=0.5, label="Python")
        plt.title(f"Mode {i + 1}")
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
