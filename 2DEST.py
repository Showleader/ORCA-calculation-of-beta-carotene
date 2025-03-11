import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import os

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y):
    """二维高斯函数"""
    return np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))

def simulate_incoherent(energies, osc_strengths, grid, sigma, delta):
    """
    无相干叠加：对每个跃迁使用二维高斯函数，
    中心设置为 (E, E+delta)，从而使得不同跃迁在检测能量方向上略有偏移
    """
    X, Y = grid
    spectrum = np.zeros_like(X)
    for E, f in zip(energies, osc_strengths):
        spectrum += f * gaussian_2d(X, Y, E, E + delta, sigma, sigma)
    return spectrum

def simulate_coherent(energies, osc_strengths, grid, sigma, T, hbar, delta):
    """
    严格的量子相干叠加：
      对于每个跃迁，计算其复数振幅：
         A_k(x, y; T) = f_k * g(x, y; E_k, σ) * exp[i*(X - Y)*T/ħ]
      其中 g(x, y; E, σ) 的中心为 (E, E+delta)
      这样相位因子依赖于 (X-Y)，更贴近2DES中激发与探测能量的关系，
      有助于实现多个电子峰值的融合而非收缩
      最后将所有跃迁振幅求和，并取绝对值的平方得到强度：
         I(x, y; T) = |∑_k A_k(x, y; T)|^2
    """
    X, Y = grid
    amplitude = np.zeros_like(X, dtype=complex)
    for E, f in zip(energies, osc_strengths):
        amplitude += f * gaussian_2d(X, Y, E, E + delta, sigma, sigma) * np.exp(1j * (X - Y) * T / hbar)
    intensity = np.abs(amplitude)**2
    return intensity

def get_envelope(T):
    """
    分段线性包络函数：
      0 <= T <= 300 fs：包络由 0 线性增加到 1（相干增强，有利于峰值融合）
      300 < T <= 600 fs：包络由 1 线性减小到 0（退相干，峰值消退）
    """
    if T <= 300:
        return T / 300
    else:
        return (600 - T) / 300

def read_transitions(filepath):
    """
    从文本文件中读取跃迁能量和振子强度
    文本格式示例：
      #   1   2.9220 eV    424.31 nm   f=  4.23584   Spin multiplicity= ?:
         H -> L 70.6%
      ...
    """
    energies = []
    osc_strengths = []
    pattern = re.compile(r'#\s*\d+\s+([\d\.]+)\s*eV\s+[\d\.]+\s*nm\s+f=\s*([\d\.]+)')
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                energies.append(float(match.group(1)))
                osc_strengths.append(float(match.group(2)))
    return energies, osc_strengths

def main():
    # 数据文件所在路径，根据实际情况修改
    folder = r"G:\BaiduNetdiskDownload\Multiwfn_3.8_dev_bin_Win64\Multiwfn_3.8_dev_bin_Win64"
    txt_filename = "carotene_exc.txt"
    filepath = os.path.join(folder, txt_filename)
    
    energies, osc_strengths = read_transitions(filepath)
    if not energies:
        print("No valid transition data found in", filepath)
        return

    # 定义能量坐标网格
    e_min, e_max = 2.5, 8.0
    n_points = 2000
    x = np.linspace(e_min, e_max, n_points)
    # 为了适应检测能量的偏移，延伸 y 轴范围
    y = np.linspace(e_min, e_max + 0.5, n_points)
    X, Y = np.meshgrid(x, y)
    
    sigma = 0.05
    hbar = 0.6582119  # 单位：eV·fs
    delta = 0.2       # 检测能量偏移，单位：eV

    # 计算无相干图谱（代表 T=0 fs 和 T=600 fs 时的情况）
    incoherent_spectrum = simulate_incoherent(energies, osc_strengths, (X, Y), sigma, delta)

    # 模拟 0 到 600 fs，每隔 100 fs 计算一次图谱，共 7 个时刻
    time_points = np.arange(0, 601, 100)
    linear_data = []
    log_data = []

    for T in time_points:
        envelope = get_envelope(T)
        coherent_spectrum = simulate_coherent(energies, osc_strengths, (X, Y), sigma, T, hbar, delta)
        # 混合无相干和相干部分
        final_spectrum = (1 - envelope) * incoherent_spectrum + envelope * coherent_spectrum
        linear_data.append((T, final_spectrum))
        
        # 计算对数图数据：对 spectrum>0 取 log10，否则赋值 -300
        with np.errstate(divide='ignore'):
            log_spectrum = np.where(final_spectrum > 0, np.log10(final_spectrum), -300)
        log_data.append((T, log_spectrum))
        print(f"Calculated spectrum at T = {T} fs")

    # 统一线性图色阶
    linear_min = min(spectrum.min() for (_, spectrum) in linear_data)
    linear_max = max(spectrum.max() for (_, spectrum) in linear_data)
    levels_linear = np.linspace(linear_min, linear_max, 100)
    
    # 统一对数图色阶（仅考虑有限值）
    finite_logs = np.hstack([log_spectrum[np.isfinite(log_spectrum)] for (_, log_spectrum) in log_data])
    log_min = finite_logs.min() if finite_logs.size > 0 else -300
    log_max = finite_logs.max() if finite_logs.size > 0 else 1
    levels_log = np.linspace(log_min, log_max, 100)

    # -----------------------
    # 合成线性图的大图：上排 3 张， 下排 4 张
    fig_linear, axes_linear = plt.subplots(2, 4, figsize=(16, 8))
    fig_linear.suptitle("Composite 2D Electronic Spectrum (Linear Scale)", fontsize=16)

    for idx, (T, spectrum) in enumerate(linear_data):
        if idx < 3:
            ax = axes_linear[0, idx]
        else:
            ax = axes_linear[1, idx - 3]
        cf = ax.contourf(X, Y, spectrum, levels=levels_linear, cmap="inferno")
        ax.set_title(f"T = {T} fs")
        ax.set_xlabel("Excitation Energy (eV)")
        ax.set_ylabel("Detection Energy (eV)")
    if axes_linear.shape[1] > 3:
        axes_linear[0, 3].axis("off")
    fig_linear.subplots_adjust(bottom=0.15, top=0.9, wspace=0.3, hspace=0.3)
    cbar = fig_linear.colorbar(cf, ax=axes_linear.ravel().tolist(), orientation='horizontal', pad=0.1, fraction=0.05)
    cbar.set_label("Intensity")
    composite_linear_filename = os.path.join(folder, "Composite_2DES_Linear.png")
    plt.savefig(composite_linear_filename, dpi=300)
    plt.close(fig_linear)
    print("Saved composite linear image:", composite_linear_filename)

    # -----------------------
    # 合成对数图的大图：上排 3 张， 下排 4 张
    base_cmap = plt.get_cmap("rainbow")
    cmap_log = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, 256)))
    cmap_log.set_bad(color="lightgrey")

    fig_log, axes_log = plt.subplots(2, 4, figsize=(16, 8))
    fig_log.suptitle("Composite 2D Electronic Spectrum (Log10 Scale)", fontsize=16)

    for idx, (T, log_spectrum) in enumerate(log_data):
        if idx < 3:
            ax = axes_log[0, idx]
        else:
            ax = axes_log[1, idx - 3]
        cf = ax.contourf(X, Y, log_spectrum, levels=levels_log, cmap=cmap_log)
        ax.contour(X, Y, log_spectrum, levels=levels_log, colors='black', 
                   linestyles='dashed', linewidths=0.5)
        cs = ax.contour(X, Y, log_spectrum, levels=[-300], colors='magenta', 
                        linestyles='solid', linewidths=1.5)
        ax.clabel(cs, fmt="<= -300", fontsize=9, colors='magenta')
        ax.set_title(f"T = {T} fs")
        ax.set_xlabel("Excitation Energy (eV)")
        ax.set_ylabel("Detection Energy (eV)")
    if axes_log.shape[1] > 3:
        axes_log[0, 3].axis("off")
    fig_log.subplots_adjust(bottom=0.15, top=0.9, wspace=0.3, hspace=0.3)
    cbar_log = fig_log.colorbar(cf, ax=axes_log.ravel().tolist(), orientation='horizontal', pad=0.1, fraction=0.05)
    cbar_log.set_label("Log10(Intensity)")
    composite_log_filename = os.path.join(folder, "Composite_2DES_Log.png")
    plt.savefig(composite_log_filename, dpi=300)
    plt.close(fig_log)
    print("Saved composite log image:", composite_log_filename)

    # ==========================================================
    # 以下为新增部分：对每个时刻的最终光谱进行二维傅里叶变换，并生成傅里叶变换后的线性和对数图像
    # 根据傅里叶变换定义：G^(1)(τ)= (1/T)∫ S(ω)e^(-iωτ)dω
    # 这里采用 np.fft.fft2 进行二维傅里叶变换，然后使用 fftshift 将零频率置于中心

    ft_linear_data = []
    ft_log_data = []
    
    for T, spectrum in linear_data:
        FT = np.fft.fftshift(np.fft.fft2(spectrum))
        FT_magnitude = np.abs(FT)
        with np.errstate(divide='ignore'):
            log_FT = np.where(FT_magnitude > 0, np.log10(FT_magnitude), -300)
        ft_linear_data.append((T, FT_magnitude))
        ft_log_data.append((T, log_FT))
        print(f"Calculated Fourier transform for T = {T} fs")
    
    # 构造傅里叶变换图像的频率坐标，原单位为1/eV，现转换为Hz
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    # 1 eV 对应的频率为 1/(4.135667696e-15) Hz
    eV_to_Hz = 1 / 4.135667696e-15
    fx = np.fft.fftshift(np.fft.fftfreq(n_points, d=dx)) * eV_to_Hz
    fy = np.fft.fftshift(np.fft.fftfreq(n_points, d=dy)) * eV_to_Hz
    FX, FY = np.meshgrid(fx, fy)
    
    # 统一傅里叶线性图色阶
    ft_linear_min = min(spectrum.min() for (_, spectrum) in ft_linear_data)
    ft_linear_max = max(spectrum.max() for (_, spectrum) in ft_linear_data)
    levels_ft_linear = np.linspace(ft_linear_min, ft_linear_max, 100)
    
    # 统一傅里叶对数图色阶（仅考虑有限值）
    finite_ft_logs = np.hstack([log_spectrum[np.isfinite(log_spectrum)] for (_, log_spectrum) in ft_log_data])
    ft_log_min = finite_ft_logs.min() if finite_ft_logs.size > 0 else -300
    ft_log_max = finite_ft_logs.max() if finite_ft_logs.size > 0 else 1
    levels_ft_log = np.linspace(ft_log_min, ft_log_max, 100)
    
    # 自定义傅里叶变换图像的色图：低值用黑色，高值用白色，并过渡多种颜色
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['black', 'blue', 'cyan', 'green', 'yellow', 'white'])
    
    # -----------------------
    # 合成傅里叶变换线性图的大图：上排 3 张， 下排 4 张
    fig_ft_linear, axes_ft_linear = plt.subplots(2, 4, figsize=(16, 8))
    fig_ft_linear.suptitle("Composite Fourier Transform Spectrum (Linear Scale)", fontsize=16)
    
    for idx, (T, ft_spectrum) in enumerate(ft_linear_data):
        if idx < 3:
            ax = axes_ft_linear[0, idx]
        else:
            ax = axes_ft_linear[1, idx - 3]
        cf = ax.contourf(FX, FY, ft_spectrum, levels=levels_ft_linear, cmap=custom_cmap)
        ax.set_title(f"T = {T} fs")
        ax.set_xlabel("Omega (Hz)")
        ax.set_ylabel("Omega (Hz)")
    if axes_ft_linear.shape[1] > 3:
        axes_ft_linear[0, 3].axis("off")
    fig_ft_linear.subplots_adjust(bottom=0.15, top=0.9, wspace=0.3, hspace=0.3)
    cbar_ft_linear = fig_ft_linear.colorbar(cf, ax=axes_ft_linear.ravel().tolist(), orientation='horizontal', pad=0.1, fraction=0.05)
    cbar_ft_linear.set_label("Fourier Intensity")
    ft_linear_filename = os.path.join(folder, "Composite_FT_Linear.png")
    plt.savefig(ft_linear_filename, dpi=300)
    plt.close(fig_ft_linear)
    print("Saved composite Fourier transform linear image:", ft_linear_filename)
    
    # -----------------------
    # 合成傅里叶变换对数图的大图：上排 3 张， 下排 4 张
    fig_ft_log, axes_ft_log = plt.subplots(2, 4, figsize=(16, 8))
    fig_ft_log.suptitle("Composite Fourier Transform Spectrum (Log10 Scale)", fontsize=16)
    
    for idx, (T, log_ft_spectrum) in enumerate(ft_log_data):
        if idx < 3:
            ax = axes_ft_log[0, idx]
        else:
            ax = axes_ft_log[1, idx - 3]
        cf = ax.contourf(FX, FY, log_ft_spectrum, levels=levels_ft_log, cmap=custom_cmap)
        ax.contour(FX, FY, log_ft_spectrum, levels=levels_ft_log, colors='black', 
                   linestyles='dashed', linewidths=0.5)
        cs = ax.contour(FX, FY, log_ft_spectrum, levels=[-300], colors='magenta', 
                        linestyles='solid', linewidths=1.5)
        ax.clabel(cs, fmt="<= -300", fontsize=9, colors='magenta')
        ax.set_title(f"T = {T} fs")
        ax.set_xlabel("Omega (Hz)")
        ax.set_ylabel("Omega (Hz)")
    if axes_ft_log.shape[1] > 3:
        axes_ft_log[0, 3].axis("off")
    fig_ft_log.subplots_adjust(bottom=0.15, top=0.9, wspace=0.3, hspace=0.3)
    cbar_ft_log = fig_ft_log.colorbar(cf, ax=axes_ft_log.ravel().tolist(), orientation='horizontal', pad=0.1, fraction=0.05)
    cbar_ft_log.set_label("Log10(Fourier Intensity)")
    ft_log_filename = os.path.join(folder, "Composite_FT_Log.png")
    plt.savefig(ft_log_filename, dpi=300)
    plt.close(fig_ft_log)
    print("Saved composite Fourier transform log image:", ft_log_filename)

if __name__ == "__main__":
    main()
