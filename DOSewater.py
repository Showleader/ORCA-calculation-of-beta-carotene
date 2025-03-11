import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 数据文件所在目录
    data_dir = r"G:\BaiduNetdiskDownload\Multiwfn_3.8_dev_bin_Win64\Multiwfn_3.8_dev_bin_Win64\allresult"
    files = [
        "DOSwater.txt",
        "DOSwater2e.txt"
    ]
    colors = ['blue', 'green']

    plt.figure()

    for filename, color in zip(files, colors):
        filepath = os.path.join(data_dir, filename)
        data = np.loadtxt(filepath)
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x, y, color=color, label=filename)

    plt.xlim(-1.5, 3.5)
    plt.xlabel("Energy(eV)")
    plt.ylabel("Density-of-states（states/eV）")
    plt.title("TDOS")
    plt.legend()
    plt.tight_layout()

    # 保存图片到和本脚本相同的文件夹
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TDOS_plot.png")
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    main()
