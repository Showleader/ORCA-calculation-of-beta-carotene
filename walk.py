import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar support
import numpy as np
from scipy.optimize import curve_fit  # 用于高斯拟合
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# ==== Circle parameters: same as before ====
center_params = {
    1: (0, 1.732),    # Small circle 1 center: (0, √3)
    2: (-1, 0),
    3: (1, 0),
    4: (0, -1.732),
}

n_points = {
    1: 12,
    2: 12,
    3: 12,
    4: 24,
}

radii = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
}

phase_offset = {
    1: 5 * math.pi / 6,
    2: math.pi / 6,
    3: 3 * math.pi / 2,
    4: math.pi / 6,
}

overlap = {
    (1, 2): (2, 0),   # Small circle 1 node 2 and Small circle 2 node 0
    (2, 0): (1, 2),
    (1, 0): (3, 2),   # Small circle 1 node 0 and Small circle 3 node 2
    (3, 2): (1, 0),
    (2, 2): (3, 0),   # Small circle 2 node 2 and Small circle 3 node 0
    (3, 0): (2, 2),
    (2, 4): (4, 20),  # Small circle 2 node 4 and Large circle node 20
    (4, 20): (2, 4),
    (3, 10): (4, 0),  # Small circle 3 node 10 and Large circle node 0
    (4, 0): (3, 10),
}

def build_positions():
    pos = {}
    for ring in n_points:
        n = n_points[ring]
        cen = center_params[ring]
        r = radii[ring]
        offset = phase_offset[ring]
        for i in range(n):
            theta = offset + 2 * math.pi * i / n
            x = cen[0] + r * math.sin(theta)
            y = cen[1] + r * math.cos(theta)
            pos[(ring, i)] = (x, y)
    return pos

def build_graph():
    G = nx.DiGraph()   # Directed graph
    for ring in n_points:
        n = n_points[ring]
        for i in range(n):
            node = (ring, i)
            G.add_node(node)
            left = (ring, (i - 1) % n)
            right = (ring, (i + 1) % n)
            G.add_edge(node, left, weight=1)
            G.add_edge(node, right, weight=1)
            if node in overlap:
                jump = overlap[node]
                G.add_edge(node, jump, weight=1)
                jump_left = (jump[0], (jump[1] - 1) % n_points[jump[0]])
                jump_right = (jump[0], (jump[1] + 1) % n_points[jump[0]])
                G.add_edge(node, jump_left, weight=1)
                G.add_edge(node, jump_right, weight=1)
    return G

start = (1, 7)

def random_walk_episode(max_steps=10000):
    current = start
    path = [current]
    steps = 0
    while current[0] != 4 and steps < max_steps:
        moves = []
        n = n_points[current[0]]
        left = (current[0], (current[1] - 1) % n)
        right = (current[0], (current[1] + 1) % n)
        moves.extend([left, right])
        if current in overlap:
            jump = overlap[current]
            jump_left = (jump[0], (jump[1] - 1) % n_points[jump[0]])
            jump_right = (jump[0], (jump[1] + 1) % n_points[jump[0]])
            moves.extend([jump, jump_left, jump_right])
            probs = [1/5] * 5
        else:
            probs = [0.5, 0.5]
        current = random.choices(moves, weights=probs, k=1)[0]
        path.append(current)
        steps += 1
        if current[0] == 4:
            break
    return path, steps

def simulate_electrons(G, n_electrons=100000):
    steps_list = []
    for _ in tqdm(range(n_electrons), desc="Simulating electrons"):
        _, steps = random_walk_episode()
        steps_list.append(steps)
    return steps_list

def compute_shortest_path(G):
    target_nodes = [node for node in G.nodes() if node[0] == 4]
    sp_lengths = nx.single_source_shortest_path_length(G, start)
    best = min([sp_lengths[t] for t in target_nodes if t in sp_lengths])
    for t in target_nodes:
        if t in sp_lengths and sp_lengths[t] == best:
            sp = nx.shortest_path(G, start, t)
            return sp, best
    return None, None

def plot_graph_path(G, pos, path, title, filename):
    fig, ax = plt.subplots(figsize=(6,6))
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color="#D3D3D3", ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=80, node_color="blue", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=0.2, edge_color="red", ax=ax)
    labels = {node: f"{node[0]}-{node[1]}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    
    for ring, center in center_params.items():
        ax.text(center[0], center[1], f"Ring {ring}", fontsize=12, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.6))
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis("off")
    plt.savefig(filename)
    plt.close()

# 定义高斯函数用于拟合
def gaussian(x, a1, a2, a3):
    return a1 * np.exp(-((x - a2) / a3)**2)

def plot_walk_steps_distribution(steps_list, curve_filename, data_txt_filename, fit_filename):
    # 使用步长为10，从10.0开始
    data = np.array(steps_list)
    bins = np.arange(10.0, data.max() + 10, 10)
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    normalized_counts = counts / counts.sum()  # 归一化处理
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, normalized_counts, '-o', linewidth=2, markersize=4, label="Proportion")
    
    # 机器学习自动拟合曲线：采用高斯过程回归（Gaussian Process Regression）
    # 定义核函数
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(bin_centers.reshape(-1, 1), normalized_counts)
    x_fit = np.linspace(bin_centers[0], bin_centers[-1], 200)
    y_fit = gpr.predict(x_fit.reshape(-1, 1))
    plt.plot(x_fit, y_fit, 'k-', linewidth=2, label="ML Fit")
    optimal_kernel = gpr.kernel_
    print("Optimal kernel:", optimal_kernel)
    
    plt.title("Proportion distribution with steps (step=10)")
    plt.xlabel("Step")
    plt.ylabel("Proportion")
    plt.legend()
    plt.savefig(curve_filename)
    plt.close()
    
    # 保存归一化后的数据
    with open(data_txt_filename, "w") as f:
        for x, y in zip(bin_centers, normalized_counts):
            f.write(f"{x}\t{y}\n")
            
    # 保存最优拟合公式到 fit.txt（这里输出的是最优 kernel）
    with open(fit_filename, "w") as f:
        f.write("Fitted kernel: " + str(optimal_kernel) + "\n")
            
def main():
    # 定义输出文件夹
    folder = r"G:\BaiduNetdiskDownload\cityudessertation\PHY6505\codeandresult"
    os.makedirs(folder, exist_ok=True)

    random.seed(0)
    pos = build_positions()
    G = build_graph()

    sp, sp_steps = compute_shortest_path(G)
    print("Shortest steps from small circle 1 to large circle:", sp_steps)
    
    sample_path, sample_steps = random_walk_episode()
    print("Sample random walk steps:", sample_steps)
    
    n_electrons = 100000  
    print("Simulating {} electrons...".format(n_electrons))
    steps_list = simulate_electrons(G, n_electrons)
    avg_steps = sum(steps_list) / len(steps_list)
    ratio = avg_steps / sp_steps
    print("Average steps of random walk: {:.2f}".format(avg_steps))
    print("Ratio (Random steps / Shortest steps): {:.2f}".format(ratio))
    
    sample_path_file = os.path.join(folder, "sample_path.png")
    shortest_path_file = os.path.join(folder, "shortest_path.png")
    distribution_file = os.path.join(folder, "proportion_distribution_with_steps.png")
    distribution_data_file = os.path.join(folder, "distribution_data.txt")
    fit_file = os.path.join(folder, "fit.txt")
    results_file = os.path.join(folder, "results.txt")
    
    plot_graph_path(G, pos, sample_path, "An example of walk routine", sample_path_file)
    plot_graph_path(G, pos, sp, "The most efficient walk routine", shortest_path_file)
    
    # 使用步长为10的数据绘制概率分布，并使用机器学习拟合曲线后保存数据到文本以及拟合公式到 fit.txt
    plot_walk_steps_distribution(steps_list, distribution_file, distribution_data_file, fit_file)
    
    with open(results_file, "w") as f:
        f.write(f"Shortest steps from small circle 1 to large circle: {sp_steps}\n")
        f.write(f"Sample random walk steps: {sample_steps}\n")
        f.write(f"Average steps of random walk: {avg_steps:.2f}\n")
        f.write(f"Ratio (Random steps/Shortest steps): {ratio:.2f}\n")

if __name__ == "__main__":
    main()
