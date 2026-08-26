# -*- coding: utf-8 -*-
"""
把 data/ 里的 FK 参考点云画成 two-step diffusion 论文 4.3 / 4.4 节的图。

论文的图都是 (a) FK 参考 + (b) Meanflow + (c) MF+DP 三者并排，我们手上只有 (a)，
所以这里复现参考列 / 参考行，热启动那两条曲线直接取论文表格里的数字做对照。

论文的画法（照抄）
------------------
    Fig.5   T^2 上的 72x72 **计数**直方图，viridis，每个 panel 各自一根 colorbar
    Fig.8   x-y / x-z / y-z 三个坐标面投影，整行共用一根 colorbar
    Fig.7   c_T 对代数 T 作图，横轴 log，虚线是长时参考
    Fig.10  同 Fig.7，3D 版
    Fig.6/9 右图  纵轴对 -log2(sigma)，绿色竖虚线标训练区间末端 3.75

分箱数为什么取 72
-----------------
Fig.5 图注写死 72x72，Fig.8 没写。但把 PDF 里的位图抠出来数同色像素段，两张图
渲染出来都只有 48 个色块（Fig.5 每块 15px/729px，Fig.8 每块 13px/607px）——
这是低 dpi 光栅化把 72 列压成 48 列的痕迹，不是真实分箱数：colorbar 上限取自
数据数组而非像素，而峰值只有按 72 箱才对得上（M=2e4 时）：

    图          论文 colorbar 上限     本实现 48 箱   64 箱   72 箱
    Fig.5 上行        40                  70          44      35
    Fig.5 下行        35                  64          49      41
    Fig.8 整行       ~37                  64          41      34

所以两处都按 72 箱画（4.4 节原文也说 "The setup parallels that of the 2D experiment"）。

产出 figs/
    fig5_2d.png    sigma = 2^-3.75（训练内）和 2^-4（外推），论文 Fig.5 选的就是这两档
    fig8_3d.png    sigma = 2^-4.5 的三面投影
    fig7_2d.png    2D 特征值收敛
    fig10_3d.png   3D 特征值收敛
    sweep_2d.png   Eq.26 全部 11 档的不变测度
    sweep_3d.png   同上，取 x-z 投影
    speed.png      c*(sigma) 随 -log2(sigma) 的变化

用法:  python viz.py
"""

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TWO_PI = 2.0 * np.pi
OUT = 'figs'
BINS = 72                                  # 见模块 docstring "分箱数为什么取 72"
CLR = dict(mf='#e8a33d', dp='#6cafe6', cold='#159e7a', ref='0.35')

# 论文 Fig.7 / Fig.10 左表：两种热启动在 T = 1,2,4,...,128 的 c_T（都在 sigma = 2^-4.5）
PAPER_WARM = {
    'kpp2d': ([0.733899, 0.763092, 0.770337, 0.795811, 0.784826, 0.787654, 0.787575, 0.783591],
              [0.761573, 0.773892, 0.788397, 0.781912, 0.785587, 0.783146, 0.775075, 0.779001]),
    'kpp3d': ([0.566547, 0.551988, 0.546482, 0.576172, 0.597887, 0.609560, 0.609690, 0.614481],
              [0.613531, 0.610457, 0.621819, 0.599699, 0.590171, 0.614999, 0.615682, 0.613865]),
}
PAPER_T = [1, 2, 4, 8, 16, 32, 64, 128]


def load(exp):
    """读 data/<exp>/*.npz，按 -log2(sigma) 升序（2.00 -> 4.50）返回 meta + 末代点云。"""
    runs = []
    for f in glob.glob(os.path.join('data', exp, '*.npz')):
        d = np.load(f)
        m = json.loads(str(d['meta']))
        m['xT'] = d['qT' if exp == 'kpp2d' else 'xT']
        runs.append(m)
    return sorted(runs, key=lambda m: -m['log2sigma'])


def save(fig, fname):
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)
    print('  ' + fname)


def grid(items, bins, ncol, fname, sup, shared=False):
    """items = [(标题, 点云, 横轴维, 纵轴维)]。

    shared=True 整行共用一根 colorbar（Fig.8 的样式），否则每个 panel 各自一根（Fig.5）。
    返回各 panel 的峰值计数，用来和论文 colorbar 上限对表。
    """
    H = [np.histogram2d(X[:, i], X[:, j], bins=bins, range=[[0, TWO_PI]] * 2)[0].T
         for _, X, i, j in items]
    n = len(items)
    ncol = min(ncol, n)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.2 * nrow),
                             squeeze=False, layout='constrained')
    vmax = max(h.max() for h in H) if shared else None
    for ax, h, (t, _, i, j) in zip(axes.ravel(), H, items):
        im = ax.imshow(h, origin='lower', extent=[0, TWO_PI, 0, TWO_PI],
                       cmap='viridis', vmax=vmax, interpolation='nearest')
        ax.set_title(t, fontsize=10)
        ax.set_xlabel('xyz'[i]); ax.set_ylabel('xyz'[j])
        ax.set_xticks([0, 2, 4, 6]); ax.set_yticks([0, 2, 4, 6])
        if not shared:
            fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    if shared:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02)
    fig.suptitle(sup)
    save(fig, fname)
    return [float(h.max()) for h in H]


def fig_eigen(m, exp, fname, sup):
    """Fig.7 / Fig.10：我们的冷启动曲线 + 论文表里的两种热启动 + 长时参考虚线。"""
    T, c = zip(*[(t, ct) for t, _, ct in m['trace']])
    mf, dp = PAPER_WARM[exp]
    fig, ax = plt.subplots(figsize=(6.6, 4.2), layout='constrained')
    ax.plot(PAPER_T, mf, 'o-', color=CLR['mf'], label='warm start (MF-only) [paper]')
    ax.plot(PAPER_T, dp, 'o-', color=CLR['dp'], label='warm start (MF+DP) [paper]')
    ax.plot(T, c, 's-', ms=4, color=CLR['cold'], label='cold start (uniform init) [ours]')
    ax.axhline(m['c_hat'], ls='--', color=CLR['ref'],
               label='long-run FK reference (T=4096) [ours]')
    ax.set_xscale('log')
    ax.set_xlabel('generations')
    ax.set_ylabel('eigval')
    ax.grid(ls='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(sup)
    save(fig, fname)


def fig_speed(runs, fname):
    """Fig.6/9 右图的样式，纵轴换成我们能直接算的 c*(sigma)。"""
    fig, ax = plt.subplots(figsize=(6.6, 4.2), layout='constrained')
    for exp, col, lab in (('kpp2d', CLR['mf'], '2D cellular flow (Sec. 4.3)'),
                          ('kpp3d', CLR['dp'], '3D Kolmogorov flow (Sec. 4.4)')):
        ax.plot([-m['log2sigma'] for m in runs[exp]], [m['c_hat'] for m in runs[exp]],
                'o-', color=col, label=lab)
    ax.axvline(3.75, ls='--', color='green')
    ax.text(3.78, ax.get_ylim()[1], ' extrapolation', va='top', fontsize=8, color='green')
    ax.set_xlabel(r'$-\log_2(\sigma)$')
    ax.set_ylabel(r'$\hat{c}_T(\lambda)$')
    ax.grid(ls='--', alpha=0.4)
    ax.legend(fontsize=9)
    ax.set_title('front speed vs diffusion level, $M=2\\times10^4$, $T=4096$')
    save(fig, fname)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = {e: load(e) for e in ('kpp2d', 'kpp3d')}
    pick = lambda e, l: next(m for m in runs[e] if abs(m['log2sigma'] - l) < 1e-9)
    lab = lambda m: r'$-\log_2\sigma=%.2f$' % -m['log2sigma']

    print('figs/')
    p = grid([('Reference  ' + lab(pick('kpp2d', l)), pick('kpp2d', l)['xT'], 0, 1)
              for l in (-3.75, -4.0)],
             BINS, 2, 'fig5_2d.png',
             'Sec. 4.3  invariant measure on $T^2$  (%dx%d histogram)' % (BINS, BINS))
    print('    peak counts %s   (paper Fig.5 colorbar tops: 40, 35)' % p)

    X = pick('kpp3d', -4.5)['xT']
    p = grid([('x-y', X, 0, 1), ('x-z', X, 0, 2), ('y-z', X, 1, 2)],
             BINS, 3, 'fig8_3d.png',
             'Sec. 4.4  reference projections at $\\sigma=2^{-4.5}$  (%dx%d histogram)'
             % (BINS, BINS), shared=True)
    print('    peak counts %s   (paper Fig.8 colorbar top: ~37)' % p)

    for exp, plane, sup in (('kpp2d', (0, 1), 'Sec. 4.3  invariant measure on $T^2$'),
                            ('kpp3d', (0, 2), 'Sec. 4.4  x-z projection')):
        grid([(lab(m), m['xT'], *plane) for m in runs[exp]], BINS, 4,
             'sweep_%s.png' % exp[-2:], sup + ', all 11 levels of Eq.26')

    fig_eigen(pick('kpp2d', -4.5), 'kpp2d', 'fig7_2d.png',
              'Sec. 4.3  eigenvalue convergence at $\\sigma=2^{-4.5}$')
    fig_eigen(pick('kpp3d', -4.5), 'kpp3d', 'fig10_3d.png',
              'Sec. 4.4  eigenvalue convergence at $\\sigma=2^{-4.5}$')
    fig_speed(runs, 'speed.png')


if __name__ == '__main__':
    main()
