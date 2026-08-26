# -*- coding: utf-8 -*-
"""
生成 two-step diffusion 论文 4.3 节的数据：
"2D KPP Front Speed Experiment"（时变胞状流，T^2 = [0,2pi)^2）

3D 的 4.4 节在 kpp3d_kolmogorov.py，两个文件互不依赖。

论文
----
[TSD] Shen, Wang, Xin, Zhang, "Two-Step Diffusion: Fast Sampling and Reliable
      Prediction for 3D Keller-Segel and KPP Equations in Fluid Flows",
      arXiv:2601.20024v1.（本目录上一级的 two-step diffusion.pdf）

4.3 节要什么
------------
KPP 反应-扩散-对流方程 (Eq.17)::

    u_t + v.grad(u) = kappa*Lap(u) + r*u(1-u)

沿 e1 的传播速度 c*(e1) = inf_{lambda>0} mu(lambda)/lambda，mu 是下述算子的主特征值 (Eq.19)::

    A w = kappa*Lap(w) + (2*kappa*lambda*e1 + v).grad(w)
                       + (kappa*lambda^2 + lambda*(v.e1) + 1) w

对应的 twisted 扩散 (Eq.20)、速度场 (Eq.21)、势 (Eq.22)::

    dX = (2*kappa*lambda*e1 + v(X,t)) dt + sqrt(2*kappa) dW
    v(x,t) = ( -cos(x2) - theta*cos(2*pi*t)*sin(x1),
                cos(x1) + theta*cos(2*pi*t)*sin(x2) )
    V_lambda(x,t) = kappa*lambda^2 + lambda*(v(x,t).e1)

Feynman-Kac 半群用权重 exp(V*dt) 做 selection (Eq.23)，代数 -> infinity 时粒子经验分布
弱收敛到不变测度 eta*_lambda —— 这就是 DeepParticle 要学的目标分布。运行估计量 (Eq.24)::

    mu_T(lambda) = kappa + (1/T) * sum_{t=1..T*rn} log(w_t),   w_t = mean_i w^i_t
    c_T(lambda)  = mu_T(lambda) / lambda

论文明确给出的参数
------------------
    空间域        T^2 = [0,2pi)^2         强迫周期  归一化为 1
    (lambda, theta) = (2.0, 1.0)          dt = 2^-8  =>  rn = 256 substeps/周期
    M = 2e4 粒子/扩散水平                 长时参考  T = 4096 代
    冷启动        T^2 上的均匀分布
    训练 sigma    Eq.25  l = -2 - 0.25*(k-1), k=1..8   -> [-2, -3.75]
    评测 sigma    Eq.26  同式 k=1..11                  -> [-2, -4.5]，后 3 个是外推
    特征值        c_T 在 dyadic 代数 T = 1,2,4,... 处记录 (Fig.7)

两处与论文字面不一致的地方，都是有意为之
-----------------------------------------
  1. 重采样频率。Eq.23 字面写的是 "multinomial resampling ... in each substep"，
     但每个 substep 都选择会让谱系坍缩。M=2e4, T=4096 的对照跑（72x72 直方图峰值）:

         sigma      每子步    每代    论文 Fig.5 的 colorbar 上限
         2^-3.75      104      39            40
         2^-4          56      40            35

     每子步那版在 sigma=2^-3.75 下长出一块论文里根本没有的饱和斑（空箱 69%），
     每代一次则逐点对上 Fig.5 的参考列。故默认每代一次，--resample-every 1 复现字面版。
     注意 Eq.24 察觉不到这个问题：两版特征值只差 0.2%（0.8825 vs 0.8843），
     能区分的只有不变测度本身。

  2. 主特征值常数项。Eq.19 的势含 +1 (= r = f'(0))，但 Eq.24 写的是 mu_T = kappa + ...。
     两者相差 (1-kappa)/lambda。实测只有 kappa 那版能对上论文数字（r=1 那版在
     sigma=2^-4.5 给 1.2055，论文是 0.781），故按 Eq.24 原文取 kappa。

复现校验（M=2e4, T=4096, 每代重采样）
--------------------------------------
论文 Fig.7 没写用的是哪个 sigma；拿冷启动曲线逐点比对可以认出是 sigma=2^-4.5：

    T          1      2      4      8     16     32     64    128    long-run
    论文    0.497  0.662  0.717  0.751  0.763  0.776  0.778  0.780     0.781
    本实现  0.504  0.648  0.700  0.733  0.758  0.771  0.778  0.781     0.7841

用法
----
    python kpp_common.py                      # Eq.26 的 11 个 sigma，各 T=4096
    python kpp_common.py --only-log2sigma -4.5
    python kpp_common.py --particles 4000 --generations 8 --cache ''   # 快速自检
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi


# =============================================================================
# 1. 配置与参数网格
# =============================================================================

@dataclass
class IPMConfig:
    """4.3 节 Feynman-Kac 粒子系统的全部参数（默认值就是论文的设定）。

    lam             lambda = 2.0。同时进 twisted 漂移 2*kappa*lambda*e1 和势
                    kappa*lambda^2 + lambda*(v.e1)，两处必须用同一个值。
                    前沿速度 c* = inf_lambda mu/lambda 要扫多个 lambda，论文固定在 2.0 没扫。
    theta           速度场时间周期扰动的幅度，论文取 1.0。theta=0 退化为定常胞状流。
    log2dt          dt = 2^-8 => rn = 256 substeps/周期。必须取负整数，否则 rn*dt != 1，
                    每代的实际时长偏离 1，时间周期会逐代漂移。
    M               20000 粒子/扩散水平。经验测度的统计误差 ~ O(M^-1/2)。
    T               代数（= 物理时间，因为周期归一化成 1）。长时参考用 4096。
    seed            随机种子，同时决定初始均匀点和全部布朗增量。
    resample_every  每几个 substep 选择一次。None -> rn（每代一次，默认）。
                    Eq.23 字面是每子步一次，见模块 docstring 第 1 点。
    """
    lam: float = 2.0
    theta: float = 1.0
    log2dt: float = -8.0
    M: int = 20000
    T: int = 4096
    seed: int = 666
    resample_every: Optional[int] = None

    @property
    def dt(self) -> float:
        return 2.0 ** self.log2dt

    @property
    def rn(self) -> int:
        """每个强迫周期的 substep 数 = 1/dt，论文的 r_n = 256。"""
        return int(round(2.0 ** (-self.log2dt)))

    @property
    def re(self) -> int:
        """实际生效的重采样间隔。"""
        return self.rn if self.resample_every is None else int(self.resample_every)


def training_log2sigma() -> List[float]:
    """[TSD] Eq.25：训练用的 8 个水平，log2(sigma) = -2, -2.25, ..., -3.75。"""
    return [-2.0 - 0.25 * k for k in range(8)]


def eval_log2sigma() -> List[float]:
    """[TSD] Eq.26：评测用的 11 个水平，log2(sigma) = -2.0 ... -4.5。

    后 3 个（-4.00, -4.25, -4.50）超出训练区间，是论文用来测外推的点。
    """
    return [-2.0 - 0.25 * k for k in range(11)]


def dyadic_generations(t_max: int) -> List[int]:
    """[TSD] 4.3 节要求记录 c_T 的代数：T = 1, 2, 4, ... <= t_max，末尾补上 t_max。"""
    out, g = [], 1
    while g <= t_max:
        out.append(g)
        g *= 2
    if out and out[-1] != t_max:
        out.append(t_max)
    return out


# =============================================================================
# 2. 主循环
# =============================================================================

def kpp2d_fk(log2sigma: float, cfg: IPMConfig,
             record_generations: Optional[Sequence[int]] = None,
             progress_every: int = 0) -> Dict[str, object]:
    """跑 2D 时变胞状流下的 Feynman-Kac 粒子系统（遗传交互粒子法）。

    返回 dict
    ---------
      q0          (M,2) float32  初始粒子云（T^2 上均匀，即源分布 mu 的样本）
      qT          (M,2) float32  第 cfg.T 代结束时的粒子云（不变测度 nu(sigma) 的样本）
      trace       [(T, mu_T, c_T), ...]  在 record_generations 处的运行估计 (Eq.24)
      mu_hat/c_hat    末代估计量，c = mu/lambda
      log_w_mean  (1/T) sum log w_t，不含常数项
      ess_mean    重采样前 ESS 的均值，用于诊断退化
      n_unique    qT 中互不相同的粒子数，用于诊断谱系坍缩
      seconds     墙钟耗时

    !! q0[i] 与 qT[i] 不是同一条轨迹：重采样会复制/丢弃粒子，索引对应关系早就断了。
       这对 DeepParticle 无害——它的损失通过 transition matrix 自己配对，本来就不需要
       成对数据。但如果下游代码把 (q0[i], qT[i]) 当监督对用，结果会完全错误。

    数值流程（严格对应论文）
    ------------------------
      对每个 substep j = 0..rn-1（在每一代内重复）：
        tc  = theta*cos(2*pi*t_j)                                       t_j = j*dt
        q  <- q + (2*kappa*lambda*e1 + v(q,t_j))*dt + sqrt(2*kappa*dt)*N(0,I)   [Eq.20]
        q  <- q mod 2*pi                                                 T^2 上折叠
        V   = kappa*lambda^2 + lambda*v1  在**移动后**的位置求值          [Eq.22]
        累加 V*dt；每 resample_every 步按 exp(累积值) 做 multinomial 重采样  [Eq.23]
    """
    t_start = time.perf_counter()
    rng = np.random.default_rng(int(cfg.seed))

    lam, th = float(cfg.lam), float(cfg.theta)
    dt, rn, nn, M = cfg.dt, cfg.rn, int(cfg.T), int(cfg.M)
    resample_every = cfg.re
    if resample_every <= 0:
        raise ValueError('resample_every 必须是正整数')
    if abs(rn * dt - 1.0) > 1e-12:
        warnings.warn(f'1/dt = {1.0/dt} 不是整数，每代实际时长 {rn*dt} != 1，周期会漂移。')

    kappa = 2.0 ** log2sigma                    # 论文的 sigma 就是扩散常数 kappa
    sd = math.sqrt(2.0 * kappa * dt)            # sqrt(2*kappa)*dW，dW ~ N(0, dt)
    drift_q1 = 2.0 * kappa * lam                # twisted 项，e1 = (1,0) 故只加在 q1 上
    v_const = kappa * lam * lam                 # 势里的 kappa*lambda^2

    # 时间因子每代完全重复，预先算好 rn 个值，省掉 1e6 次三角函数。
    # 论文 IPM 的 fitness 用 c(T - i*dt, .)（时间反演，源自 Feynman-Kac 的 c(t-s, X_s)）。
    # 这里用正向时间：cos 是偶函数且周期恰为 1，cos(2*pi*(1-t)) = cos(2*pi*t)，两者恒等。
    # ==> 本例中正/反向等价，不是 bug；但换成 sin 或任何非对称时间剖面就必须显式反演
    #     （3D 的 Eq.29 用的正是 sin，那边就有这个区别）。
    thetacos = th * np.cos(TWO_PI * np.arange(rn) * dt)

    q = rng.uniform(0.0, TWO_PI, size=(M, 2))   # 冷启动：T^2 上均匀
    q0 = q.astype(np.float32).copy()

    if record_generations is None:
        record_generations = dyadic_generations(nn)
    record_set = {int(g) for g in record_generations if 1 <= int(g) <= nn}

    log_w_sum = 0.0                 # Eq.24 的 sum_t log(mean_i w^i_t)
    logw = np.zeros(M)              # resample_every > 1 时跨 substep 累积
    ess_sum, ess_cnt = 0.0, 0
    trace: List[Tuple[int, float, float]] = []

    for g in range(nn):
        for j in range(rn):
            tc = thetacos[j]

            # ---- 一步 Euler-Maruyama (Eq.20) --------------------------------
            # 速度场是 Eq.21，逐字对应：
            #     v1 = -cos(q2) - theta*cos(2*pi*t)*sin(q1)
            #     v2 =  cos(q1) + theta*cos(2*pi*t)*sin(q2)
            # 定常部分 (-cos q2, cos q1) 的流函数是 psi = sin q1 + sin q2，散度为 0；
            # 但 theta 扰动部分 (-sin q1, +sin q2) 是纯梯度场 -grad(cos q2 - cos q1)，
            # div v = theta*cos(2*pi*t)*(cos q2 - cos q1)，实测 max|div v| = 2.0 (theta=1)。
            # 也就是说 Eq.21 并不满足 Eq.17 "incompressible velocity field" 的前提。
            # 代码忠实于 Eq.21——要复现 4.3 节就该这么写，只是别把它当不可压流来解释。
            # 两个分量必须都用**本步开始时**的坐标，否则就成了 Gauss-Seidel 而非 EM。
            c1, c2 = np.cos(q[:, 0]), np.cos(q[:, 1])
            s1, s2 = np.sin(q[:, 0]), np.sin(q[:, 1])
            q[:, 0] += (drift_q1 - c2 - s1 * tc) * dt
            q[:, 1] += (c1 + s2 * tc) * dt
            q += sd * rng.standard_normal((M, 2))
            q %= TWO_PI

            # ---- fitness (Eq.22-23)，在移动后的位置上求值 -------------------
            # V = kappa*lambda^2 + lambda*v1 = v_const - lambda*(cos q2 + tc*sin q1)
            logw += (v_const - lam * (np.cos(q[:, 1]) + tc * np.sin(q[:, 0]))) * dt

            # ---- selection --------------------------------------------------
            if (j + 1) % resample_every == 0:
                mx = logw.max()
                w = np.exp(logw - mx)
                sw = w.sum()
                # Eq.24 的 log w_t = log(mean_i exp(V_i*dt))，用 logsumexp 防上溢
                log_w_sum += mx + math.log(sw / M)
                p = w / sw
                ess_sum += 1.0 / float(np.sum(p * p)); ess_cnt += 1
                cdf = np.cumsum(p)
                cdf[-1] = 1.0               # 消除 cumsum 的累积舍入，防 searchsorted 越界
                idx = np.searchsorted(cdf, rng.random(M), side='right')
                np.clip(idx, 0, M - 1, out=idx)
                q = q[idx]
                logw.fill(0.0)

        gen = g + 1
        if gen in record_set:
            mu = kappa + log_w_sum / gen            # Eq.24
            trace.append((gen, mu, mu / lam))
        if progress_every and gen % progress_every == 0:
            el = time.perf_counter() - t_start
            print(f'    gen {gen}/{nn}  {el/gen*1000:.0f} ms/gen  '
                  f'elapsed {el/60:.1f} min  ETA {(nn-gen)*el/gen/60:.1f} min', flush=True)

    if rn % resample_every != 0:
        warnings.warn('resample_every 不整除 rn，末段累积的权重被丢弃，估计有偏。')

    log_w_mean = log_w_sum / nn
    mu_hat = kappa + log_w_mean
    qT = q.astype(np.float32)
    return dict(
        q0=q0, qT=qT, trace=trace,
        mu_hat=float(mu_hat), c_hat=float(mu_hat / lam),
        log_w_mean=float(log_w_mean),
        ess_mean=float(ess_sum / max(ess_cnt, 1)),
        n_unique=int(np.unique(qT, axis=0).shape[0]),
        log2sigma=float(log2sigma), kappa=float(kappa),
        seconds=float(time.perf_counter() - t_start),
    )


# =============================================================================
# 3. 磁盘缓存
# =============================================================================

def _tag(x) -> str:
    return str(x).replace('.', 'p').replace('-', 'm')


def _cache_path(cache_dir: str, name: str, log2sigma: float, cfg: IPMConfig) -> Path:
    """缓存文件名。所有会影响数值的参数都要进名字，否则不同设置会互相覆盖。"""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    fn = (f"{name}_l2s{_tag(log2sigma)}_M{cfg.M}_T{cfg.T}_seed{cfg.seed}"
          f"_lam{_tag(cfg.lam)}_th{_tag(cfg.theta)}_dt{_tag(cfg.log2dt)}"
          f"_re{cfg.re}")
    return d / (fn + '.npz')


def load_or_make_2d(log2sigma: float, cfg: IPMConfig,
                    cache_dir: Optional[str] = 'kpp2d_cache',
                    name: str = 'ref',
                    progress_every: int = 0) -> Dict[str, object]:
    """带缓存的 kpp2d_fk。缓存键涵盖全部会影响数值的参数。"""
    p = _cache_path(cache_dir, name, log2sigma, cfg) if cache_dir else None
    if p is not None and p.exists():
        data = np.load(p)
        meta = json.loads(str(data['meta']))
        if meta.get('cfg') == asdict(cfg) and \
                abs(meta.get('log2sigma', 1e9) - log2sigma) < 1e-12:
            out = dict(meta)
            out.pop('cfg', None)
            out['q0'], out['qT'] = data['q0'], data['qT']
            out['trace'] = [tuple(t) for t in meta.get('trace', [])]
            return out
        warnings.warn(f'[cache mismatch] recomputing {p.name}')

    res = kpp2d_fk(log2sigma, cfg, progress_every=progress_every)
    if p is not None:
        meta = {k: v for k, v in res.items() if k not in ('q0', 'qT')}
        meta['cfg'] = asdict(cfg)
        np.savez_compressed(p, q0=res['q0'], qT=res['qT'], meta=json.dumps(meta))
    return res


# =============================================================================
# 4. 命令行入口
# =============================================================================

def _fmt_secs(s: float) -> str:
    return f'{s/60:.1f} min' if s < 3600 else f'{s/3600:.2f} h'


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='生成 two-step diffusion 4.3 节 (2D KPP + 时变胞状流) 的 FK 参考数据')
    ap.add_argument('--particles', type=int, default=20000, help='M，论文用 2e4')
    ap.add_argument('--generations', type=int, default=4096, help='T，论文的长时参考用 4096')
    ap.add_argument('--seed', type=int, default=666)
    ap.add_argument('--cache', default='kpp2d_cache', help="缓存目录，传 '' 关闭")
    ap.add_argument('--progress', type=int, default=64, help='每几代打印一次进度，0 = 不打印')
    ap.add_argument('--only-log2sigma', nargs='+', default=None, metavar='L2S',
                    help='只跑指定的 log2(sigma)，如 -4.5。各 sigma 完全独立、缓存文件也分开，'
                         '多核机器上每个 sigma 开一个进程是安全的加速方式')
    ap.add_argument('--resample-every', dest='resample_every', default='gen',
                    metavar='N|gen',
                    help="每几个 substep 选择一次。默认 'gen' = 每代一次 (= rn = 256)。"
                         'Eq.23 字面是每子步一次 (传 1)，但那样谱系会坍缩，'
                         '见模块 docstring 第 1 点')
    args = ap.parse_args()

    cfg = IPMConfig(M=args.particles, T=args.generations, seed=args.seed)
    if args.resample_every != 'gen':
        cfg = IPMConfig(**{**asdict(cfg), 'resample_every': int(args.resample_every)})

    levels = eval_log2sigma()
    if args.only_log2sigma:
        want = [float(x) for x in args.only_log2sigma]
        missing = [w for w in want if not any(abs(l - w) < 1e-9 for l in levels)]
        if missing:
            raise SystemExit(f'{missing} 不在 Eq.26 的水平表里: {levels}')
        levels = [l for l in levels if any(abs(l - w) < 1e-9 for w in want)]

    print(f'=== 4.3 节 FK 参考点云: {len(levels)} 个 sigma, M={cfg.M}, T={cfg.T}, '
          f'rn={cfg.rn}, resample_every={cfg.re}, seed={cfg.seed} ===', flush=True)
    t0 = time.perf_counter()
    train = training_log2sigma()
    for i, l2s in enumerate(levels):
        kind = 'train' if any(abs(l2s - t) < 1e-9 for t in train) else 'extrap'
        print(f'[{i+1}/{len(levels)}] log2sigma = {l2s:.2f}  '
              f'(-log2 sigma = {-l2s:.2f}, {kind})', flush=True)
        res = load_or_make_2d(l2s, cfg, cache_dir=args.cache or None,
                              progress_every=args.progress)
        print(f'    c_hat = {res["c_hat"]:.6f}   ESS = {res["ess_mean"]:.0f}/{cfg.M}   '
              f'unique = {res["n_unique"]}/{cfg.M}   {_fmt_secs(res["seconds"])}', flush=True)
        print('    trace: ' + ', '.join(f'T={t}:{c:.4f}' for t, _, c in res['trace'][:8]),
              flush=True)
    print(f'total {_fmt_secs(time.perf_counter() - t0)}')


if __name__ == '__main__':
    main()
