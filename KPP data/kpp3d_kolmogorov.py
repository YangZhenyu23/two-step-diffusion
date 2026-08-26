# -*- coding: utf-8 -*-
"""
生成 two-step diffusion 论文 4.4 节的数据：
"3D KPP Front Speed Experiment with time-dependent Kolmogorov flow"

2D 的 4.3 节在 kpp_common.py，两个文件互不依赖。

论文
----
[TSD] Shen, Wang, Xin, Zhang, "Two-Step Diffusion: Fast Sampling and Reliable
      Prediction for 3D Keller-Segel and KPP Equations in Fluid Flows",
      arXiv:2601.20024v1.（本目录上一级的 two-step diffusion.pdf）

4.4 节要什么
------------
3D KPP 方程 (Eq.28)::

    u_t + v.grad(u) = kappa*Lap(u) + r*u(1-u)

3D 时变 Kolmogorov 流 (Eq.29)::

    v(x,t) = ( sin(x3 + sin(2*pi*t)),
               sin(x1 + sin(2*pi*t)),
               sin(x2 + sin(2*pi*t)) )

三个分量各自只依赖"别的"坐标，所以 dv1/dx1 = dv2/dx2 = dv3/dx3 = 0，
div v = 0 对任意 t 严格成立，是真正的不可压流。
（对比 4.3 节 Eq.21 的 2D 流场，那个 theta 项其实是有散度的。）

Feynman-Kac 粒子系统（把 Eq.19/20/22/23 推广到 3D，e1 = (1,0,0)）::

    twisted SDE   dX = (2*kappa*lambda*e1 + v(X,t)) dt + sqrt(2*kappa) dW
    势            V_lambda(x,t) = kappa*lambda^2 + lambda*(v.e1)
                                = kappa*lambda^2 + lambda*sin(x3 + sin(2*pi*t))
    权重          w^i_j = exp( V_lambda(X^i_j, t_j) * dt )
    主特征值      mu_T(lambda) = kappa + (1/T) * sum_{t=1..T*rn} log(w_t),  w_t = mean_i w^i_t
                  c_T(lambda)  = mu_T(lambda) / lambda

论文明确给出的参数
------------------
    空间域        T^3 = [0,2pi)^3         强迫周期  归一化为 1
    lambda = 2.0（4.3 节 "we fix (lambda, theta) = (2.0, 1.0)"；3D 流没有 theta）
    dt = 2^-8  =>  rn = 256 substeps/周期
    M = 2e4 粒子/扩散水平                 长时参考  T = 4096 代
    冷启动        T^3 上的均匀分布
    训练 sigma    -log2(sigma) <= 3.75，即 8 个水平 (Eq.25)
    评测 sigma    -log2(sigma) 到 4.50，即 11 个水平 (Eq.26)
    特征值        c_T 在 dyadic 代数 T = 1,2,4,... 处记录 (Fig.10 的表)

三处与论文字面不一致的地方，都是有意为之
-----------------------------------------
  1. 重采样频率。Eq.23 字面写的是 "multinomial resampling ... in each substep"。
     问题在于每步权重几乎均匀（V*dt ~ 1e-2，实测每步 ESS 满格 M/M），此时
     multinomial 重采样不带来任何选择效果，只是一个 Wright-Fisher 随机复制过程：
     每次调用就地杀掉约 1-1/e = 37% 的粒子（实测稳定在 63.3% 互异，理论值 63.2%），
     纯粹往经验测度里灌方差。累计上百万次之后，经验测度已经随种子漂到很远。

     实测 sigma=2^-4.5, M=2e4, T=4096:
         每子步一次   48 箱峰值 219、67% 空箱，c_hat = 0.6097
         每代一次     48 箱峰值  56、26% 空箱，c_hat = 0.6140
     论文 Fig.8 的峰值约 37、表里是 0.6139-0.6145 —— 每代一次才对得上。
     故默认每代一次，--resample-every 1 复现字面版。

  2. 主特征值常数项。Eq.19 的势含 +1 (= r = f'(0))，但 Eq.24 写的是 mu_T = kappa + ...。
     两者相差 (1-kappa)/lambda，在 kappa = 2^-4.5、lambda = 2 时约 0.478。
     实测只有 kappa 那版能对上论文：本实现在 -log2(sigma) = 4.25/4.50 得 0.617/0.626，
     论文 Fig.10 收敛到 0.614；换成 r = 1 则一律在 1.10 附近。故按 Eq.24 原文取 kappa。

  3. 时间方向。Eq.20/23 用正向时间 t_j，严格的 Feynman-Kac 是 c(t-s, X_s)（时间反演）。
     4.3 节那个 cos(2*pi*t) 是偶函数所以两者恒等，但这里是 sin(2*pi*t)，反演会变号，
     两种约定给出的不变测度不同（特征值不变）。默认照论文取正向。

  另注：可视化论文 Fig.8 时会发现质量沿 x3 差了 pi。势 V = kappa*lambda^2 +
  lambda*sin(x3+ph) 的极大在 x3 ~ pi/2，我们的质量就落在那里；论文的质量却压在
  x3 ~ 3*pi/2（势的极小处），说明论文那侧 lambda*v.e1 的符号相反。该差异不改变
  特征值（两边都是 0.614），只把不变测度沿 x3 挪了 pi。这里保留数学上正确的符号。

用法
----
    python kpp3d_kolmogorov.py                      # Eq.26 的 11 个 sigma，各 T=4096
    python kpp3d_kolmogorov.py --only-log2sigma -4.5
    python kpp3d_kolmogorov.py --particles 4000 --generations 8 --cache ''   # 快速自检
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
class FK3DConfig:
    """4.4 节 Feynman-Kac 粒子系统的全部参数（默认值就是论文的设定）。

    lam             lambda = 2.0。同时进 twisted 漂移 2*kappa*lambda*e1 和势
                    kappa*lambda^2 + lambda*(v.e1)，两处必须用同一个值。
                    前沿速度 c* = inf_lambda mu/lambda 要扫多个 lambda，论文固定在 2.0 没扫。
    log2dt          dt = 2^-8 => rn = 256 substeps/周期。必须取负整数，否则 rn*dt != 1，
                    每代的实际时长偏离 1，时间周期会逐代漂移。
    M               20000 粒子/扩散水平。
    T               代数（= 物理时间，因为周期归一化成 1）。长时参考用 4096。
    seed            随机种子，同时决定初始均匀点和全部布朗增量。
    resample_every  每几个 substep 选择一次。None -> rn（每代一次，默认）。
                    Eq.23 字面是每子步一次，见模块 docstring 第 1 点。
    """
    lam: float = 2.0
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
    """[TSD] Fig.10 的记录点：T = 1, 2, 4, ... <= t_max，末尾补上 t_max。"""
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

def kpp3d_fk(log2sigma: float, cfg: FK3DConfig,
             record_generations: Optional[Sequence[int]] = None,
             progress_every: int = 0) -> Dict[str, object]:
    """跑 3D 时变 Kolmogorov 流下的 Feynman-Kac 粒子系统。

    返回 dict
    ---------
      x0          (M,3) float32  初始粒子云（T^3 上均匀，论文的 cold start / reference group）
      xT          (M,3) float32  第 cfg.T 代结束时的粒子云（不变测度 eta*_lambda 的样本）
      trace       [(T, mu_T, c_T), ...]  在 record_generations 处的运行估计 (Eq.24)
      mu_hat/c_hat    末代估计量，c = mu/lambda
      log_w_mean  (1/T) sum log w_t，不含常数项
      ess_mean    重采样前 ESS 的均值，用于诊断退化
      seconds     墙钟耗时

    数值流程（严格对应论文）
    ------------------------
      对每个 substep j = 0..rn-1（在每一代内重复）：
        ph  = sin(2*pi*t_j)                                              t_j = j*dt
        v   = ( sin(x3+ph), sin(x1+ph), sin(x2+ph) )                     [Eq.29]
        X  <- X + (2*kappa*lambda*e1 + v)*dt + sqrt(2*kappa*dt)*N(0,I)   [Eq.20]
        X  <- X mod 2*pi                                                 T^3 上折叠
        V   = kappa*lambda^2 + lambda*sin(x3+ph)  在**移动后**的位置求值  [Eq.22]
        累加 V*dt；每 resample_every 步按 exp(累积值) 做 multinomial 重采样  [Eq.23]
    """
    t_start = time.perf_counter()
    rng = np.random.default_rng(int(cfg.seed))

    lam = float(cfg.lam)
    dt, rn, nn, M = cfg.dt, cfg.rn, int(cfg.T), int(cfg.M)
    resample_every = cfg.re
    if resample_every <= 0:
        raise ValueError('resample_every 必须是正整数')
    if abs(rn * dt - 1.0) > 1e-12:
        warnings.warn(f'1/dt = {1.0/dt} 不是整数，每代实际时长 {rn*dt} != 1，周期会漂移。')

    kappa = 2.0 ** log2sigma                    # 论文的 sigma 就是扩散常数 kappa
    sd = math.sqrt(2.0 * kappa * dt)            # sqrt(2*kappa)*dW，dW ~ N(0, dt)
    drift_x1 = 2.0 * kappa * lam                # twisted 项，e1 = (1,0,0) 故只加在 x1 上
    v_const = kappa * lam * lam                 # 势里的 kappa*lambda^2

    # 相位 sin(2*pi*t_j) 每代完全重复，预先算好 rn 个值，省掉上百万次三角函数
    phases = np.sin(TWO_PI * np.arange(rn) * dt)

    X = rng.uniform(0.0, TWO_PI, size=(M, 3))   # 冷启动：T^3 上均匀
    x0 = X.astype(np.float32).copy()

    if record_generations is None:
        record_generations = dyadic_generations(nn)
    record_set = {int(g) for g in record_generations if 1 <= int(g) <= nn}

    log_w_sum = 0.0                 # Eq.24 的 sum_t log(mean_i w^i_t)
    logw = np.zeros(M)              # resample_every > 1 时跨 substep 累积
    ess_sum, ess_cnt = 0.0, 0
    trace: List[Tuple[int, float, float]] = []

    buf = np.empty((M, 3))
    for g in range(nn):
        for j in range(rn):
            ph = phases[j]

            # ---- 一步 Euler-Maruyama (Eq.20) --------------------------------
            # Eq.29 的分量是**错位**的：v1 看 x3、v2 看 x1、v3 看 x2。
            # 三个分量都必须用同一时刻的旧坐标，所以先把 sin(X+ph) 整体算进 buf 再取用。
            # buf 与 X 不共享内存，下面原地改 X 不会污染 buf，故 sx* 取视图即可。
            np.add(X, ph, out=buf)
            np.sin(buf, out=buf)
            sx1, sx2, sx3 = buf[:, 0], buf[:, 1], buf[:, 2]   # sin(x_k + ph), k=1,2,3
            X[:, 0] += (drift_x1 + sx3) * dt     # v1 = sin(x3+ph)，外加 twisted 项
            X[:, 1] += sx1 * dt                  # v2 = sin(x1+ph)
            X[:, 2] += sx2 * dt                  # v3 = sin(x2+ph)
            X += sd * rng.standard_normal((M, 3))
            X %= TWO_PI

            # ---- fitness (Eq.22-23)，在移动后的位置上求值 -------------------
            logw += (v_const + lam * np.sin(X[:, 2] + ph)) * dt

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
                # Multinomial (Eq.23) 用 M 次独立逆 CDF 抽样实现，与 rng.multinomial(M, p)
                # 同分布但快得多：numpy 的 multinomial 对 k 个类别要跑 k 次条件二项抽样，
                # k = M = 20000 时非常慢；逆 CDF 只要 O(M log M)。
                idx = np.searchsorted(cdf, rng.random(M), side='right')
                np.clip(idx, 0, M - 1, out=idx)
                X = X[idx]
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
    return dict(
        x0=x0, xT=X.astype(np.float32), trace=trace,
        mu_hat=float(mu_hat), c_hat=float(mu_hat / lam),
        log_w_mean=float(log_w_mean),
        ess_mean=float(ess_sum / max(ess_cnt, 1)),
        log2sigma=float(log2sigma), kappa=float(kappa),
        seconds=float(time.perf_counter() - t_start),
    )


# =============================================================================
# 3. 磁盘缓存
# =============================================================================

def _tag(x) -> str:
    return str(x).replace('.', 'p').replace('-', 'm')


def _cache_path(cache_dir: str, name: str, log2sigma: float, cfg: FK3DConfig) -> Path:
    """缓存文件名。所有会影响数值的参数都要进名字，否则不同设置会互相覆盖。"""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    fn = (f"{name}_l2s{_tag(log2sigma)}_M{cfg.M}_T{cfg.T}_seed{cfg.seed}"
          f"_lam{_tag(cfg.lam)}_dt{_tag(cfg.log2dt)}_re{cfg.re}")
    return d / (fn + '.npz')


def load_or_make_3d(log2sigma: float, cfg: FK3DConfig,
                    cache_dir: Optional[str] = 'kpp3d_cache',
                    name: str = 'ref',
                    progress_every: int = 0) -> Dict[str, object]:
    """带缓存的 kpp3d_fk。缓存键涵盖全部会影响数值的参数。"""
    p = _cache_path(cache_dir, name, log2sigma, cfg) if cache_dir else None
    if p is not None and p.exists():
        data = np.load(p)
        meta = json.loads(str(data['meta']))
        if meta.get('cfg') == asdict(cfg) and \
                abs(meta.get('log2sigma', 1e9) - log2sigma) < 1e-12:
            out = dict(meta)
            out.pop('cfg', None)
            out['x0'], out['xT'] = data['x0'], data['xT']
            out['trace'] = [tuple(t) for t in meta.get('trace', [])]
            return out
        warnings.warn(f'[cache mismatch] recomputing {p.name}')

    res = kpp3d_fk(log2sigma, cfg, progress_every=progress_every)
    if p is not None:
        meta = {k: v for k, v in res.items() if k not in ('x0', 'xT')}
        meta['cfg'] = asdict(cfg)
        np.savez_compressed(p, x0=res['x0'], xT=res['xT'], meta=json.dumps(meta))
    return res


# =============================================================================
# 4. 命令行入口
# =============================================================================

def _fmt_secs(s: float) -> str:
    return f'{s/60:.1f} min' if s < 3600 else f'{s/3600:.2f} h'


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='生成 two-step diffusion 4.4 节 (3D KPP + 时变 Kolmogorov 流) 的 FK 参考数据')
    ap.add_argument('--particles', type=int, default=20000, help='M，论文用 2e4')
    ap.add_argument('--generations', type=int, default=4096, help='T，论文的长时参考用 4096')
    ap.add_argument('--seed', type=int, default=666)
    ap.add_argument('--cache', default='kpp3d_cache', help="缓存目录，传 '' 关闭")
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

    cfg = FK3DConfig(M=args.particles, T=args.generations, seed=args.seed)
    if args.resample_every != 'gen':
        cfg = FK3DConfig(**{**asdict(cfg), 'resample_every': int(args.resample_every)})

    levels = eval_log2sigma()
    if args.only_log2sigma:
        want = [float(x) for x in args.only_log2sigma]
        missing = [w for w in want if not any(abs(l - w) < 1e-9 for l in levels)]
        if missing:
            raise SystemExit(f'{missing} 不在 Eq.26 的水平表里: {levels}')
        levels = [l for l in levels if any(abs(l - w) < 1e-9 for w in want)]

    print(f'=== 4.4 节 FK 参考点云: {len(levels)} 个 sigma, M={cfg.M}, T={cfg.T}, '
          f'rn={cfg.rn}, resample_every={cfg.re}, seed={cfg.seed} ===', flush=True)
    t0 = time.perf_counter()
    train = training_log2sigma()
    for i, l2s in enumerate(levels):
        kind = 'train' if any(abs(l2s - t) < 1e-9 for t in train) else 'extrap'
        print(f'[{i+1}/{len(levels)}] log2sigma = {l2s:.2f}  '
              f'(-log2 sigma = {-l2s:.2f}, {kind})', flush=True)
        res = load_or_make_3d(l2s, cfg, cache_dir=args.cache or None,
                              progress_every=args.progress)
        print(f'    c_hat = {res["c_hat"]:.6f}   ESS = {res["ess_mean"]:.0f}/{cfg.M}   '
              f'{_fmt_secs(res["seconds"])}', flush=True)
        print('    trace: ' + ', '.join(f'T={t}:{c:.4f}' for t, _, c in res['trace'][:8]),
              flush=True)
    print(f'total {_fmt_secs(time.perf_counter() - t0)}')


if __name__ == '__main__':
    main()
