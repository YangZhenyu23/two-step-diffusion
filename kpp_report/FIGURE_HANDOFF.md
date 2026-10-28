# 图版交接说明

本文件用于后续重新训练后替换六张论文图并同步正文数字。

## 锁定图文件

| PDF 编号 | 小节 | label | 文件名 |
|---|---|---|---|
| Figure 5 | 4.3 Invariant measure | `fig:2d-inv` | `fig5_invariant_2d.png` |
| Figure 6 | 4.3 Torus Wasserstein Error | `fig:2d-w2` | `fig6_w2_2d.png` |
| Figure 7 | 4.3 Eigenvalue estimator | `fig:2d-eig` | `fig7_eigen_2d.png` |
| Figure 8 | 4.4 Projected invariant measure | `fig:3d-inv` | `fig8_projections_3d.png` |
| Figure 9 | 4.4 Projected Wasserstein Error | `fig:3d-w2` | `fig9_w2_3d.png` |
| Figure 10 | 4.4 Eigenvalue estimator | `fig:3d-eig` | `fig10_eigen_3d.png` |

替换规则：覆盖同名 PNG，不要修改 `\includegraphics` 或 `\label`。

## 重新生成图片

项目根目录运行：

```powershell
python kpp_report/plot_paper_figures.py --device cuda
```

若没有 CUDA：

```powershell
python kpp_report/plot_paper_figures.py --device cpu
```

## 提取数字

```powershell
python kpp_report/extract_paper_numbers.py
```

数据来源：

- `two_step_2d/runs/kpp2d/full_2d_v2_dpfix2/metrics.json`
- `two_step_3d/runs/kpp3d/full_3d_otgeo_dpfix2/metrics.json`

不要使用 `runs_backup_20260829/`。

## 编译

在 `kpp_report/` 下：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error kpp_sections_4_3_4_4.tex
```

## 正文中需要更新的是

这些数值来自 `metrics.json`，重新训练后要改：

2D：

- cold \(W_2\) 范围
- Meanflow 网格均值
- \(-\log_2\sigma=2.00\) 的 MF / DP
- \(-\log_2\sigma=4.50\) 的 MF / DP
- \(\hat c\)、MF/DP 的 \(T=1\)、\(T=128\)

3D：

- cold 范围
- Meanflow / DP 网格均值
- \(-\log_2\sigma=4.50\) 的 MF / DP
- \(\hat c\)、MF/DP 的 \(T=1\)、\(T=128\)

## 不能改的论文常数

- 2D Meanflow 均值 `0.0769`
- 2D DP 在 \(-\log_2\sigma=4.50\) 为 `0.0456`
- 2D DP warm start \(T=1\) 为 `0.7616`
- 3D Meanflow 均值 `0.1312`
- 3D DP 均值 `0.0294`
- 3D DP warm start \(T=1\) 为 `0.6135`

这些是论文值，不是我们的运行值。

## 警告

Fig. 6/7/9/10 的 PNG 内部已经嵌入了左侧数字表。若只改正文而不重新画图，正文和图中表格会不一致。

## 不要插入

- `fig5_ref.png`
- `fig8_ref.png`
- `fig7_cT_2d_fk.png`
- `fig10_cT_3d_fk.png`
- loss 曲线

## 替换后检查

1. 覆盖六张 PNG。
2. 更新正文数字。
3. 若热图几何明显变化，改写 4.3.2 和 4.4.2 的几何描述。
4. 重新编译。
5. 确认 Fig. 5–7 在 4.3，Fig. 8–10 在 4.4。
