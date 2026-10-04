# 平行束 CT 滤波反投影 + 双能材料分解服务

纯后端实现：Python 3.10 + FastAPI 0.115.12 + NumPy 2.2.6（滤波反投影 FBP 与双能分解全部用 NumPy 手写，不调用任何现成重建/求解函数）。

## 运行

    .venv/bin/python -m uvicorn ctrecon.app:app --host 127.0.0.1 --port 8000

## 接口一览

- `POST /reconstruct`：单能 FBP 重建（原接口，保持不变）。
- `POST /decompose`：双能材料分解 + 矩形 ROI 局部用量核算。
- `GET /health`：健康检查。

## /reconstruct 输入格式

`multipart/form-data`：

| 字段 | 说明 |
| --- | --- |
| `file` | NPZ，含三个数组 |
| `detector_spacing_mm` | 探测器单元间距（毫米，>0） |
| `center_index` | 旋转中心对应的小数探测器索引 |
| `output_size` | 输出图像边长（像素，1–256） |
| `pixel_spacing_mm` | 输出像素间距（毫米，>0） |
| `filter` | `ram-lak`（默认）或 `hann` |

NPZ 数组：

- `intensity`：二维数组，形状 **角度 × 探测器**，即 `(n_angles, n_detectors)`。
- `dark`、`flat`：一维数组，长度等于探测器宽度。

角度约定：**从 0 开始、等间距覆盖 180°、不含终点**，即
`theta_k = k * pi / n_angles`（k = 0 … n_angles−1）。

限制与校验（全部返回 HTTP 422）：

- 角度数 2–360，探测器数 2–512，输出边长 1–256。
- 拒绝对象数组（`allow_pickle=False`）、形状不匹配、含 NaN/Inf 的数组。
- 拒绝非有限或非正的间距、非有限中心索引、未知滤波器。
- ZIP/NPZ 解压后总大小上限 64 MiB。

## 标定（线积分）

逐探测器执行

    T(theta, i) = (I(theta, i) - dark[i]) / (flat[i] - dark[i])
    p(theta, i) = -ln(T)

- 要求每个探测器 `flat > dark`，否则拒绝。
- 透射率必须有限且严格为正，否则拒绝。
- 透射率 > 1 时**保留负线积分**，不裁剪，也不把坏值当零。

## 重建（FBP）

- 滤波器在 FFT 网格上按**真实探测器间距**构造频率 `f = k / (N * d)`（cycles/mm），Ram-Lak 响应 `|f|`（截止于探测器 Nyquist `1/(2d)`）；Hann 在 Ram-Lak 上乘 Hann 窗。
- 每行 sinogram FFT **补零到至少 2 倍探测器长度**（取 2 的幂），避免循环卷积混叠。
- 物理频率网格下的 IFFT 已携带正确的 `1/d` 测度，因此滤波结果**不再额外乘探测器间距**（此前版本的额外 `d` 缩放会使重建值随 `d` 变化、单位错误，已修复；现有测试覆盖 `d = 0.25 / 0.5 / 1.0 mm` 下重建值不变）。
- 反投影按 `t = x*cos(theta) + y*sin(theta)` 线性插值，探测器范围外取 0；按角度步长 `pi/n_angles` 积分。输出单位为**每毫米线性衰减系数 mm^-1**。
- 不做逐图归一化；保留负重建值。

### 坐标与平行束覆盖范围

- 图像以**几何中心为原点**：列向右为 +x，行向上为 +y（数组第 0 行对应 +y）。
- 0° 时射线法向为 +x，探测器坐标 `t = (detector_index - center_index) * detector_spacing_mm`。
- 平行束在 0–180°（不含 180°）内等角距采样即可覆盖完整物体：平行射线在 `theta` 与 `theta+180°` 方向的投影等价，因此只需半圆采样。FOV 半径约为旋转中心两侧的有效探测器宽度；偏心或超出探测器轨迹的结构不会被完整采样。

## /reconstruct 输出（ZIP）

响应为 `application/zip`，包含：

- `reconstruction.npy`：float64 二维数组（`numpy.save` 格式），物理单位 mm^-1。
- `preview.png`：8 位灰度预览，按图像 min/max 做**仅用于显示**的线性拉伸，不影响 NPY。
- `metadata.json`：回显参数及图像/sinogram 数值范围与单位。

## /decompose 双能材料分解

`multipart/form-data`：

| 字段 | 说明 |
| --- | --- |
| `low_file` / `high_file` | 低、高能 NPZ（格式同 /reconstruct，**形状必须相同**，已对齐，不做图像配准） |
| `detector_spacing_mm` / `center_index` / `output_size` / `pixel_spacing_mm` / `filter` | 共同几何参数，两个能量共用，校验与限制同上 |
| `materials` | JSON 对象（见下） |
| `slice_thickness_mm` | 截面厚度（毫米，>0） |
| `rois` | JSON 数组，1–8 个矩形 |

`materials` 示例：

    {"names": ["aluminum", "pvc"],
     "mass_attenuation_matrix_mm2_per_mg": [[0.030, 0.012],
                                            [0.012, 0.008]]}

- 两个材料名必须**唯一且非空**。
- 矩阵 2×2，**行为低/高能、列为材料**，单位 mm²/mg，元素必须有限且严格为正。
- 矩阵 2 范数条件数 > 10000 时拒绝（两材料不可辨识），**不做任何正则化**。

`rois` 示例：

    [{"name": "al_disk", "x0": 36, "y0": 46, "x1": 60, "y1": 70}]

- 像素索引，**左上含、右下不含**：`x in [x0, x1)`，`y in [y0, y1)`（x 为列、y 为行）。
- 名称唯一非空；拒绝越界或空区。

### 线性双材料假设

重建得到每能量每毫米衰减图 `mu_e(x, y)`（mm^-1），假设

    mu_e(x, y) = sum_j A[e, j] * rho_j(x, y)

其中 `rho_j` 为材料密度（mg/mm³）。逐像素解**精确二变量非负最小二乘**（枚举内点与两条坐标轴边界候选取最优），使 `||A rho - mu||^2` 最小：

- **不是**把无约束解的负分量截零（截零一般不是 NNLS 最优解）。
- 衰减图中的**负值保留**参与拟合，不裁剪。
- 输出残差 `r_e = (A rho)_e - mu_e`（预测减观测），单位 mm^-1。

ROI 质量按体素积分：

    mass_j = sum_{pixels in ROI} rho_j * pixel_spacing_mm^2 * slice_thickness_mm   (mg)

### /decompose 输出（ZIP）

- `density_<材料名>.npy` ×2：float64 密度图，mg/mm³。
- `residual_low.npy` / `residual_high.npy`：float64 残差图（预测−观测），mm^-1。
- `preview_<材料名>.png` ×2：8 位灰度预览，min/max 拉伸**仅用于显示**，不改动定量 NPY 数据。
- `result.json`：回显参数、材料矩阵与条件数、各 ROI 的两种材料质量（mg）及两能量平均残差。

## 解析示例

### 偏心圆盘（单能）

均匀圆盘（半径 R、衰减 μ、圆心 (cx, cy)）的解析投影为弦长公式：

    p(theta, t) = 2 * mu * sqrt(R^2 - (t - p0)^2),  p0 = cx*cos(theta) + cy*sin(theta)
                 （|t - p0| <= R，否则为 0）

生成示例 NPZ：

    .venv/bin/python examples/offcenter_disk_demo.py

### 两材料混合（双能）

铝盘与 PVC 盘各居一侧，每个能量的线积分为两种材料密度线积分的线性混合
`p_e = sum_j A[e, j] * q_j`（`q_j` 为圆盘弦长公式给出的密度线积分）。生成并打印 curl：

    .venv/bin/python examples/dual_energy_demo.py

输出 `examples/dual_energy_low.npz` / `examples/dual_energy_high.npz`，可直接用于 `POST /decompose`。

## 模块结构

- `ctrecon/io_utils.py`：NPZ 与标量参数校验（两个接口共用）。
- `ctrecon/calibration.py`：暗/平场标定 → 线积分。
- `ctrecon/reconstruct.py`：NumPy 手写 FBP。
- `ctrecon/decomposition.py`：材料/矩阵校验 + 精确 2×2 NNLS 分解。
- `ctrecon/rois.py`：ROI 校验与质量/残差积分。
- `ctrecon/service.py`：跨模块流水线（标定 → 重建 → 分解 → 区域计量）。
- `ctrecon/app.py`：HTTP 层与 ZIP 交付；`ctrecon/preview.py`：NPY/PNG 序列化。
- `ctrecon/synthetic.py`：解析圆盘投影，供示例与测试共用。

## 测试

    .venv/bin/python -m compileall -q ctrecon examples tests
    .venv/bin/python -m pytest -q

