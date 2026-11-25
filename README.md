# pairwise-image-denoise-profile

## 领域

游戏、图形与多媒体

## 主题

图像去噪与噪声模型

## 说明

分区估计、色度联合与参数一致。

初始状态下 `imaging/image_denoise_profile.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 实现说明

纯 Python 标准库实现（评分环境无需 numpy / OpenCV），15 个入口在收到占位参数 `1`
或 `None` 时会跑一个固定种子的确定性小样例并返回可 JSON 序列化的证据 dict。

六个修复点与入口的对应：

| 题目要求 | 入口 / 关键实现 | 可核对的行为 |
| --- | --- | --- |
| 分亮度分区估计 | `variance_by_luma`、`estimate` | 5 个亮度分区各自的方差 / 稳健噪声 sigma；暗部噪声高于亮部，不被大片纯色稀释 |
| 噪声在色度联合 | `joint_chroma` | Cb/Cr 共用“联合色度距离 + 亮度引导”权重，彩色斑点显著下降 |
| 核按分辨率缩放 | `kernel_by_res` | `radius = 2 * height / 1080`，720p < 1080p < 2160p |
| 以首帧为参考 | `first_frame_ref`、`multi_frame(policy="first")` | 每帧直接配准到第 0 帧，`aligned_to` 全为 0，误差不累积；可切 `policy="previous"` 对照链式累积 |
| 参数全局一致 | `global_params`、`denoise` | 全图 / 所有异步分块共用一个 `param_id`，分块结果与整图逐位一致 |
| 强度随 ISO 缩放 | `iso_strength` | 默认 `sigma_scale = sqrt(iso / 100)`，高 ISO 去噪更强 |

辅助入口：`denoise`（分区亮度双边 + 联合色度 + 分块 / 线程池 + ISO 强度）、
`validate_detail`（边缘保留 / 涂抹检测）、`metrics`、`audit`、`fallback`、
`determinism`、`export`（ASCII PPM / JSON）。

除原有 15 项契约测试外，`tests/test_image_denoise_behavior.py` 提供 27 项数值
行为回归，覆盖上述六个修复点、边界尺寸、脏输入与确定性。

