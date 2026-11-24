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
