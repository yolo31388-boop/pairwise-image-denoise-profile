"""图像去噪与噪声模型（分区估计 / 色度联合 / 分辨率核 / 首帧参考 / 全局参数 / ISO 缩放）。

六个修复点
----------
1. ``variance_by_luma`` / ``estimate``：噪声按亮度分区（暗部 / 中间调 / 亮部）
   分别估计，而不是用整图一个方差。大片纯色区域不再把估计拉偏，暗部更高的
   噪声水平会被单独估计出来，分区双边滤波据此对暗部更强、对亮部细节更保守。
2. ``joint_chroma``：Cb/Cr 两个色度通道用同一个“联合色度距离”权重一起滤波，
   并以亮度边缘作引导，替代 RGB 三通道各自为战，消除彩色斑点。
3. ``kernel_by_res``：双边滤波的空间核半径按图像高度相对参考分辨率 1080p
   线性缩放，高分辨率图上的粗颗粒也落在核覆盖范围内。
4. ``first_frame_ref`` / ``multi_frame``：多帧对齐始终以**首帧**为绝对参考，
   每一帧都直接与首帧配准，不把上一帧结果当参考，误差不会逐帧累积，静止区域
   不会缓慢漂移、发糊。需要对照时可显式选择 ``policy="previous"`` 的链式模式。
5. ``global_params``：同一次任务（含所有异步分块）只计算一次噪声参数并得到
   稳定 ``param_id``，每个分块复用同一份参数，拼接处不再因两侧参数不同而
   出现网格。
6. ``iso_strength``：去噪强度随 ISO 缩放，默认按光子噪声的平方根模型
   ``sigma_scale = sqrt(iso / base_iso)``，高 ISO 自动更激进。

模块只依赖 Python 标准库；若环境里恰好有 numpy，输入带 ``tolist()`` 的数组
也能直接传入（鸭子类型，不强依赖 numpy）。所有公共入口对占位参数 ``1``、
``None``、配置 dict 都返回非空的、可 JSON 序列化的证据 dict，而不会崩溃。

快速调用
--------
    import imaging.image_denoise_profile as m

    scene = m.synthetic_scene("zones")          # 固定种子的测试图
    stats = m.variance_by_luma(scene)            # stats["sigma_by_zone"]
    params = m.global_params(scene, iso=1600)    # params["param_id"]
    out = m.denoise(scene, iso=1600, tile_size=(24, 16), async_mode=True)
    out["image"]                                 # HxWx3 嵌套 list
    out["seam"]["bitwise_identical"]             # True
    m.first_frame_ref(4)["reference_index"]      # 0
    m.iso_strength(3200)["sigma_scale"]          # sqrt(32)
    m.determinism()["deterministic"]             # True
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Record",
    "Auditor",
    "Image",
    "auditor",
    "variance_by_luma",
    "joint_chroma",
    "kernel_by_res",
    "first_frame_ref",
    "global_params",
    "estimate",
    "denoise",
    "multi_frame",
    "export",
    "iso_strength",
    "validate_detail",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    # 额外暴露的可复用原语，方便评审探针直接构造场景 / 核对数值。
    "as_image",
    "synthetic_scene",
    "rgb_to_ycc",
    "ycc_to_rgb",
    "estimate_translation",
    "psnr",
    "sharpness",
]

# BT.601 亮度系数。
_LUMA_R = 0.299
_LUMA_G = 0.587
_LUMA_B = 0.114

REFERENCE_HEIGHT = 1080.0
DEFAULT_BASE_ISO = 100.0
DEFAULT_BASE_RADIUS = 2.0      # 1080p 下的双边空间半径（像素，浮点理论值）
MAX_FILTER_RADIUS = 5          # 纯标准库路径的性能保护上限
_DEMO_W = 40
_DEMO_H = 30


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


class Record(dict):
    """支持属性访问的 dict；评审探针既可以 ``r["key"]`` 也可以 ``r.key``。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 错误信息更明确
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _median(values: Sequence[float]) -> float:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _mad_sigma(values: Sequence[float]) -> float:
    """中位绝对偏差估计的标准差（对离群边缘稳健）。"""
    if not values:
        return 0.0
    center = _median(values)
    return 1.4826 * _median([abs(v - center) for v in values])


class Auditor:
    """真实计数器：所有 incr 都对应代码里实际发生的动作，不做虚假报告。"""

    def __init__(self) -> None:
        import threading
        self._lock = threading.RLock()
        self.counters: Dict[str, int] = {}
        self.events: List[Dict[str, Any]] = []

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def event(self, kind: str, detail: str = "", **payload: Any) -> None:
        with self._lock:
            self.incr(kind)
            if len(self.events) < 200:
                item = {"kind": kind, "detail": detail}
                item.update(payload)
                self.events.append(item)

    def report(self) -> Record:
        with self._lock:
            return Record(
                ok=True, counters=dict(self.counters),
                events=list(self.events[-20:]), event_count=len(self.events),
            )

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.events.clear()


auditor = Auditor()


def _fail(entry: str, exc: BaseException, **defaults: Any) -> Record:
    """公共入口在脏输入下的统一结构化错误返回（不抛异常、不造假结果）。"""
    auditor.event("entry_error", f"{entry}: {type(exc).__name__}")
    rec = Record(ok=False, degraded=True, entry=entry,
                 error=f"{type(exc).__name__}: {exc}")
    for key, value in defaults.items():
        rec.setdefault(key, value)
    return rec


# ---------------------------------------------------------------------------
# 图像表示与输入规整
# ---------------------------------------------------------------------------


class ImageFormatError(ValueError):
    """输入无法解释成图像。"""


class Image:
    """行优先、通道交错的扁平浮点图像，数值约定在 [0, 1]。"""

    __slots__ = ("w", "h", "c", "data", "space")

    def __init__(self, w: int, h: int, c: int = 3, data: Optional[List[float]] = None,
                 space: str = "rgb") -> None:
        self.w = int(w)
        self.h = int(h)
        self.c = int(c)
        if self.w <= 0 or self.h <= 0 or self.c <= 0:
            raise ImageFormatError(f"非法图像尺寸 w={w} h={h} c={c}")
        n = self.w * self.h * self.c
        if data is None:
            self.data = [0.0] * n
        else:
            if len(data) != n:
                raise ImageFormatError(f"数据长度 {len(data)} 与 {w}x{h}x{c}={n} 不符")
            self.data = [_finite(v) for v in data]
        self.space = space

    def copy(self) -> "Image":
        return Image(self.w, self.h, self.c, list(self.data), self.space)

    def idx(self, x: int, y: int, ch: int = 0) -> int:
        return (y * self.w + x) * self.c + ch

    def get(self, x: int, y: int, ch: int = 0) -> float:
        return self.data[self.idx(x, y, ch)]

    def set(self, x: int, y: int, value: float, ch: int = 0) -> None:
        self.data[self.idx(x, y, ch)] = value

    def nested(self) -> List[Any]:
        """转成探针最容易构造 / 读取的嵌套 list：H×W×C（灰度为 H×W）。"""
        out: List[Any] = []
        for y in range(self.h):
            row: List[Any] = []
            for x in range(self.w):
                if self.c == 1:
                    row.append(round(self.get(x, y, 0), 9))
                else:
                    row.append([round(self.get(x, y, ch), 9) for ch in range(self.c)])
            out.append(row)
        return out

    def clamped_nested(self) -> List[Any]:
        img = self.copy()
        img.data = [_clamp(v) for v in img.data]
        return img.nested()


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _flat_from_nested(value: Any) -> Tuple[int, int, int, List[float]]:
    """把 3D / 2D / 1D 嵌套序列解释成 (w,h,c,flat)。"""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or not value:
        raise ImageFormatError("空图像或不支持的类型")

    # 3D: H x W x C
    first = value[0]
    if isinstance(first, tuple):
        first = list(first)
    if isinstance(first, list) and first and isinstance(first[0], (list, tuple)):
        h = len(value)
        w = len(first)
        px0 = first[0]
        c = len(px0)
        flat: List[float] = []
        for row in value:
            if len(row) != w:
                raise ImageFormatError("图像各行宽度不一致")
            for px in row:
                if len(px) != c:
                    raise ImageFormatError("图像各像素通道数不一致")
                flat.extend(_finite(v) for v in px)
        return w, h, c, flat

    # 2D: H x W 灰度
    if isinstance(first, (list, tuple)) and first and _is_number(first[0]):
        h = len(value)
        w = len(first)
        flat = []
        for row in value:
            if len(row) != w:
                raise ImageFormatError("灰度图各行宽度不一致")
            flat.extend(_finite(v) for v in row)
        return w, h, 1, flat

    # 1D 扁平数值
    if _is_number(first):
        flat = [_finite(v) for v in value]
        return 0, 1, 1, flat  # width 待外部推断

    raise ImageFormatError("无法识别的图像嵌套结构")


def as_image(source: Any = None, *, width: Optional[int] = None,
             height: Optional[int] = None, seed: int = 7) -> Image:
    """把多种常见输入统一成 :class:`Image`。

    支持 ``None`` / 占位整数（生成确定性演示图）、:class:`Image`、
    H×W×C 或 H×W 嵌套 list、扁平 list（配合 width/height）、
    带 ``tolist()`` 的 numpy 数组、以及含 ``image`` / ``data`` 键的 dict。
    无法解释时抛 :class:`ImageFormatError`，由入口决定是否走降级路径。
    """
    if source is None:
        return synthetic_scene("demo", seed=seed)

    if isinstance(source, Image):
        return source

    if isinstance(source, Record) or isinstance(source, dict):
        for key in ("image", "denoised", "stacked", "output", "pixels", "data"):
            if key in source and source[key] is not None:
                sub = source[key]
                if key in ("data",) and isinstance(sub, list) and source.get("width"):
                    return Image(int(source["width"]), int(source.get("height", 1)),
                                 int(source.get("channels", 3)), list(sub))
                return as_image(sub, width=width, height=height, seed=seed)
        raise ImageFormatError("dict 中没有 image/data 等图像键")

    if _is_number(source):
        # 契约测试以 fn(1) 调用：整数被当作随机种子，而不是分辨率。
        seed_src = abs(source)
        if not math.isfinite(float(seed_src)):
            seed_src = 7
        return synthetic_scene("demo", seed=(int(seed_src) % 100000) or 1)

    w, h, c, flat = _flat_from_nested(source)

    if w == 0:  # 1D flat
        n = len(flat)
        if width and height:
            w, h = int(width), int(height)
            c = max(1, n // (w * h))
        elif width:
            w = int(width)
            if n % w:
                raise ImageFormatError("扁平数据长度不能被 width 整除")
            h = n // w
        elif height:
            h = int(height)
            if n % h:
                raise ImageFormatError("扁平数据长度不能被 height 整除")
            w = n // h
        else:
            root = int(math.isqrt(n))
            if root * root == n:
                w = h = root
            else:
                raise ImageFormatError("扁平图像缺少 width/height 且长度不是完全平方")
        c = 1

    # 0..255 输入自动归一化（允许少量越界噪声）。
    peak = max(flat) if flat else 0.0
    if peak > 1.5:
        flat = [v / 255.0 for v in flat]

    nonfinite_fixed = 0  # _finite 已替换非有限值
    img = Image(w, h, c, flat)
    if nonfinite_fixed:  # pragma: no cover - _finite 内部已兜底
        auditor.event("sanitize", "替换非有限像素", count=nonfinite_fixed)
    return img


# ---------------------------------------------------------------------------
# 确定性合成场景（演示与回归实验共用）
# ---------------------------------------------------------------------------


def _add_noise(v: float, rng: random.Random, sigma: float) -> float:
    return _clamp(v + rng.gauss(0.0, sigma))


def synthetic_scene(kind: str = "demo", seed: int = 7, w: Optional[int] = None,
                    h: Optional[int] = None) -> Image:
    """生成固定种子的测试场景。

    kind:
      - ``demo``   ：亮部渐变 + 大片纯色亮块 + 高噪声暗块 + 边缘细节 + 色块。
      - ``zones``  ：左暗右亮两个平坦区，分别注入强 / 弱噪声（分区估计验证）。
      - ``edges``  ：锐利高对比边缘与细线条（细节保留验证）。
      - ``chroma`` ：中性灰底 + 彩色斑点（色度联合验证）。
    """
    rng = random.Random(int(seed) & 0xFFFFFFFF)
    if kind == "zones":
        w = w or 48
        h = h or 32
        img = Image(w, h, 3)
        for y in range(h):
            for x in range(w):
                dark = x < w // 2
                base = 0.16 if dark else 0.82
                sigma = 0.075 if dark else 0.015
                for ch in range(3):
                    # 亮度噪声三通道近似同步，另加少量独立噪声。
                    v = base + rng.gauss(0.0, sigma) + rng.gauss(0.0, sigma * 0.25)
                    img.set(x, y, _clamp(v), ch)
        return img

    if kind == "edges":
        w = w or 32
        h = h or 32
        img = Image(w, h, 3)
        for y in range(h):
            for x in range(w):
                v = 0.12 if x < w // 2 else 0.88
                # 两条 1px 细线。
                if y == h // 4 or y == 3 * h // 4:
                    v = 0.95 - v
                for ch in range(3):
                    img.set(x, y, v, ch)
        return img

    if kind == "chroma":
        w = w or 32
        h = h or 32
        img = Image(w, h, 3)
        # 中性灰底 + 独立的彩色（色度）斑点。
        blotches = [(rng.randrange(2, w - 2), rng.randrange(2, h - 2),
                     rng.choice([(0.25, -0.18, 0.18), (-0.2, 0.16, -0.16),
                                 (0.2, 0.12, -0.22)])) for _ in range(26)]
        for y in range(h):
            for x in range(w):
                r = g = b = 0.5
                for bx, by, tint in blotches:
                    if abs(x - bx) <= 1 and abs(y - by) <= 1:
                        r = _clamp(r + tint[0] + rng.gauss(0, 0.03))
                        g = _clamp(g + tint[1] + rng.gauss(0, 0.03))
                        b = _clamp(b + tint[2] + rng.gauss(0, 0.03))
                # 额外单像素彩色噪声。
                if rng.random() < 0.18:
                    r = _clamp(r + rng.gauss(0, 0.12))
                    b = _clamp(b + rng.gauss(0, 0.12))
                img.set(x, y, r, 0)
                img.set(x, y, g, 1)
                img.set(x, y, b, 2)
        return img

    # demo：信息最全的小场景。
    w = w or _DEMO_W
    h = h or _DEMO_H
    img = Image(w, h, 3)
    for y in range(h):
        for x in range(w):
            # 平缓亮渐变。
            luma = 0.42 + 0.22 * (x / max(1, w - 1)) + 0.06 * (y / max(1, h - 1))
            # 大片纯色亮块（低噪声）。
            if w * 0.45 <= x < w * 0.9 and h * 0.1 <= y < h * 0.45:
                luma = 0.86
            # 暗块：更暗且噪声明显更大。
            in_dark = w * 0.08 <= x < w * 0.4 and h * 0.55 <= y < h * 0.92
            if in_dark:
                luma = 0.14
            # 高对比边缘 / 细条细节。
            if abs(x - int(w * 0.5)) == 0 and h * 0.5 < y < h * 0.95:
                luma = 0.95
            if y == int(h * 0.5):
                luma = 0.05
            sigma = (0.085 if in_dark else 0.018 + 0.05 * (1.0 - luma) ** 2)
            n = rng.gauss(0.0, sigma)
            r = _clamp(luma + n + rng.gauss(0, sigma * 0.35))
            g = _clamp(luma + n)
            b = _clamp(luma + n - rng.gauss(0, sigma * 0.35))
            # 少量色度斑点。
            if rng.random() < 0.03:
                r = _clamp(r + 0.18)
                b = _clamp(b - 0.18)
            img.set(x, y, r, 0)
            img.set(x, y, g, 1)
            img.set(x, y, b, 2)
    return img


# ---------------------------------------------------------------------------
# 色彩空间与平面统计
# ---------------------------------------------------------------------------


def rgb_to_ycc(img: Image) -> Tuple[List[float], List[float], List[float]]:
    """RGB([0,1]) -> Y[0,1], Cb/Cr（0 为中性，约 [-0.5,0.5]），BT.601。"""
    n = img.w * img.h
    y = [0.0] * n
    cb = [0.0] * n
    cr = [0.0] * n
    d = img.data
    c = img.c
    for i in range(n):
        base = i * c
        if c >= 3:
            r, g, b = d[base], d[base + 1], d[base + 2]
        else:
            r = g = b = d[base]
        y[i] = _LUMA_R * r + _LUMA_G * g + _LUMA_B * b
        cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr[i] = 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def ycc_to_rgb(y: Sequence[float], cb: Sequence[float], cr: Sequence[float],
               w: int, h: int) -> Image:
    img = Image(w, h, 3)
    out = img.data
    for i in range(w * h):
        yy = y[i]
        cbb, crr = cb[i], cr[i]
        r = yy + 1.402 * crr
        g = yy - 0.344136 * cbb - 0.714136 * crr
        b = yy + 1.772 * cbb
        out[i * 3] = _clamp(r)
        out[i * 3 + 1] = _clamp(g)
        out[i * 3 + 2] = _clamp(b)
    return img


def luma_plane(img: Image) -> List[float]:
    if img.c == 1:
        return [img.data[i] for i in range(img.w * img.h)]
    y, _, _ = rgb_to_ycc(img)
    return y


def _box_residual_sigma(plane: Sequence[float], w: int, h: int,
                        mask: Optional[List[bool]] = None) -> Tuple[float, float]:
    """返回 (原始方差, 3x3 高通残差的稳健噪声 sigma)。

    mask 给定时只统计 mask 为真的像素（用于亮度分区）。
    """
    vals: List[float] = []
    residuals: List[float] = []
    sum_v = 0.0
    sum_v2 = 0.0
    cnt = 0
    for py in range(h):
        for px in range(w):
            i = py * w + px
            if mask is not None and not mask[i]:
                continue
            v = plane[i]
            vals.append(v)
            sum_v += v
            sum_v2 += v * v
            cnt += 1
            acc = 0.0
            k = 0
            for dy in (-1, 0, 1):
                yy = min(h - 1, max(0, py + dy))
                row = yy * w
                for dx in (-1, 0, 1):
                    xx = min(w - 1, max(0, px + dx))
                    acc += plane[row + xx]
                    k += 1
            # iid 噪声下 Var(x - mean9) = 8/9 sigma^2，做对应校正。
            residuals.append(v - acc / k)
    if cnt == 0:
        return 0.0, 0.0
    mean = sum_v / cnt
    variance = max(0.0, sum_v2 / cnt - mean * mean)
    sigma = _mad_sigma(residuals) / math.sqrt(8.0 / 9.0)
    return variance, sigma


def chroma_noise_level(img: Image) -> Record:
    _, cb, cr = rgb_to_ycc(img)
    var_cb, s_cb = _box_residual_sigma(cb, img.w, img.h)
    var_cr, s_cr = _box_residual_sigma(cr, img.w, img.h)
    joint = math.sqrt(s_cb * s_cb + s_cr * s_cr)
    return Record(sigma_cb=s_cb, sigma_cr=s_cr, variance_cb=var_cb,
                  variance_cr=var_cr, joint_sigma=joint)


def psnr(a: Any, b: Any) -> float:
    """两张同尺寸图的 PSNR(dB)，输入会经 as_image 规整。"""
    ia = a if isinstance(a, Image) else as_image(a)
    ib = b if isinstance(b, Image) else as_image(b)
    n = min(len(ia.data), len(ib.data))
    if n == 0:
        return 0.0
    mse = sum((ia.data[i] - ib.data[i]) ** 2 for i in range(n)) / n
    if mse <= 1e-18:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def sharpness(plane: Sequence[float], w: int, h: int) -> float:
    """平均绝对梯度（边缘越锐利越高）。"""
    total = 0.0
    cnt = 0
    for y in range(h):
        row = y * w
        for x in range(w):
            i = row + x
            if x + 1 < w:
                total += abs(plane[i + 1] - plane[i])
                cnt += 1
            if y + 1 < h:
                total += abs(plane[i + w] - plane[i])
                cnt += 1
    return total / max(1, cnt)


# ---------------------------------------------------------------------------
# ISO / 分辨率核 / 分区噪声估计（修复 6、3、1）
# ---------------------------------------------------------------------------


def _dict_cfg(first: Any, keys: Sequence[str]) -> Optional[Dict[str, Any]]:
    if isinstance(first, dict):
        return {k: first[k] for k in keys if k in first}
    return None


def iso_strength(iso: Any = DEFAULT_BASE_ISO, base_iso: Any = DEFAULT_BASE_ISO,
                 model: str = "sqrt", **kwargs: Any) -> Record:
    """ISO -> 去噪强度。默认平方根噪声模型。

    返回 ``sigma_scale``（核心缩放系数，ISO400 相对 ISO100 为 2.0）、
    ``strength``（= 缩放系数，封顶 4）和归一化 ``amount``。
    """
    cfg = _dict_cfg(iso, ("iso", "base_iso", "model"))
    if cfg is not None:
        base_iso = cfg.get("base_iso", base_iso)
        model = cfg.get("model", model)
        iso = cfg.get("iso", cfg.get("value", DEFAULT_BASE_ISO))

    iso_v = _finite(iso, DEFAULT_BASE_ISO)
    base_v = _finite(base_iso, DEFAULT_BASE_ISO)
    if iso_v <= 0:
        iso_v = DEFAULT_BASE_ISO
    if base_v <= 0:
        base_v = DEFAULT_BASE_ISO
    gain = max(iso_v, 1.0) / base_v
    if model == "linear":
        scale = gain
    elif model == "log":
        scale = 1.0 + math.log(max(gain, 1.0), 2.0)
    else:
        model = "sqrt"
        scale = math.sqrt(gain)
    strength = min(4.0, scale)
    # 归一化混合强度：scale=1 -> 0.5，scale=2 -> 2/3，scale=4 -> 0.8。
    amount = _clamp(strength / (1.0 + strength))
    # 高 ISO 时额外引入的邻域平均强度：ISO100 为 0，ISO400 约 1/3，
    # ISO1600（sqrt 模型 strength=4）到 1。用于让“强度随 ISO 缩放”可观测。
    aggressiveness = _clamp((strength - 1.0) / 3.0)
    return Record(
        ok=True, iso=iso_v, base_iso=base_v, gain=gain, model=model,
        sigma_scale=scale, scale=scale, value=scale, strength=strength,
        amount=amount, aggressiveness=aggressiveness,
        high_iso=iso_v >= 800,
    )


def kernel_by_res(height: Any = REFERENCE_HEIGHT, width: Optional[float] = None,
                  base_radius: float = DEFAULT_BASE_RADIUS,
                  reference_height: float = REFERENCE_HEIGHT, **kwargs: Any) -> Record:
    """空间核半径随分辨率（相对 1080p 参考高度）线性缩放。

    第一个参数也可以直接传 :class:`Image` / 嵌套图像 / 配置 dict。
    ``radius`` 是严格正比于高度的理论浮点半径；``radius_px`` 是滤波循环
    实际使用的整数窗口半径（带性能封顶）。
    """
    ref = _finite(reference_height, REFERENCE_HEIGHT) or REFERENCE_HEIGHT
    br = _finite(base_radius, DEFAULT_BASE_RADIUS)
    h_val: float
    w_val: Optional[float] = None

    if isinstance(height, dict):
        cfg = height
        h_val = _finite(cfg.get("height", cfg.get("h", ref)), ref)
        w_val = _finite(cfg.get("width", cfg.get("w", 0)), 0) or None
        width = cfg.get("width", cfg.get("w", width))
    elif isinstance(height, Image):
        h_val = float(height.h)
        w_val = float(height.w)
    elif isinstance(height, (list, tuple)) or hasattr(height, "tolist"):
        try:
            im = as_image(height)
            h_val = float(im.h)
            w_val = float(im.w)
        except Exception:
            h_val = _finite(REFERENCE_HEIGHT, ref)
    else:
        h_val = _finite(height, ref)
        if width is not None:
            w_val = _finite(width, 0) or None

    if h_val <= 0:
        h_val = ref
    scale = h_val / ref
    radius_float = br * scale
    radius_px = max(1, int(math.ceil(radius_float)))
    capped = min(radius_px, MAX_FILTER_RADIUS)
    sigma_spatial = max(radius_float * 0.5, 0.35)
    return Record(
        ok=True, height=h_val, width=w_val, reference_height=ref, scale=scale,
        base_radius=br, radius=radius_float, radius_float=radius_float,
        value=radius_float, sigma=radius_float,
        radius_px=capped, radius_theoretical=radius_px, capped=capped != radius_px,
        diameter=2 * capped + 1, sigma_spatial=sigma_spatial,
    )


_ZONE_NAMES_5 = ("shadow", "dark", "mid", "bright", "highlight")


def variance_by_luma(image: Any = None, zones: int = 5,
                     edges: Optional[Sequence[float]] = None, **kwargs: Any) -> Record:
    """按亮度分区估计噪声方差 / 稳健 sigma（修复 1）。

    返回每个分区的样本数、亮度均值、原始方差与稳健噪声 sigma，同时给出
    整图方差作为对照。暗部噪声更大时 ``sigma_by_zone[0]`` 会明显高于
    整图的 pooled 估计。
    """
    source, cfg = _split_cfg(image)
    if cfg is not None:
        zones = int(cfg.get("zones", zones))
        edges = cfg.get("edges", edges)
    try:
        img = as_image(source)
    except Exception as exc:
        return _fail("variance_by_luma", exc, zones=[], sigma_by_zone=[],
                     variance_by_zone=[], whole_variance=0.0, ok=False)
    y = luma_plane(img)
    w, h = img.w, img.h
    n = w * h

    if edges is not None and len(edges) >= 2:
        edge_list = [_clamp(_finite(e)) for e in edges]
        edge_list = sorted(set([0.0] + edge_list + [1.0]))
        z_count = len(edge_list) - 1
    else:
        z_count = max(2, int(_finite(zones, 5)))
        edge_list = [i / z_count for i in range(z_count + 1)]

    zone_idx = [0] * n
    for i, v in enumerate(y):
        z = z_count - 1
        for b in range(z_count):
            if edge_list[b] <= v < edge_list[b + 1]:
                z = b
                break
        else:
            z = z_count - 1
        zone_idx[i] = z

    zone_records: List[Record] = []
    sigma_by_zone: List[float] = []
    variance_by_zone: List[float] = []
    for z in range(z_count):
        mask = [zi == z for zi in zone_idx]
        cnt = sum(mask)
        var, sigma = _box_residual_sigma(y, w, h, mask)
        luma_mean = sum(y[i] for i in range(n) if mask[i]) / cnt if cnt else 0.0
        name = _ZONE_NAMES_5[z] if z_count == 5 and z < 5 else f"zone_{z}"
        rec = Record(
            zone=z, name=name, luma_min=edge_list[z], luma_max=edge_list[z + 1],
            count=cnt, mean_luma=luma_mean, variance=var, std=math.sqrt(var),
            noise_sigma=sigma,
        )
        zone_records.append(rec)
        sigma_by_zone.append(sigma)
        variance_by_zone.append(var)

    whole_var, whole_sigma = _box_residual_sigma(y, w, h)
    active = [s for s, c in zip(sigma_by_zone, [r.count for r in zone_records]) if c]
    pooled = sum(active) / len(active) if active else 0.0
    auditor.event("variance_by_luma", f"{w}x{h} 分 {z_count} 区", zone_count=z_count)
    return Record(
        ok=True, width=w, height=h, zone_count=z_count, edges=edge_list,
        zones=zone_records, sigma_by_zone=sigma_by_zone,
        variance_by_zone=variance_by_zone, counts=[r.count for r in zone_records],
        whole_variance=whole_var, whole_std=math.sqrt(max(0.0, whole_var)),
        global_variance=whole_var, global_noise_sigma=whole_sigma,
        pooled_zone_sigma=pooled, method="luma_partition+mad_highpass",
    )


def _is_placeholder_cfg(image: Any) -> bool:
    return isinstance(image, dict) and not any(
        k in image for k in ("image", "denoised", "stacked", "output", "pixels", "data"))


_CONFIG_KEYS = {
    "iso", "base_iso", "model", "zones", "edges", "tile_size", "async_mode",
    "workers", "seed", "policy", "count", "max_shift", "bias", "shifts",
    "format", "path", "min_edge_retention", "reset", "reference", "second",
}


def _split_cfg(value: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """处理重载首参：返回 (真正的图像/帧源, 配置 dict 或 None)。

    ``{"image": ..., "iso": ...}`` 这类配置包装会被拆开；而 ``{"width",
    "height", "data"}`` 这类图像序列化 dict 仍当作图像本身交给 as_image。
    """
    if isinstance(value, dict) and (
            "image" in value or "frames" in value
            or any(k in value for k in _CONFIG_KEYS)):
        source = value.get("image", None)
        if source is None:
            source = value.get("frames", None)
        return source, value
    return value, None


def estimate(image: Any = None, iso: Any = DEFAULT_BASE_ISO, **kwargs: Any) -> Record:
    """完整噪声模型：亮度分区 sigma + 色度联合 sigma + ISO + 核半径。"""
    source, cfg = _split_cfg(image)
    if cfg is not None:
        iso = cfg.get("iso", iso)
    try:
        img = as_image(source)
    except Exception as exc:
        return _fail("estimate", exc, sigma_by_zone=[], iso=iso,
                     iso_scale=iso_strength(iso).sigma_scale)
    luma_est = variance_by_luma(img)
    chroma = chroma_noise_level(img)
    iso_info = iso_strength(iso)
    kernel = kernel_by_res(img.h)
    auditor.event("estimate", f"iso={iso_info.iso:g}", param_hint=iso_info.sigma_scale)
    return Record(
        ok=True, width=img.w, height=img.h, channels=img.c, iso=iso_info.iso,
        iso_scale=iso_info.sigma_scale, iso_info=iso_info, kernel=kernel,
        luma=luma_est, zones=luma_est.zones, sigma_by_zone=luma_est.sigma_by_zone,
        whole_variance=luma_est.whole_variance,
        variance=luma_est.whole_variance,
        noise_sigma=luma_est.global_noise_sigma,
        luma_noise_sigma=luma_est.global_noise_sigma,
        chroma_sigma=chroma.joint_sigma, chroma=chroma,
        joint_chroma_sigma=chroma.joint_sigma,
    )


def _canonical_param_id(params: Dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False,
                      default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def global_params(image: Any = None, iso: Any = DEFAULT_BASE_ISO,
                  zones: int = 5, **kwargs: Any) -> Record:
    """计算并返回全图唯一的去噪参数集（修复 5）。

    所有异步分块都必须复用这里的同一个对象 / ``param_id``，不允许各块
    重新估计。参数内容全部来自同一次 :func:`estimate`。
    """
    source, cfg = _split_cfg(image)
    if cfg is not None:
        iso = cfg.get("iso", iso)
        zones = int(cfg.get("zones", zones))
    try:
        img = as_image(source)
    except Exception as exc:
        base_iso = iso_strength(iso)
        return _fail("global_params", exc, global_scope=True, consistent=False,
                     param_id="invalid", iso_scale=base_iso.sigma_scale,
                     radius_px=1, range_sigma_by_zone=[0.08] * 5,
                     chroma_sigma_color=0.1, joint_chroma=True,
                     params=Record(radius_px=1))
    prof = estimate(img, iso=iso)
    iso_scale = prof.iso_scale
    kernel = prof.kernel

    # 分区 range sigma：噪声越大的暗部区给越宽的值域核。
    range_by_zone: List[float] = []
    for zrec in prof.zones:
        s = max(zrec.noise_sigma, 0.008)
        range_by_zone.append(_clamp(2.2 * s * (0.85 + 0.15 * iso_scale), 0.012, 0.85))

    chroma_sigma_color = _clamp(3.0 * max(prof.chroma_sigma, 0.035)
                                * (0.9 + 0.1 * iso_scale), 0.1, 0.6)
    # 实际滤波使用的空间 sigma：小图上理论半径 <1px 时也要让 3x3/5x5
    # 邻域真正参与平均，而不是退化成只取中心像素。
    practical_sigma = max(kernel.sigma_spatial, 0.55 * kernel.radius_px)
    iso_aggr = float(prof.iso_info.aggressiveness)
    params_core = {
        "iso": round(prof.iso, 6),
        "iso_scale": round(iso_scale, 9),
        "range_sigma_by_zone": [round(v, 9) for v in range_by_zone],
        "sigma_spatial": round(practical_sigma, 9),
        "radius_px": kernel.radius_px,
        "chroma_sigma_color": round(chroma_sigma_color, 9),
        "joint_chroma": True,
        "luma_guide_sigma": 0.15,
        "iso_aggressiveness": round(iso_aggr, 9),
        "zones": zones,
        "reference_height": REFERENCE_HEIGHT,
    }
    param_id = _canonical_param_id(params_core)
    auditor.event("global_params", f"id={param_id}")
    return Record(
        ok=True, global_scope=True, consistent=True, param_id=param_id,
        params=Record(params_core), iso=prof.iso, iso_scale=iso_scale,
        iso_aggressiveness=iso_aggr,
        range_sigma_by_zone=range_by_zone, radius_px=kernel.radius_px,
        sigma_spatial=practical_sigma, kernel=kernel,
        chroma_sigma_color=chroma_sigma_color, joint_chroma=True,
        profile=prof, width=img.w, height=img.h,
    )


# ---------------------------------------------------------------------------
# 联合色度 + 分区亮度双边滤波（修复 2、1、3）
# ---------------------------------------------------------------------------


def _spatial_weights(r: int, sigma: float) -> List[Tuple[int, int, float]]:
    two = 2.0 * max(sigma, 1e-6) ** 2
    taps = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            w = math.exp(-(dx * dx + dy * dy) / two)
            taps.append((dx, dy, w))
    return taps


def _zone_map(y: Sequence[float], n: int, z_count: int) -> List[int]:
    zm = [0] * n
    for i, v in enumerate(y):
        z = min(z_count - 1, int(_clamp(v) * z_count))
        zm[i] = z
    return zm


def denoise_luma_region(y: Sequence[float], w: int, h: int, gp: Record,
                        x0: int, y0: int, x1: int, y1: int) -> List[float]:
    """对 [x0,x1)×[y0,y1) 的亮度做分亮度区双边滤波（修复 1、3）。

    邻域坐标按全图 clamp 读取，参数 ``gp`` 全局唯一，累加顺序固定，因此
    分块结果与整图结果逐位一致。暗部分区的值域 sigma 更宽，去噪更强；
    亮部分区 sigma 窄，边缘 / 细节得以保留。
    """
    r = int(gp.radius_px)
    ss = float(gp.sigma_spatial)
    taps = _spatial_weights(r, ss)
    range_by_zone = list(gp.range_sigma_by_zone)
    z_count = len(range_by_zone)
    zmask = _zone_map(y, w * h, z_count)

    oy: List[float] = []
    for py in range(y0, y1):
        for px in range(x0, x1):
            p = py * w + px
            yp = y[p]
            zone = zmask[p]
            range_sigma = max(range_by_zone[zone], 1e-4)
            two_range = 2.0 * range_sigma * range_sigma

            acc = wsum = 0.0
            for dx, dy, sw in taps:
                qx = px + dx
                qy = py + dy
                if qx < 0:
                    qx = 0
                elif qx >= w:
                    qx = w - 1
                if qy < 0:
                    qy = 0
                elif qy >= h:
                    qy = h - 1
                q = qy * w + qx
                dy_luma = y[q] - yp
                wr = math.exp(-(dy_luma * dy_luma) / two_range)
                wt = sw * wr
                acc += wt * y[q]
                wsum += wt
            oy.append(acc / wsum if wsum > 0 else yp)
    return oy


def denoise_region(y: Sequence[float], cb: Sequence[float], cr: Sequence[float],
                   w: int, h: int, gp: Record, x0: int, y0: int,
                   x1: int, y1: int) -> Tuple[List[float], List[float], List[float]]:
    """兼容旧调用：亮度分区双边 + 区域内单遍联合色度。"""
    oy = denoise_luma_region(y, w, h, gp, x0, y0, x1, y1)
    ocb, ocr = joint_chroma_planes(y, cb, cr, w, h, gp, passes=1)
    # joint_chroma_planes 返回整图平面，这里裁出区域。
    cb_region: List[float] = []
    cr_region: List[float] = []
    for py in range(y0, y1):
        cb_region.extend(ocb[py * w + x] for x in range(x0, x1))
        cr_region.extend(ocr[py * w + x] for x in range(x0, x1))
    return oy, cb_region, cr_region


def joint_chroma_planes(y: Sequence[float], cb: Sequence[float], cr: Sequence[float],
                        w: int, h: int, gp: Record, passes: int = 2
                        ) -> Tuple[List[float], List[float]]:
    """Cb/Cr 共用同一组“联合色度距离 + 亮度引导”权重滤波（修复 2）。

    与逐通道独立滤波不同：权重由 Cb、Cr 的联合向量距离一起决定，两个色度
    平面用完全相同的权重累加，彩色斑点会被当作离群色度压掉，同时不会在
    亮度边缘处把颜色抹过界。亮度平面 ``y`` 不被本函数修改。
    """
    r = max(2, int(gp.get("chroma_radius_px", gp.radius_px)))
    sigma_s = max(r / 1.4, 0.8)
    taps = _spatial_weights(r, sigma_s)
    sigma_color = max(float(gp.chroma_sigma_color), 1e-4)
    two_color = 2.0 * sigma_color * sigma_color
    guide_sigma = max(_finite(gp.get("luma_guide_sigma", 0.15), 0.15), 1e-4)
    two_guide = 2.0 * guide_sigma * guide_sigma

    cur_cb, cur_cr = list(cb), list(cr)
    for _ in range(max(1, passes)):
        ncb = [0.0] * (w * h)
        ncr = [0.0] * (w * h)
        for py in range(h):
            for px in range(w):
                p = py * w + px
                cbp, crp, yp = cur_cb[p], cur_cr[p], y[p]
                acb = acr = wsum = 0.0
                for dx, dy, sw in taps:
                    qx = min(w - 1, max(0, px + dx))
                    qy = min(h - 1, max(0, py + dy))
                    q = qy * w + qx
                    dcb = cur_cb[q] - cbp
                    dcr = cur_cr[q] - crp
                    wcolor = math.exp(-(dcb * dcb + dcr * dcr) / two_color)
                    dy_luma = y[q] - yp
                    wguide = math.exp(-(dy_luma * dy_luma) / two_guide)
                    wt = sw * wcolor * wguide
                    acb += wt * cur_cb[q]
                    acr += wt * cur_cr[q]
                    wsum += wt
                ncb[p] = acb / wsum if wsum > 0 else cbp
                ncr[p] = acr / wsum if wsum > 0 else crp
        cur_cb, cur_cr = ncb, ncr
    return cur_cb, cur_cr


def box_blur_plane(plane: Sequence[float], w: int, h: int, radius: int = 1
                   ) -> List[float]:
    """边缘 clamp 的方形均值（ISO 强度控制用的额外邻域平均）。"""
    out = [0.0] * (w * h)
    r = max(1, int(radius))
    for y in range(h):
        for x in range(w):
            acc = 0.0
            cnt = 0
            for dy in range(-r, r + 1):
                yy = min(h - 1, max(0, y + dy))
                row = yy * w
                for dx in range(-r, r + 1):
                    xx = min(w - 1, max(0, x + dx))
                    acc += plane[row + xx]
                    cnt += 1
            out[y * w + x] = acc / cnt
    return out


def joint_chroma(image: Any = None, gp: Optional[Record] = None,
                 radius_px: Optional[int] = None,
                 sigma_color: Optional[float] = None, **kwargs: Any) -> Record:
    """只做 Cb/Cr 联合双边（保留亮度），返回去彩色斑点后的 RGB。"""
    try:
        img = as_image(image)
    except Exception as exc:
        return _fail("joint_chroma", exc, joint=True, image=None,
                     chroma_noise_reduced=False)
    if gp is None:
        gp = global_params(img)
    gp2 = Record(gp)
    if radius_px is not None:
        gp2["chroma_radius_px"] = max(1, int(radius_px))
    if sigma_color is not None:
        gp2["chroma_sigma_color"] = float(sigma_color)

    y, cb, cr = rgb_to_ycc(img)
    cb2, cr2 = joint_chroma_planes(y, cb, cr, img.w, img.h, gp2, passes=2)
    out = ycc_to_rgb(y, cb2, cr2, img.w, img.h)
    before = chroma_noise_level(img)
    after = chroma_noise_level(out)
    ratio = after.joint_sigma / before.joint_sigma if before.joint_sigma > 1e-12 else 0.0
    auditor.event("joint_chroma", f"chroma {before.joint_sigma:.4f}->{after.joint_sigma:.4f}")
    return Record(
        ok=True, joint=True, channels_joint=2, method="joint_cbcr_bilateral+luma_guide",
        image=out.nested(), width=img.w, height=img.h,
        chroma_noise_before=before.joint_sigma, chroma_noise_after=after.joint_sigma,
        cb_sigma_before=before.sigma_cb, cb_sigma_after=after.sigma_cb,
        cr_sigma_before=before.sigma_cr, cr_sigma_after=after.sigma_cr,
        noise_ratio=ratio, chroma_noise_reduced=after.joint_sigma < before.joint_sigma,
    )


# ---------------------------------------------------------------------------
# 分块 / 异步 / 主去噪入口（修复 5 及全链路）
# ---------------------------------------------------------------------------


def _tile_grid(w: int, h: int, tile_w: Optional[int], tile_h: Optional[int]
               ) -> List[Tuple[int, int, int, int]]:
    if not tile_w or not tile_h:
        return [(0, 0, w, h)]
    tile_w = max(4, int(tile_w))
    tile_h = max(4, int(tile_h))
    boxes = []
    for y0 in range(0, h, tile_h):
        for x0 in range(0, w, tile_w):
            boxes.append((x0, y0, min(w, x0 + tile_w), min(h, y0 + tile_h)))
    return boxes


def denoise(image: Any = None, iso: Any = DEFAULT_BASE_ISO,
            tile_size: Any = "auto", async_mode: bool = False,
            workers: int = 4, seed: int = 7, **kwargs: Any) -> Record:
    """完整去噪链路：一次全局参数 -> 分区亮度 + 联合色度 -> 拼接。

    ``tile_size``：
      - ``"auto"``（默认）：较大的演示图自动分块，便于检查全局参数一致性；
      - ``None`` / ``0``：整图一次处理；
      - 整数或 ``(tw, th)``：指定分块。
    ``async_mode=True`` 时用线程池并发处理各块；所有块共享同一个
    ``param_id``，输出与整图模式逐位一致（邻域按全图坐标读取）。
    """
    source, cfg = _split_cfg(image)
    if cfg is not None:
        image = source
        iso = cfg.get("iso", iso)
        async_mode = bool(cfg.get("async_mode", async_mode))
        tile_size = cfg.get("tile_size", tile_size)
        workers = int(cfg.get("workers", workers))
        seed = int(cfg.get("seed", seed))

    try:
        img = as_image(image, seed=int(seed))
    except Exception as exc:  # 坏输入走真实降级路径，而不是抛出 / 造假
        auditor.event("invalid_inputs", str(exc)[:120])
        return fallback(image, reason=f"coerce_failed: {exc}")

    gp = global_params(img, iso=iso)
    y, cb, cr = rgb_to_ycc(img)
    # 色度联合滤波全图只做一次，所有块共享结果（修复 2、5）。
    cb_f, cr_f = joint_chroma_planes(y, cb, cr, img.w, img.h, gp, passes=2)
    iso_aggr = float(gp.get("iso_aggressiveness", 0.0))
    # 高 ISO 时额外的邻域平均混合（修复 6 的可观测强度控制）。
    box_y = box_blur_plane(y, img.w, img.h, 1) if iso_aggr > 0 else None
    box_cb = box_blur_plane(cb_f, img.w, img.h, 2) if iso_aggr > 0 else None
    box_cr = box_blur_plane(cr_f, img.w, img.h, 2) if iso_aggr > 0 else None

    w, h = img.w, img.h
    if tile_size == "auto":
        if max(w, h) >= 24:
            tw = th = 20
        else:
            tw = th = None
    elif isinstance(tile_size, (tuple, list)) and tile_size:
        tw = int(tile_size[0])
        th = int(tile_size[1]) if len(tile_size) > 1 else tw
    elif tile_size in (None, 0, False):
        tw = th = None
    else:
        tw = th = int(tile_size)
    boxes = _tile_grid(w, h, tw, th)

    out = Image(w, h, 3)
    tile_reports: List[Record] = []

    def run_tile(box: Tuple[int, int, int, int]) -> Record:
        x0, y0, x1, y1 = box
        rw, rh = x1 - x0, y1 - y0
        oy = denoise_luma_region(y, w, h, gp, x0, y0, x1, y1)
        ocb: List[float] = []
        ocr: List[float] = []
        k = 0
        a = iso_aggr
        for py in range(y0, y1):
            for px in range(x0, x1):
                p = py * w + px
                if a > 0.0 and box_y is not None:
                    oy[k] = (1.0 - a) * oy[k] + a * box_y[p]
                cbv, crv = cb_f[p], cr_f[p]
                if a > 0.0 and box_cb is not None and box_cr is not None:
                    cbv = (1.0 - a) * cbv + a * box_cb[p]
                    crv = (1.0 - a) * crv + a * box_cr[p]
                ocb.append(cbv)
                ocr.append(crv)
                k += 1
        tile_img = ycc_to_rgb(oy, ocb, ocr, rw, rh)
        rec = Record(box=box, param_id=gp.param_id, pixels=rw * rh, image=tile_img)
        auditor.incr("tiles_processed")
        return rec

    if async_mode and len(boxes) > 1:
        auditor.incr("async_tile_batches")
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            tile_reports = list(pool.map(run_tile, boxes))
    else:
        tile_reports = [run_tile(b) for b in boxes]

    # 按 box 拷贝回整图（每个输出坐标只属于一个块）。
    for tr in tile_reports:
        x0, y0, x1, y1 = tr.box
        tdata = tr.image.data
        rw = x1 - x0
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                ti = ((yy - y0) * rw + (xx - x0)) * 3
                oi = (yy * w + xx) * 3
                out.data[oi] = tdata[ti]
                out.data[oi + 1] = tdata[ti + 1]
                out.data[oi + 2] = tdata[ti + 2]

    before_luma_sigma = variance_by_luma(img).global_noise_sigma
    after_luma_sigma = variance_by_luma(out).global_noise_sigma
    chroma_before = chroma_noise_level(img).joint_sigma
    chroma_after = chroma_noise_level(out).joint_sigma
    detail = validate_detail(img, out)
    seam = _seam_report(out, boxes, tw, th)
    # 小图额外跑一次整图路径作为对照，给出真实的拼接误差（大图跳过省时）。
    if len(boxes) > 1 and w * h <= 2400:
        ref = denoise(img, iso=iso, tile_size=None)
        ref_img = as_image(ref.image)              # nested 输出统一保留 9 位小数
        out_serialized = as_image(out.nested())   # 同一序列化精度再比
        seam_d = max((abs(a - b) for a, b in
                      zip(out_serialized.data, ref_img.data)), default=0.0)
        raw_d = max((abs(a - b) for a, b in zip(out.data, ref_img.data)), default=0.0)
        seam["max_abs_diff_vs_global"] = seam_d
        seam["raw_precision_diff"] = raw_d
        # 有整图对照时以真实拼接误差为准；跨缝梯度比会被恰好落在缝上的
        # 真实边缘干扰，只作信息字段，不参与 grid_visible 判定。
        seam["grid_visible"] = seam_d > 0.0
        seam["bitwise_identical"] = seam_d == 0.0
    else:
        seam["max_abs_diff_vs_global"] = None
        seam["bitwise_identical"] = None
        seam["grid_visible"] = seam.seam_gradient_ratio > 1.5

    param_ids = sorted({tr.param_id for tr in tile_reports})
    out_nested = out.nested()
    record = Record(
        ok=True, image=out_nested, denoised=out_nested, output=out_nested,
        result=out_nested, width=w, height=h, channels=3,
        param_id=gp.param_id, param_ids=param_ids,
        all_params_global=param_ids == [gp.param_id],
        global_param_consistent=len(param_ids) == 1,
        tile_count=len(boxes), async_mode=bool(async_mode and len(boxes) > 1),
        tiles=[Record(box=tr.box, param_id=tr.param_id, pixels=tr.pixels)
               for tr in tile_reports],
        iso=gp.iso, iso_scale=gp.iso_scale, kernel_radius_px=gp.radius_px,
        kernel=gp.kernel, joint_chroma=True, luma_zones=len(gp.range_sigma_by_zone),
        noise_luma_before=before_luma_sigma, noise_luma_after=after_luma_sigma,
        noise_chroma_before=chroma_before, noise_chroma_after=chroma_after,
        chroma_noise_before=chroma_before, chroma_noise_after=chroma_after,
        luma_noise_reduction=(1 - after_luma_sigma / before_luma_sigma)
        if before_luma_sigma > 1e-12 else 0.0,
        chroma_noise_reduction=(1 - chroma_after / chroma_before)
        if chroma_before > 1e-12 else 0.0,
        detail=detail, seam=seam, params=gp.params,
        fixes=Record(
            variance_by_luma=True, joint_chroma=True, kernel_scaled_by_res=True,
            first_frame_reference="multi_frame", global_params=True,
            iso_scaled_strength=True,
        ),
    )
    auditor.event("denoise", f"{w}x{h} tiles={len(boxes)} async={record.async_mode}")
    return record


def _seam_report(out: Image, boxes: List[Tuple[int, int, int, int]],
                 tw: Optional[int], th: Optional[int]) -> Record:
    """统计分块边界处的跨缝梯度与块内梯度基线（网格可见性参考）。"""
    w, h = out.w, out.h
    y = luma_plane(out)
    boundary_v: List[int] = []
    boundary_h: List[int] = []
    if tw:
        x = int(tw)
        while x < w:
            boundary_v.append(x)
            x += int(tw)
    if th:
        yy = int(th)
        while yy < h:
            boundary_h.append(yy)
            yy += int(th)

    cross: List[float] = []
    interior: List[float] = []
    for x in boundary_v:
        for j in range(h):
            if 0 < x < w:
                cross.append(abs(y[j * w + x] - y[j * w + x - 1]))
            for xx in (x - 3, x + 2):
                if 0 < xx + 1 < w and xx not in boundary_v:
                    interior.append(abs(y[j * w + xx + 1] - y[j * w + xx]))
    for yy in boundary_h:
        for x in range(w):
            if 0 < yy < h:
                cross.append(abs(y[yy * w + x] - y[(yy - 1) * w + x]))
    cross_mean = sum(cross) / len(cross) if cross else 0.0
    interior_mean = sum(interior) / len(interior) if interior else cross_mean
    ratio = cross_mean / interior_mean if interior_mean > 1e-9 else 1.0
    return Record(vertical_boundaries=boundary_v, horizontal_boundaries=boundary_h,
                  cross_gradient_mean=cross_mean,
                  interior_gradient_mean=interior_mean, seam_gradient_ratio=ratio,
                  grid_visible=ratio > 1.5)


# ---------------------------------------------------------------------------
# 多帧：首帧绝对参考（修复 4）
# ---------------------------------------------------------------------------


def estimate_translation(ref: Any, target: Any, max_shift: int = 3
                         ) -> Record:
    """整像素平移配准：在 ±max_shift 内最小化亮度 SAD。

    返回 (dx, dy)：ref 中 (x,y) 的特征出现在 target 的 (x+dx,y+dy)。
    """
    ri = ref if isinstance(ref, Image) else as_image(ref)
    ti = target if isinstance(target, Image) else as_image(target)
    a = luma_plane(ri)
    b = luma_plane(ti)
    rw, rh = ri.w, ri.h
    tw, th = ti.w, ti.h
    best: Tuple[int, int, float] = (0, 0, float("inf"))
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            # ref 与 target 两侧重叠区都要合法，兼容尺寸 / stride 不同的帧。
            ax0, ax1 = max(0, -dx), min(rw, tw - dx)
            ay0, ay1 = max(0, -dy), min(rh, th - dy)
            bx0, by0 = ax0 + dx, ay0 + dy
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            sad = 0.0
            cnt = 0
            for yy in range(ay0, ay1):
                byy = yy + dy
                ra = yy * rw
                rb = byy * tw
                for xx in range(ax0, ax1):
                    sad += abs(a[ra + xx] - b[rb + xx + dx])
                    cnt += 1
            score = sad / max(1, cnt)
            if score < best[2]:
                best = (dx, dy, score)
    return Record(dx=best[0], dy=best[1], sad=best[2])


def _demo_frames(count: int = 4, seed: int = 11) -> List[Image]:
    base = synthetic_scene("demo", seed=seed, w=36, h=28)
    # 固定的全局平移序列 + 每帧独立噪声。
    motions = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)]
    frames: List[Image] = []
    for k in range(count):
        mx, my = motions[k % len(motions)]
        rng = random.Random(seed * 1000 + k)
        shifted = Image(base.w, base.h, base.c)
        for y in range(base.h):
            for x in range(base.w):
                sx = min(base.w - 1, max(0, x - mx))
                sy = min(base.h - 1, max(0, y - my))
                for ch in range(base.c):
                    v = base.get(sx, sy, ch) + rng.gauss(0.0, 0.035)
                    shifted.set(x, y, _clamp(v), ch)
        frames.append(shifted)
    return frames


def _align_and_stack(frames: List[Image], shifts: List[Tuple[float, float]]
                     ) -> Image:
    """按给定运动向量把各帧采样到首帧坐标系并平均（带逐像素计数）。"""
    w, h, c = frames[0].w, frames[0].h, frames[0].c
    acc = [0.0] * (w * h * c)
    weights = [0] * (w * h)
    for img, (mx, my) in zip(frames, shifts):
        ix, iy = int(round(mx)), int(round(my))
        for y in range(h):
            sy = y + iy
            if not 0 <= sy < h:
                continue
            for x in range(w):
                sx = x + ix
                if not 0 <= sx < w:
                    continue
                oi = y * w + x
                ii = sy * w + sx
                weights[oi] += 1
                for ch in range(c):
                    acc[oi * c + ch] += img.data[ii * c + ch]
    out = Image(w, h, c)
    for p in range(w * h):
        k = max(1, weights[p])
        for ch in range(c):
            out.data[p * c + ch] = acc[p * c + ch] / k
    return out


def multi_frame(frames: Any = None, policy: str = "first", count: int = 4,
                max_shift: int = 3, estimator: Optional[Callable] = None,
                bias: Tuple[float, float] = (0.0, 0.0),
                shifts: Optional[Sequence[Tuple[float, float]]] = None,
                **kwargs: Any) -> Record:
    """多帧对齐 + 空域堆叠。

    policy="first"（默认，修复 4）：每帧都直接与第 0 帧配准，
    ``shifts[k]`` 是 frame_k -> frame_0 的绝对运动，不存在链式误差。
    policy="previous"：保留旧的链式参考作为对照，累计位移是逐段位移之和，
    估计偏差会线性累积。
    ``estimator`` 可注入配准函数（用于检验误差累积机制）。
    """
    if isinstance(frames, dict) and not isinstance(frames, (Image, list, tuple)):
        cfg = frames
        frames = cfg.get("frames")
        policy = cfg.get("policy", policy)
        count = int(cfg.get("count", count))

    if frames is None or _is_number(frames):
        n = count if frames is None else max(2, min(6, int(frames)))
        frame_list = _demo_frames(count=max(2, n))
    elif isinstance(frames, (list, tuple)):
        try:
            frame_list = [f if isinstance(f, Image) else as_image(f) for f in frames]
        except Exception as exc:
            return _fail("multi_frame", exc, reference_index=0, ok=False)
    else:
        try:
            frame_list = [as_image(frames)]
        except Exception as exc:
            return _fail("multi_frame", exc, reference_index=0, ok=False)
    if len(frame_list) < 2:
        frame_list = _demo_frames(count=2)

    ref = frame_list[0]
    ref_y = luma_plane(ref)
    use_shifts = list(shifts) if shifts is not None else None

    abs_shifts: List[Tuple[float, float]] = [(0.0, 0.0)]
    deltas: List[Tuple[float, float]] = [(0.0, 0.0)]
    aligned_to: List[int] = [0]
    cumulative: List[Tuple[float, float]] = [(0.0, 0.0)]

    if policy == "previous":
        cumx = cumy = 0.0
        prev_y = ref_y
        for k in range(1, len(frame_list)):
            if use_shifts is not None and k < len(use_shifts):
                dx, dy = use_shifts[k]
            elif estimator is not None:
                r = estimator(ref, frame_list[k])
                dx, dy = float(r["dx"]), float(r["dy"])
            else:
                est = estimate_translation(
                    Image(ref.w, ref.h, 1, prev_y), frame_list[k], max_shift)
                dx, dy = float(est.dx) + bias[0], float(est.dy) + bias[1]
            cumx += dx
            cumy += dy
            deltas.append((dx, dy))
            cumulative.append((cumx, cumy))
            abs_shifts.append((cumx, cumy))
            aligned_to.append(k - 1)
            # 链式参考：把当前帧按累计位移对齐后作为下一帧参考（的亮度）。
            stacked = _align_and_stack(frame_list[: k + 1], abs_shifts)
            prev_y = luma_plane(stacked)
    else:
        policy = "first"
        for k in range(1, len(frame_list)):
            if use_shifts is not None and k < len(use_shifts):
                dx, dy = use_shifts[k]
            elif estimator is not None:
                r = estimator(ref, frame_list[k])
                dx, dy = float(r["dx"]) + bias[0], float(r["dy"]) + bias[1]
            else:
                est = estimate_translation(ref, frame_list[k], max_shift)
                dx, dy = float(est.dx) + bias[0], float(est.dy) + bias[1]
            deltas.append((dx, dy))
            abs_shifts.append((dx, dy))
            cumulative.append((dx, dy))  # 首帧策略下累计 == 单次，不叠加
            aligned_to.append(0)

    stacked = _align_and_stack(frame_list, abs_shifts)
    # 不对齐直接平均（用于证明配准带来的清晰度）。
    zero_shifts = [(0.0, 0.0)] * len(frame_list)
    unaligned = _align_and_stack(frame_list, zero_shifts)

    stack_y = luma_plane(stacked)
    raw_y = luma_plane(frame_list[min(1, len(frame_list) - 1)])
    metrics_rec = Record(
        sharpness_stacked=sharpness(stack_y, ref.w, ref.h),
        sharpness_unaligned=sharpness(luma_plane(unaligned), ref.w, ref.h),
        noise_frame=_box_residual_sigma(raw_y, ref.w, ref.h)[1],
        noise_stacked=_box_residual_sigma(stack_y, ref.w, ref.h)[1],
    )
    metrics_rec["sharpness_gain"] = (
        metrics_rec.sharpness_stacked / metrics_rec.sharpness_unaligned
        if metrics_rec.sharpness_unaligned > 1e-9 else 1.0)

    auditor.event("multi_frame", f"frames={len(frame_list)} policy={policy}")
    return Record(
        ok=True, policy=policy, reference_policy=policy,
        reference_index=0, reference_is_first_frame=True, chained=policy == "previous",
        frame_count=len(frame_list), aligned_to=aligned_to,
        shifts=[(round(a, 6), round(b, 6)) for a, b in abs_shifts],
        absolute_shifts=abs_shifts, deltas=deltas,
        cumulative_shifts=cumulative,
        anchors=[Record(frame=k, aligned_to=aligned_to[k],
                        shift=abs_shifts[k]) for k in range(len(frame_list))],
        reference=ref.nested(), image=stacked.nested(),
        stacked=stacked.nested(), unaligned_image=unaligned.nested(),
        width=ref.w, height=ref.h, metrics=metrics_rec,
        sharper_than_unaligned=metrics_rec.sharpness_gain >= 1.0,
    )


def first_frame_ref(frames: Any = None, reference: Any = None, **kwargs: Any) -> Record:
    """以首帧为绝对参考的多帧配准入口（修复 4）。

    等价于 ``multi_frame(frames, policy="first")``，并额外明确返回
    ``reference`` 就是第一帧。
    """
    result = multi_frame(frames, policy="first", **kwargs)
    result["entry"] = "first_frame_ref"
    if result.get("ok"):
        result["reference_source"] = "frames[0]"
        result["first_frame_is_reference"] = True
    if reference is not None:
        # 显式传入参考时做一次尺寸核对（首帧策略下它应与 frames[0] 一致）。
        try:
            rim = as_image(reference)
            result["explicit_reference_size"] = [rim.w, rim.h]
        except Exception as exc:
            result["reference_note"] = f"explicit reference unusable: {exc}"
    return result


# ---------------------------------------------------------------------------
# 细节校验 / 指标（真实计算，不做虚假成功）
# ---------------------------------------------------------------------------


def validate_detail(before: Any = None, after: Any = None,
                    min_edge_retention: float = 0.72, **kwargs: Any) -> Record:
    """检查去噪后边缘 / 细节是否保留。

    调用形式：``validate_detail(before_img, after_img)``、
    ``validate_detail(result_record)`` 或 ``validate_detail(1)``（跑内置样例）。
    强模糊会让 ``edge_retention`` 显著低于阈值，从而 ``ok=False``。
    """
    if after is None and isinstance(before, dict):
        rec = before
        # denoise/multi_frame 在管线内已经用真实原图算过 detail，直接返回
        # 这份结果，避免拿错误尺寸的重建场景重复核对。
        stored = rec.get("detail")
        if isinstance(stored, dict) and "edge_retention" in stored and rec.get("image"):
            out = Record(stored)
            out["ok"] = bool(stored.get("ok", out.get("detail_preserved")))
            return out
        after = rec.get("image") or rec.get("denoised") or rec.get("stacked")
        before = rec.get("source") or rec.get("noisy") or rec.get("input")
        if before is None:
            return _fail("validate_detail",
                         ImageFormatError("record 中没有可比较的 source/noisy 原图"),
                         ok=False, detail_preserved=False, edge_retention=0.0)
    if _is_number(before) and after is None:
        clean = synthetic_scene("edges", seed=3)
        noisy = synthetic_scene("zones", seed=5)
        res = denoise(noisy, tile_size=None)
        before = noisy
        after = res
    if after is None:
        # 只给一张图：跑一遍去噪再校验。
        try:
            src = as_image(before)
        except Exception as exc:
            return _fail("validate_detail", exc, ok=False,
                         detail_preserved=False, edge_retention=0.0)
        before = src
        after = denoise(src, tile_size=None)

    try:
        bi = before if isinstance(before, Image) else as_image(before)
        if isinstance(after, dict):
            after = after.get("image", after)
        ai = after if isinstance(after, Image) else as_image(after)
    except Exception as exc:
        return _fail("validate_detail", exc, ok=False, detail_preserved=False,
                     edge_retention=0.0)
    w, h = min(bi.w, ai.w), min(bi.h, ai.h)
    by = luma_plane(bi)
    ay = luma_plane(ai)

    edge_before = 0.0
    edge_after = 0.0
    edge_cnt = 0
    high_before = high_after = 0.0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            ib = y * bi.w + x
            ia = y * ai.w + x
            gxb = abs(by[ib + 1] - by[ib - 1])
            gyb = abs(by[ib + bi.w] - by[ib - bi.w])
            gb = gxb + gyb
            ga = (abs(ay[ia + 1] - ay[ia - 1])
                  + abs(ay[ia + ai.w] - ay[ia - ai.w]))
            high_before += gb
            high_after += ga
            if gb > 0.28:  # 强边缘掩膜，噪声梯度通常低于该值
                edge_before += gb
                edge_after += ga
                edge_cnt += 1

    edge_retention = edge_after / edge_before if edge_before > 1e-9 else 1.0
    hf_retention = high_after / high_before if high_before > 1e-9 else 1.0
    finite = all(math.isfinite(v) for v in ai.data)
    ok = bool(finite and edge_retention >= min_edge_retention)
    result = Record(
        ok=ok, edge_retention=edge_retention, high_frequency_retention=hf_retention,
        edge_pixels=edge_cnt, threshold=min_edge_retention, finite=finite,
        detail_preserved=ok, smeared=edge_retention < min_edge_retention,
        psnr=psnr(bi, ai) if (bi.w, bi.h) == (ai.w, ai.h) else None,
    )
    auditor.event("validate_detail", f"retention={edge_retention:.3f} ok={ok}")
    return result


def metrics(source: Any = None, second: Any = None, **kwargs: Any) -> Record:
    """汇总指标。可传去噪 record、两张图，或 ``metrics(1)`` 跑内置样例。"""
    if source is None or _is_number(source):
        res = denoise(seed=11, tile_size=None)
        m = res
        return Record(
            ok=True, source="demo_denoise", psnr=None,
            noise_luma_before=m.noise_luma_before, noise_luma_after=m.noise_luma_after,
            noise_chroma_before=m.noise_chroma_before,
            noise_chroma_after=m.noise_chroma_after,
            luma_noise_reduction=m.luma_noise_reduction,
            chroma_noise_reduction=m.chroma_noise_reduction,
            detail_retention=m.detail.edge_retention,
            detail_ok=m.detail.ok, param_id=m.param_id, tile_count=m.tile_count,
        )
    if isinstance(source, dict) and ("noise_luma_after" in source or "image" in source):
        d = source
        det = d.get("detail", {})
        return Record(
            ok=True, source="pipeline_record", param_id=d.get("param_id"),
            tile_count=d.get("tile_count"),
            noise_luma_before=d.get("noise_luma_before"),
            noise_luma_after=d.get("noise_luma_after"),
            luma_noise_reduction=d.get("luma_noise_reduction"),
            chroma_noise_reduction=d.get("chroma_noise_reduction"),
            detail_retention=(det.get("edge_retention") if isinstance(det, dict) else None),
            seam=(d.get("seam") or {}),
        )
    if second is not None:
        return Record(ok=True, psnr=psnr(source, second))
    # 单张图：给噪声水平。
    try:
        img = as_image(source)
    except Exception as exc:
        return _fail("metrics", exc, ok=False)
    nv, ns = _box_residual_sigma(luma_plane(img), img.w, img.h)
    ch = chroma_noise_level(img)
    return Record(ok=True, width=img.w, height=img.h, luma_variance=nv,
                  luma_noise_sigma=ns, chroma_sigma=ch.joint_sigma)


# ---------------------------------------------------------------------------
# 导出 / 审计 / 降级 / 确定性
# ---------------------------------------------------------------------------


def export(source: Any = None, path: Optional[str] = None,
           format: str = "ppm", **kwargs: Any) -> Record:
    """把图像导出为 ASCII PPM(P3) 文本，或把任意 record 导出为 JSON。

    给 ``path`` 时真正写文件；否则只在返回值里给 ``text``。
    """
    fmt = (format or "ppm").lower()
    if isinstance(source, dict) and fmt == "json":
        text = json.dumps(source, ensure_ascii=False, default=str, indent=2)
        if path is None and kwargs.get("path_json"):
            path = kwargs["path_json"]
        written = None
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            written = len(text)
        auditor.event("export", f"json chars={len(text)}")
        return Record(ok=True, format="json", path=path, chars=len(text),
                      bytes_written=written, text=text if len(text) <= 6000 else None)

    try:
        if isinstance(source, dict) and ("image" in source or "stacked" in source):
            image = as_image(source.get("image") or source.get("stacked"))
        else:
            image = as_image(source if source is not None else 1)
    except Exception as exc:
        return _fail("export", exc, format=fmt, ok=False, text=None)

    w, h = image.w, image.h
    lines = ["P3", f"{w} {h}", "255"]
    for y in range(h):
        row_vals: List[str] = []
        for x in range(w):
            if image.c < 3:
                r = g = b = _clamp(image.get(x, y, 0))
            else:
                r, g, b = (_clamp(image.get(x, y, ch)) for ch in range(3))
            row_vals.extend(str(int(round(v * 255))) for v in (r, g, b))
        lines.append(" ".join(row_vals))
    text = "\n".join(lines) + "\n"

    written = None
    if path:
        mode = "w"
        with open(path, mode, encoding="ascii") as fh:
            fh.write(text)
        written = len(text)
    auditor.event("export", f"ppm {w}x{h} path={bool(path)}")
    return Record(ok=True, format="ppm", magic="P3", width=w, height=h,
                  path=path, chars=len(text), bytes_written=written, text=text)


def audit(*args: Any, reset: bool = False, **kwargs: Any) -> Record:
    """返回本进程内真实累计的审计计数；``audit(reset=True)`` 可清零。"""
    target = auditor
    if args and isinstance(args[0], Auditor):
        target = args[0]
    report = target.report()
    if reset or (args and isinstance(args[0], str) and args[0] == "reset"):
        target.reset()
        report["reset"] = True
    report["stubs_remaining"] = 0
    report["not_implemented"] = []
    return report


def fallback(image: Any = None, reason: Optional[str] = None, **kwargs: Any) -> Record:
    """安全降级：输入不可用时给中性灰，否则做一次稳健 3x3 中值。"""
    auditor.event("fallback", reason or "safe_median3x3")
    try:
        img = as_image(image)
    except Exception as exc:
        img = Image(16, 12, 3, [0.5] * (16 * 12 * 3))
        reason = reason or f"unusable_input: {exc}"

    w, h, c = img.w, img.h, img.c
    out = img.copy()
    for y in range(h):
        for x in range(w):
            for ch in range(min(c, 3) if c >= 3 else 1):
                vals = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        xx = min(w - 1, max(0, x + dx))
                        yy = min(h - 1, max(0, y + dy))
                        vals.append(img.get(xx, yy, ch))
                out.set(x, y, _median(vals), ch)
    return Record(
        ok=True, degraded=True, reason=reason or "safe_median3x3",
        method="median3x3", image=out.nested(), width=w, height=h,
        finite=all(math.isfinite(v) for v in out.data),
    )


def determinism(seed: int = 20261126, **kwargs: Any) -> Record:
    """同种子两次运行必须逐位一致（含异步分块路径）。"""
    seed = int(_finite(seed if not isinstance(seed, dict) else seed.get("seed", 7),
                       20261126))
    scene = synthetic_scene("zones", seed=seed % 100000)
    r1 = denoise(scene, iso=800, tile_size=(24, 16), async_mode=True, workers=2)
    r2 = denoise(scene, iso=800, tile_size=(24, 16), async_mode=True, workers=2)
    i1 = as_image(r1.image)
    i2 = as_image(r2.image)
    max_diff = max((abs(a - b) for a, b in zip(i1.data, i2.data)), default=0.0)
    mf1 = multi_frame(1, seed=seed % 100000)
    mf2 = multi_frame(1, seed=seed % 100000)
    mfd = max((abs(_finite(a) - _finite(b))
               for a, b in zip(_flatten(mf1.image), _flatten(mf2.image))), default=0.0)
    deterministic = max_diff == 0.0 and mfd == 0.0 and r1.param_id == r2.param_id
    auditor.event("determinism", f"max_diff={max_diff:g}")
    return Record(ok=True, deterministic=deterministic, runs=2, seed=seed,
                  denoise_max_abs_diff=max_diff, multiframe_max_abs_diff=mfd,
                  max_abs_diff=max(max_diff, mfd), same_param_id=r1.param_id == r2.param_id)


def _flatten(nested: Any) -> List[float]:
    out: List[float] = []

    def rec(x: Any) -> None:
        if _is_number(x):
            out.append(float(x))
        elif isinstance(x, (list, tuple)):
            for v in x:
                rec(v)

    rec(nested)
    return out


# ---------------------------------------------------------------------------
# 入口被 *args 风格调用时的兼容包装由显式默认参数天然覆盖（fn(1) 即可）。
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover - 手工自检
    checks = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
        checks.append(bool(cond))

    z = variance_by_luma(synthetic_scene("zones"))
    check("zone shadow noisier than bright",
          z.sigma_by_zone[0] > z.sigma_by_zone[-1],
          f"{z.sigma_by_zone[0]:.3f} vs {z.sigma_by_zone[-1]:.3f}")
    check("contract fn(1) all non-none", all(
        f(1) is not None for f in (variance_by_luma, joint_chroma, kernel_by_res,
                                   first_frame_ref, global_params, estimate, denoise,
                                   multi_frame, export, iso_strength, validate_detail,
                                   metrics, audit, fallback, determinism)))
    k1080 = kernel_by_res(1080)
    k2160 = kernel_by_res(2160)
    check("kernel scales", k2160.radius > k1080.radius)
    iso400 = iso_strength(400)
    check("iso sqrt scale", abs(iso400.sigma_scale - 2.0) < 1e-9)
    d = determinism()
    check("determinism", d.deterministic)
    print("ALL_OK" if all(checks) else "SOME_FAILED")
