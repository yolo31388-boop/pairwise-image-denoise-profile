"""六个修复点的行为回归测试（在 15 项契约冒烟之外验证真实语义）。

全部只用标准库 + pytest，固定随机种子，纯函数 / 小图，秒级跑完。
"""

import json
import math
import random

import pytest

import imaging.image_denoise_profile as mod


# --------------------------------------------------------------------------- #
# 测试场景构造
# --------------------------------------------------------------------------- #


def noisy_gray(w, h, base, sigma, seed):
    rng = random.Random(seed)
    data = []
    for y in range(h):
        for x in range(w):
            data.append(max(0.0, min(1.0, base + rng.gauss(0.0, sigma))))
    return mod.Image(w, h, 1, data)


def zone_image(w=48, h=32, seed=1):
    """左半暗、噪声大；右半亮、噪声小。"""
    rng = random.Random(seed)
    img = mod.Image(w, h, 1)
    for y in range(h):
        for x in range(w):
            if x < w // 2:
                v = 0.18 + rng.gauss(0.0, 0.08)
            else:
                v = 0.82 + rng.gauss(0.0, 0.012)
            img.set(x, y, max(0.0, min(1.0, v)), 0)
    return img


def dilution_image(w=50, h=40):
    """80% 完全平坦的纯色 + 20% 高噪声条带，复现“整图估计被拉低”。"""
    img = mod.Image(w, h, 1)
    rng = random.Random(2)
    for y in range(h):
        for x in range(w):
            v = 0.30 + rng.gauss(0.0, 0.07) if x < 10 else 0.60
            img.set(x, y, v, 0)
    return img


def coarse_grain_image(w=48, h=48, seed=3):
    """2x2 像素块共享一个随机偏移，模拟尺度 > 1px 的粗颗粒。"""
    rng = random.Random(seed)
    img = mod.Image(w, h, 1)
    for by in range(h // 2):
        for bx in range(w // 2):
            off = rng.gauss(0.0, 0.12)
            for dy in (0, 1):
                for dx in (0, 1):
                    x, y = 2 * bx + dx, 2 * by + dy
                    if x < w and y < h:
                        img.set(x, y, 0.5 + off + rng.gauss(0.0, 0.01), 0)
    return img


def coarse_rmse(img, radius, sigma):
    gp = mod.Record(radius_px=radius, sigma_spatial=sigma,
                    range_sigma_by_zone=[1.0] * 5)
    out = mod.denoise_luma_region(mod.luma_plane(img), img.w, img.h, gp,
                                  0, 0, img.w, img.h)
    return math.sqrt(sum((v - 0.5) ** 2 for v in out) / len(out))


# --------------------------------------------------------------------------- #
# 修复 1：分亮度分区估计
# --------------------------------------------------------------------------- #


def test_variance_by_luma_returns_partition():
    res = mod.variance_by_luma(zone_image())
    assert res.ok and res.zone_count == 5
    assert len(res.zones) == 5 and len(res.sigma_by_zone) == 5
    assert sum(res.counts) == 48 * 32
    assert all(z.count >= 0 for z in res.zones)
    assert "whole_variance" in res and "global_noise_sigma" in res


def test_dark_zone_noise_estimated_higher_than_bright():
    res = mod.variance_by_luma(zone_image())
    dark, bright = res.sigma_by_zone[0], res.sigma_by_zone[-1]
    assert 0.05 < dark < 0.12          # 接近注入的 0.08
    assert bright < 0.03
    assert dark > 3.0 * bright


def test_partition_estimate_not_diluted_by_large_flat_area():
    res = mod.variance_by_luma(dilution_image())
    strongest_zone = max(res.sigma_by_zone)
    # 大片纯色让整图 MAD 估计接近 0，分区仍能抓到噪声条带。
    assert strongest_zone > 1.8 * max(res.global_noise_sigma, 1e-6)
    assert strongest_zone > 0.04


def test_zone_adaptive_denoise_and_detail_kept():
    edges = mod.synthetic_scene("edges")
    out = mod.denoise(edges, tile_size=None)
    detail = mod.validate_detail(edges, out)
    assert detail.edge_retention >= 0.72
    assert detail.ok is True
    assert out.luma_zones == 5


# --------------------------------------------------------------------------- #
# 修复 2：色度联合
# --------------------------------------------------------------------------- #


def test_joint_chroma_removes_color_speckle():
    scene = mod.synthetic_scene("chroma", seed=4)
    res = mod.joint_chroma(scene, radius_px=2)
    assert res.ok and res.joint and res.channels_joint == 2
    assert res.chroma_noise_after < 0.65 * res.chroma_noise_before
    # Cb、Cr 两个色度平面都被压下去，而不是只修一个。
    assert res.cb_sigma_after < res.cb_sigma_before
    assert res.cr_sigma_after < res.cr_sigma_before
    assert len(res.image) == scene.h and len(res.image[0]) == scene.w


def test_full_denoise_reduces_chroma_noise():
    scene = mod.synthetic_scene("demo", seed=8)
    res = mod.denoise(scene, async_mode=True, tile_size=(18, 14))
    assert res.chroma_noise_after < res.chroma_noise_before
    assert res.joint_chroma is True


# --------------------------------------------------------------------------- #
# 修复 3：核按分辨率缩放
# --------------------------------------------------------------------------- #


def test_kernel_scales_linearly_with_resolution():
    k720 = mod.kernel_by_res(720)
    k1080 = mod.kernel_by_res(1080)
    k2160 = mod.kernel_by_res(2160)
    assert k1080.radius == pytest.approx(2.0)
    assert k2160.radius == pytest.approx(4.0)
    assert k720.radius < k1080.radius < k2160.radius
    assert k2160.radius / k1080.radius == pytest.approx(2.0)
    assert k2160.diameter > k1080.diameter


def test_kernel_accepts_image_and_config():
    img = mod.synthetic_scene("demo")
    assert mod.kernel_by_res(img).height == img.h
    assert mod.kernel_by_res({"height": 2160}).radius == pytest.approx(4.0)


def test_wider_kernel_removes_coarse_grain():
    img = coarse_grain_image()
    r1 = coarse_rmse(img, 1, 0.6)
    r2 = coarse_rmse(img, 2, 1.2)
    assert r2 < 0.8 * r1       # 0.089 -> 0.058，粗颗粒明显更干净


# --------------------------------------------------------------------------- #
# 修复 4：以首帧为参考
# --------------------------------------------------------------------------- #


def test_first_frame_is_absolute_reference():
    res = mod.first_frame_ref(1)
    assert res.ok
    assert res.reference_index == 0
    assert res.reference_is_first_frame is True
    assert res.chained is False
    assert all(a == 0 for a in res.aligned_to)
    assert res.reference_source == "frames[0]"


def test_first_frame_policy_does_not_accumulate_bias():
    frames = mod._demo_frames(6)

    def biased(ref, target):
        est = mod.estimate_translation(ref, target, max_shift=2)
        return mod.Record(dx=est.dx + 0.45, dy=est.dy)

    first = mod.multi_frame(frames, policy="first", estimator=biased)
    prev = mod.multi_frame(frames, policy="previous", estimator=biased)
    assert abs(first.cumulative_shifts[-1][0]) < 1.5
    assert abs(prev.cumulative_shifts[-1][0]) > abs(first.cumulative_shifts[-1][0]) + 1.0


def test_alignment_recovers_shifts_and_stack_is_sharper():
    frames = mod._demo_frames(4)
    res = mod.multi_frame(frames)
    assert res.shifts[0] == (0.0, 0.0)
    assert res.shifts[1] == pytest.approx((1.0, 0.0), abs=1e-9)
    assert res.shifts[2] == pytest.approx((0.0, 1.0), abs=1e-9)
    assert res.metrics.sharpness_stacked >= res.metrics.sharpness_unaligned
    assert res.sharper_than_unaligned is True


# --------------------------------------------------------------------------- #
# 修复 5：参数全局一致 + 分块拼接
# --------------------------------------------------------------------------- #


def test_param_id_stable_and_iso_sensitive():
    scene = mod.synthetic_scene("demo", seed=8)
    a = mod.global_params(scene, iso=100).param_id
    b = mod.global_params(scene, iso=100).param_id
    c = mod.global_params(scene, iso=1600).param_id
    assert a == b and len(a) == 12
    assert c != a


def test_async_tiles_share_one_param_and_match_full_frame():
    scene = mod.synthetic_scene("demo", seed=8)
    full = mod.denoise(scene, tile_size=None)
    tiled = mod.denoise(scene, tile_size=(18, 14), async_mode=True, workers=3)

    a = mod.as_image(full.image).data
    b = mod.as_image(tiled.image).data
    max_diff = max(abs(x - y) for x, y in zip(a, b))
    assert max_diff == 0.0                       # 同参数 + 全图邻域 => 逐位一致
    assert tiled.all_params_global is True
    assert tiled.param_ids == [tiled.param_id]
    assert tiled.tile_count >= 2
    assert tiled.seam.bitwise_identical is True
    assert tiled.seam.max_abs_diff_vs_global == 0.0


def test_tile_grid_partitions_all_pixels_once():
    boxes = mod._tile_grid(40, 30, 18, 14)
    covered = set()
    for x0, y0, x1, y1 in boxes:
        for y in range(y0, y1):
            for x in range(x0, x1):
                assert (x, y) not in covered     # 不重叠
                covered.add((x, y))
    assert len(covered) == 40 * 30               # 无遗漏


# --------------------------------------------------------------------------- #
# 修复 6：强度随 ISO 缩放
# --------------------------------------------------------------------------- #


def test_iso_strength_sqrt_model():
    assert mod.iso_strength(100).sigma_scale == pytest.approx(1.0)
    assert mod.iso_strength(400).sigma_scale == pytest.approx(2.0)
    assert mod.iso_strength(900).sigma_scale == pytest.approx(3.0)
    assert mod.iso_strength(3200).sigma_scale == pytest.approx(math.sqrt(32.0))
    low = mod.iso_strength(100)
    high = mod.iso_strength(12800)
    assert high.sigma_scale > low.sigma_scale and high.aggressiveness > low.aggressiveness


def test_high_iso_denoises_flat_noise_more_aggressively():
    img = noisy_gray(32, 32, 0.5, 0.06, 77)
    low = mod.denoise(img, iso=100, tile_size=None)
    high = mod.denoise(img, iso=6400, tile_size=None)
    assert high.iso_scale > low.iso_scale
    assert high.noise_luma_after < low.noise_luma_after


# --------------------------------------------------------------------------- #
# 细节校验 / 指标 / 确定性 / 降级 / 导出 / 审计
# --------------------------------------------------------------------------- #


def test_validate_detects_smear():
    edges = mod.synthetic_scene("edges")
    y = mod.luma_plane(edges)
    blurred = mod.box_blur_plane(y, edges.w, edges.h, radius=3)
    smear = mod.Image(edges.w, edges.h, 1, blurred)
    verdict = mod.validate_detail(edges, smear)
    assert verdict.ok is False and verdict.smeared is True
    assert verdict.edge_retention < 0.5


def test_validate_identical_image():
    img = mod.synthetic_scene("edges")
    verdict = mod.validate_detail(img, img)
    assert verdict.edge_retention == pytest.approx(1.0, abs=1e-6)
    assert verdict.ok is True


def test_determinism_exact_including_async_tiles():
    res = mod.determinism()
    assert res.deterministic is True
    assert res.max_abs_diff == 0.0
    assert res.same_param_id is True


def test_fallback_for_unusable_input():
    res = mod.fallback(object())
    assert res.ok and res.degraded and res.finite
    assert len(res.image) == 12 and len(res.image[0]) == 16
    # denoise 主入口遇到脏输入也走降级而不是抛异常。
    via_denoise = mod.denoise(object())
    assert via_denoise.degraded is True


def test_export_ppm_and_json(tmp_path):
    res = mod.export(1)
    assert res.ok and res.magic == "P3" and res.text.startswith("P3")
    p = tmp_path / "out.ppm"
    written = mod.export(mod.synthetic_scene("demo"), str(p))
    assert written.path == str(p) and p.exists()
    assert p.read_text(encoding="ascii").startswith("P3")

    jp = tmp_path / "params.json"
    jres = mod.export(mod.global_params(1).params, format="json", path=str(jp))
    assert jres.ok and jp.exists()
    json.loads(jp.read_text(encoding="utf-8"))


def test_metrics_entry_shapes():
    demo = mod.metrics(1)
    assert demo.ok and demo.noise_luma_before >= 0
    rec = mod.denoise(mod.synthetic_scene("demo", seed=8), tile_size=None)
    again = mod.metrics(rec)
    assert again.ok and again.param_id == rec.param_id


def test_auditor_real_counters():
    aud = mod.Auditor()
    aud.incr("tiles", 3)
    aud.event("run", "x")
    rep = aud.report()
    assert rep.counters["tiles"] == 3
    assert rep.counters["run"] == 1
    aud.reset()
    assert aud.report().counters == {}
    assert mod.audit().ok and mod.audit().stubs_remaining == 0


def test_all_entries_fn1_non_none_and_json_safe():
    names = ["variance_by_luma", "joint_chroma", "kernel_by_res", "first_frame_ref",
             "global_params", "estimate", "denoise", "multi_frame", "export",
             "iso_strength", "validate_detail", "metrics", "audit", "fallback",
             "determinism"]
    for name in names:
        result = getattr(mod, name)(1)
        assert result is not None
        json.dumps(result, default=str)   # 证据必须可序列化


def test_garbage_inputs_return_error_records_not_raise():
    for name in ("variance_by_luma", "joint_chroma", "estimate", "global_params",
                 "export", "validate_detail", "metrics"):
        res = getattr(mod, name)(object())
        assert res is not None and res.ok is False
    assert mod.iso_strength("not-a-number").ok is True   # 数值入口自身兜底


def test_no_top_level_numpy_dependency():
    # 评分 venv 只有 pytest：模块顶层不得 import numpy/cv2/PIL。
    import ast
    import pathlib
    src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_modules = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_modules.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_modules.append(node.module.split(".")[0])
    assert "numpy" not in top_modules
    assert "cv2" not in top_modules and "PIL" not in top_modules
