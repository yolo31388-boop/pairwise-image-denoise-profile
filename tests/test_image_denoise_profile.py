import pytest
import imaging.image_denoise_profile as mod

CASES = [
    ("variance_by_luma", (1,)),
    ("joint_chroma", (1,)),
    ("kernel_by_res", (1,)),
    ("first_frame_ref", (1,)),
    ("global_params", (1,)),
    ("estimate", (1,)),
    ("denoise", (1,)),
    ("multi_frame", (1,)),
    ("export", (1,)),
    ("iso_strength", (1,)),
    ("validate_detail", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
