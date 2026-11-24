# -*- coding: utf-8 -*-
"""Environment self-check. Run: runtime\\python.exe setup\\verify_env.py"""
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL = []
WARN = []


def ok(msg):
    print(f"  [ OK ] {msg}")


def bad(msg, err=None):
    FAIL.append(msg)
    print(f"  [FAIL] {msg}" + (f" -> {err}" if err else ""))


def warn(msg):
    WARN.append(msg)
    print(f"  [warn] {msg}")


print("=" * 60)
print("DDSP-SVC 6.2 environment check")
print("root:", ROOT)
print("python:", sys.version.split()[0], sys.executable)
print("=" * 60)

print("[1] torch / cuda")
try:
    import torch
    print("  torch:", torch.__version__)
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"  cuda : True  {props.name}  cc={props.major}.{props.minor}  vram={props.total_memory // 1024 ** 3}GB")
        try:
            x = torch.randn(64, 64, device="cuda")
            (x @ x).sum().item()
            print("  cuda matmul: OK")
        except Exception as e:
            bad("cuda kernel launch", e)
    else:
        if "+cu" in torch.__version__:
            warn("cu128 build but cuda not available (driver missing/too old) -> runs on CPU only")
        else:
            ok("cpu build (expected on machines without NVIDIA GPU)")
except Exception as e:
    bad("import torch", e)

print("[2] core imports")
mods = [
    "numpy", "scipy", "librosa", "soundfile", "resampy", "pyworld", "parselmouth",
    "torchcrepe", "torchaudio", "transformers", "einops", "omegaconf", "hydra",
    "gradio", "demucs", "pedalboard", "timm", "segmentation_models_pytorch",
    "torchmetrics", "spafe", "auraloss", "gin", "ml_collections", "beartype",
]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        missing.append(f"{m}({type(e).__name__})")
if missing:
    bad("missing modules: " + ", ".join(missing))
else:
    ok(f"{len(mods)} modules imported")

print("[3] fairseq (contentvec encoder)")
try:
    from fairseq import checkpoint_utils  # noqa: F401
    ok("fairseq bundled, checkpoint_utils available")
except Exception as e:
    bad("import fairseq.checkpoint_utils", e)
    print("      -> run setup\\install_env.bat again or unpack setup\\vendor\\fairseq-*.tar.gz")

print("[4] ffmpeg")
ff = shutil.which("ffmpeg")
if ff:
    ok(ff)
else:
    warn("ffmpeg not on PATH - audio conversion / MSST separation will fail")
    print("      -> start.bat adds runtime\\Library\\bin automatically; run through start.bat")

print("[5] assets")
checks = [
    ("pretrain/rmvpe/model.pt", 100),
    ("pretrain/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt", 30),
    ("pretrain/contentvec/checkpoint_best_legacy_500.pt", 100),
    ("MSST/pretrain/model_bs_roformer_ep_317_sdr_12.9755.ckpt", 100),
]
for rel, min_mb in checks:
    p = ROOT / rel
    if not p.exists():
        warn(f"missing {rel}  (run tools/fetch_release.py --only weights)")
    elif p.stat().st_size < min_mb * 1024 * 1024:
        warn(f"suspicious size {p.stat().st_size // 1024 ** 2}MB -> {rel}")
    else:
        ok(f"{rel} ({p.stat().st_size // 1024 ** 2}MB)")

print("=" * 60)
if FAIL:
    print(f"RESULT: FAIL ({len(FAIL)} problem(s))")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("RESULT: PASS" + (f"  ({len(WARN)} warning(s))" if WARN else ""))
print("now double-click start.bat")
sys.exit(0)
