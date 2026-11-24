import os
import re
import sys
import time
import json
import yaml
import zipfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf
import librosa
import gradio as gr

BASE_DIR = Path(__file__).resolve().parent
MSST_DIR = Path(os.environ["MSST_DIR"]) if os.environ.get("MSST_DIR") else (
    (BASE_DIR / "MSST") if (BASE_DIR / "MSST").exists() else (BASE_DIR.parent / "MSST")
)
os.chdir(BASE_DIR)

MODELS_DIR = BASE_DIR / "models"
EXP_DIR = BASE_DIR / "exp"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_COVER_DIR = RESULTS_DIR / "01_翻唱结果"
RESULTS_ONECLICK_DIR = RESULTS_DIR / "02_一条龙翻唱"
RESULTS_MSST_DIR = RESULTS_DIR / "03_模型分离结果"
RESULTS_MIX_DIR = RESULTS_DIR / "04_混音与后期处理"
RESULTS_TMP_DIR = RESULTS_DIR / "_tmp"
RAW_DIR = BASE_DIR / "raw"
CACHE_DIR = BASE_DIR / "cache"
TRAIN_DIR = BASE_DIR / "data" / "train"
VAL_DIR = BASE_DIR / "data" / "val"
TRAIN_AUDIO_DIR = TRAIN_DIR / "audio"
VAL_AUDIO_DIR = VAL_DIR / "audio"
CONFIG_PATH = BASE_DIR / "configs" / "reflow.yaml"
BASE_PRETRAIN_CKPT = BASE_DIR / "exp" / "reflow-test" / "model_0.pt"
TRAIN_LOG_PATH = BASE_DIR / "train_webui.log"

for d in [MODELS_DIR, EXP_DIR, RESULTS_DIR, RESULTS_COVER_DIR, RESULTS_ONECLICK_DIR, RESULTS_MSST_DIR, RESULTS_MIX_DIR, RESULTS_TMP_DIR, RAW_DIR, CACHE_DIR, TRAIN_AUDIO_DIR, VAL_AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_PROC = None
PE_CHOICES = [
    ("rmvpe（推荐 · 抗混响与气声最强）", "rmvpe"),
    ("fcpe（快速 · 高音稳定）", "fcpe"),
    ("crepe（高精度 · 较慢）", "crepe"),
]

METHOD_CHOICES = [
    ("auto（跟随模型默认）", "auto"),
    ("rk4（推荐 · 音质细腻）", "rk4"),
    ("euler（速度更快）", "euler"),
]

HARMONY_MODES = [
    "只翻唱主唱，保留原曲和声并入伴奏（推荐默认：副歌最饱满自然）",
    "主唱与和声分别去混响后都翻唱（彻底替换为目标音色）",
    "彻底去除和声，只留主唱 + 纯伴奏（极简干声模式）",
]


# ==================== 暖白纸感工作室主题与可开关动态效果 ====================
FORCE_LIGHT_JS = """() => {
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
}"""

STUDIO_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.rose,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
).set(
    body_background_fill="#EAE3DF",
    body_background_fill_dark="#EAE3DF",
    body_text_color="#2A1E21",
    body_text_color_dark="#2A1E21",
    body_text_color_subdued="#7B6D71",
    body_text_color_subdued_dark="#7B6D71",
    background_fill_primary="#FDFBF9",
    background_fill_primary_dark="#FDFBF9",
    background_fill_secondary="#F3EFEA",
    background_fill_secondary_dark="#F3EFEA",
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_border_color="transparent",
    block_border_color_dark="transparent",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="#4B3B3F",
    block_label_text_color_dark="#4B3B3F",
    block_title_text_color="#2A1E21",
    block_title_text_color_dark="#2A1E21",
    block_info_text_color="#87787C",
    block_info_text_color_dark="#87787C",
    input_background_fill="#F4F0EC",
    input_background_fill_dark="#F4F0EC",
    input_border_color="rgba(42, 30, 33, 0.09)",
    input_border_color_dark="rgba(42, 30, 33, 0.09)",
    button_primary_background_fill="#721B35",
    button_primary_background_fill_dark="#721B35",
    button_primary_background_fill_hover="#5A1429",
    button_primary_background_fill_hover_dark="#5A1429",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#EFECE8",
    button_secondary_background_fill_dark="#EFECE8",
    button_secondary_background_fill_hover="#E5DFDA",
    button_secondary_background_fill_hover_dark="#E5DFDA",
    button_secondary_text_color="#2A1E21",
    button_secondary_text_color_dark="#2A1E21",
    button_secondary_border_color="rgba(42, 30, 33, 0.08)",
    button_secondary_border_color_dark="rgba(42, 30, 33, 0.08)",
    border_color_primary="transparent",
    border_color_primary_dark="transparent",
    checkbox_background_color="#FFFFFF",
    checkbox_background_color_dark="#FFFFFF",
    checkbox_background_color_selected="#721B35",
    checkbox_background_color_selected_dark="#721B35",
    checkbox_border_color="rgba(42, 30, 33, 0.22)",
    checkbox_border_color_dark="rgba(42, 30, 33, 0.22)",
    checkbox_label_background_fill="transparent",
    checkbox_label_background_fill_dark="transparent",
    checkbox_label_background_fill_selected="transparent",
    checkbox_label_background_fill_selected_dark="transparent",
    checkbox_label_text_color="#2A1E21",
    checkbox_label_text_color_dark="#2A1E21",
    slider_color="#721B35",
    slider_color_dark="#721B35",
)

CUSTOM_CSS = """
:root {
    --studio-serif: "Songti SC", "Noto Serif SC", "Source Han Serif SC", "STSong", Georgia, serif;
    --studio-sans: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", -apple-system, BlinkMacSystemFont, sans-serif;
    --studio-sidebar: #EAE3DF;
    --studio-canvas: #FDFBF9;
    --studio-aside: #F2EEEB;
    --studio-soft: #F3EFEA;
    --studio-wine: #721B35;
    --studio-wine-hover: #5A1429;
    --studio-wine-deep: #3B1A22;
    --studio-ink: #24171A;
    --studio-muted: #7C6E72;
}

/* ===== 精致暖调工作室动效关键帧 ===== */
@keyframes ambientDriftTop {
    0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.72; }
    50% { transform: translate(-34px, 24px) scale(1.14); opacity: 1; }
}
@keyframes ambientDriftBottom {
    0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.65; }
    50% { transform: translate(28px, -22px) scale(1.12); opacity: 0.95; }
}
@keyframes silkRibbon {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}
@keyframes markPulse {
    0%, 100% { box-shadow: 0 2px 8px rgba(114, 27, 53, 0.28), 0 0 0 0 rgba(114, 27, 53, 0.28); }
    50% { box-shadow: 0 2px 12px rgba(114, 27, 53, 0.42), 0 0 0 6px rgba(114, 27, 53, 0); }
}
@keyframes panelFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes eqBounce {
    0%, 100% { transform: scaleY(0.28); }
    50% { transform: scaleY(1); }
}
@keyframes btnShimmer {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

#studio-shell, #studio-shell * {
    box-sizing: border-box;
}

html, body {
    background: var(--studio-sidebar) !important;
    color: var(--studio-ink) !important;
    font-family: var(--studio-sans) !important;
    max-width: 100vw !important;
    overflow-x: hidden !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    background: var(--studio-sidebar) !important;
    color: var(--studio-ink) !important;
}
.gradio-container > .main {
    padding: 0 !important;
}
footer { display: none !important; }

#fx-style-holder {
    position: absolute !important;
    height: 0 !important;
    width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    border: none !important;
}

#studio-shell {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    min-height: 100vh !important;
    width: 100% !important;
    gap: 0 !important;
    background: var(--studio-sidebar) !important;
    align-items: stretch !important;
}

/* 左侧边栏 */
#studio-sidebar {
    flex: 0 0 222px !important;
    width: 222px !important;
    max-width: 222px !important;
    min-width: 222px !important;
    background: var(--studio-sidebar) !important;
    padding: 24px 14px 20px 14px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
    border-right: none !important;
    position: sticky !important;
    top: 0 !important;
    height: 100vh !important;
    box-sizing: border-box !important;
}

.brand-header {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 4px 10px 18px 10px;
}
.brand-mark {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: radial-gradient(circle at 32% 32%, #A63354 0%, #721B35 55%, #3B1A22 100%);
    position: relative;
    flex-shrink: 0;
    animation: markPulse 3.4s ease-in-out infinite;
}
.brand-mark::after {
    content: "";
    position: absolute;
    top: 3px;
    right: 3px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--studio-sidebar);
}
.brand-title {
    font-family: var(--studio-serif);
    font-size: 21px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: var(--studio-ink) !important;
    line-height: 1.1;
}
.brand-sub {
    font-size: 11px;
    color: var(--studio-muted) !important;
    letter-spacing: 0.06em;
    margin-top: 2px;
}

#studio-sidebar button.nav-btn {
    width: 100% !important;
    justify-content: flex-start !important;
    text-align: left !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #4A3A3E !important;
    font-family: var(--studio-serif) !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 11px 14px !important;
    border-radius: 12px !important;
    letter-spacing: 0.03em !important;
    transition: background 0.22s ease, color 0.22s ease, transform 0.2s ease !important;
}
#studio-sidebar button.nav-btn:hover {
    background: rgba(255, 255, 255, 0.50) !important;
    color: var(--studio-ink) !important;
    transform: translateX(3px) !important;
}
#studio-sidebar button.nav-btn.nav-active {
    background: #DDD4CF !important;
    color: var(--studio-ink) !important;
    font-weight: 600 !important;
    box-shadow: inset 3px 0 0 var(--studio-wine) !important;
}

.sidebar-spacer {
    flex: 1 1 auto !important;
    min-height: 24px !important;
}

.author-credit {
    padding: 4px 12px 6px 12px;
    font-size: 12.5px;
    color: #6B5C60 !important;
    line-height: 1.6;
}
.author-credit span {
    color: #4D3D41 !important;
    font-weight: 500;
}

#fx-cb-item {
    padding: 2px 10px 6px 10px !important;
    background: transparent !important;
    border: none !important;
}
#fx-cb-item label {
    background: transparent !important;
    border: none !important;
    padding: 2px 0 !important;
    font-size: 12.5px !important;
    color: #5A4B4F !important;
    box-shadow: none !important;
}

.studio-status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--studio-wine-deep);
    padding: 10px 14px;
    border-radius: 999px;
    font-size: 12.5px;
    letter-spacing: 0.02em;
    box-shadow: 0 6px 18px rgba(59, 26, 34, 0.22);
    white-space: nowrap;
}
.studio-status-pill,
.studio-status-pill span {
    color: #F7ECEF !important;
    white-space: nowrap !important;
}
.studio-status-pill .pill-sub {
    color: rgba(247, 236, 239, 0.72) !important;
    font-size: 11px !important;
    margin-left: auto;
}
.eq-bars {
    display: inline-flex;
    align-items: center;
    gap: 2.5px;
    height: 14px;
    flex-shrink: 0;
}
.eq-bars span {
    display: block;
    width: 2.5px;
    height: 12px;
    border-radius: 2px;
    background: #F4DCE3 !important;
    transform-origin: center;
    animation: eqBounce 1.05s ease-in-out infinite;
}
.eq-bars span:nth-child(2) { animation-delay: 0.15s; height: 14px; }
.eq-bars span:nth-child(3) { animation-delay: 0.30s; height: 8px; }
.eq-bars span:nth-child(4) { animation-delay: 0.45s; height: 13px; }
.eq-bars span:nth-child(5) { animation-delay: 0.60s; height: 10px; }

#studio-content-wrap {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    padding: 10px 10px 10px 0 !important;
    gap: 0 !important;
    background: var(--studio-sidebar) !important;
}

.workspace-row {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 0 !important;
    min-height: calc(100vh - 20px) !important;
    width: 100% !important;
    align-items: stretch !important;
    animation: panelFadeIn 0.32s cubic-bezier(0.22, 1, 0.36, 1) !important;
}
.workspace-row.hide,
.workspace-row.hidden,
.hide,
.hidden {
    display: none !important;
}

/* 中间暖白纸感主卡 + 双层暖霞呼吸光晕 + 顶部丝绸流光细线 */
.studio-main-card {
    flex: 1.55 1 0% !important;
    min-width: 0 !important;
    background: var(--studio-canvas) !important;
    border-radius: 22px 0 0 22px !important;
    padding: 32px 32px 36px 32px !important;
    box-shadow: -2px 0 24px rgba(59, 26, 34, 0.04) !important;
    position: relative !important;
    overflow: hidden !important;
    gap: 14px !important;
    border-top: 2px solid transparent !important;
    background-image:
        linear-gradient(var(--studio-canvas), var(--studio-canvas)),
        linear-gradient(90deg, rgba(114,27,53,0.08), rgba(168,58,91,0.52), rgba(212,163,115,0.48), rgba(114,27,53,0.08)) !important;
    background-origin: border-box !important;
    background-clip: padding-box, border-box !important;
    background-size: 100% 100%, 220% 100% !important;
    animation: silkRibbon 9s linear infinite !important;
}
.studio-main-card::before {
    content: "";
    position: absolute;
    top: -110px;
    right: -80px;
    width: 380px;
    height: 380px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(156, 44, 77, 0.11) 0%, rgba(214, 162, 134, 0.06) 45%, transparent 72%);
    pointer-events: none;
    animation: ambientDriftTop 14s ease-in-out infinite;
}
.studio-main-card::after {
    content: "";
    position: absolute;
    bottom: -120px;
    left: -90px;
    width: 340px;
    height: 340px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(212, 163, 115, 0.11) 0%, rgba(114, 27, 53, 0.04) 48%, transparent 72%);
    pointer-events: none;
    animation: ambientDriftBottom 17s ease-in-out infinite;
}
.studio-main-card > * {
    position: relative;
    z-index: 1;
}

/* 最右侧记录栏 */
.studio-aside-panel {
    flex: 1 1 0% !important;
    min-width: 270px !important;
    max-width: 390px !important;
    background: var(--studio-aside) !important;
    border-radius: 0 22px 22px 0 !important;
    padding: 32px 24px !important;
    border-left: 1px solid rgba(42, 30, 33, 0.05) !important;
    gap: 14px !important;
}

/* 精准底部对齐行（让下拉框和右侧按钮底部完全齐平） */
.aligned-row {
    align-items: flex-end !important;
    gap: 10px !important;
}
.aligned-row > * {
    min-width: 0 !important;
}

.page-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 2px;
}
.page-title {
    font-family: var(--studio-serif);
    font-size: 24px;
    font-weight: 600;
    color: var(--studio-ink) !important;
    letter-spacing: 0.02em;
    line-height: 1.25;
}
.page-meta {
    font-size: 13px;
    color: var(--studio-muted) !important;
}
.aside-title {
    font-family: var(--studio-serif);
    font-size: 18px;
    font-weight: 600;
    color: var(--studio-ink) !important;
    margin-bottom: 4px;
}
.aside-desc {
    font-size: 13px;
    color: var(--studio-muted) !important;
    line-height: 1.6;
}
.section-label {
    font-family: var(--studio-serif);
    font-size: 16px;
    font-weight: 600;
    color: var(--studio-ink) !important;
    margin: 6px 0 2px 0;
}

.soft-guide-box {
    background: #F6F1ED;
    border-left: 3px solid var(--studio-wine);
    border-radius: 10px;
    padding: 11px 14px;
    font-size: 13px;
    line-height: 1.7;
    color: #4A393D !important;
}
.soft-guide-box b {
    color: var(--studio-ink) !important;
}

.danger-confirm-box {
    background: #FDF1F3 !important;
    border: 1px solid rgba(159, 24, 57, 0.28) !important;
    border-left: 4px solid #9F1839 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-top: 6px !important;
}
.danger-warn-text {
    color: #8A1632 !important;
    font-size: 13px;
    line-height: 1.65;
    font-weight: 500;
    margin-bottom: 8px;
}

/* 彻底清除 Gradio 5 默认灰色 .form / .block 外壳 */
.gradio-container .form,
.gradio-container .block,
.gradio-container .styler {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    overflow: visible !important;
}
.gradio-container .form {
    gap: 10px !important;
}
.gradio-container .block {
    padding: 0 !important;
}

/* 消除 Dropdown 双层套框，保留干净单层圆角框 */
.gradio-container .secondary-wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea,
.gradio-container .wrap-inner {
    background: var(--studio-soft) !important;
    border: 1px solid rgba(42, 30, 33, 0.10) !important;
    border-radius: 12px !important;
    color: var(--studio-ink) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.gradio-container .wrap-inner:hover,
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: rgba(114, 27, 53, 0.30) !important;
    box-shadow: 0 0 0 3px rgba(114, 27, 53, 0.06) !important;
}

/* 音频上传与播放区悬浮微浮起（严格锁死宽度，防止手机端上传音频后波形撑爆视口） */
.gradio-container [data-testid="audio"],
.gradio-container .audio-container,
.gradio-container .file-preview,
.gradio-container .upload-container {
    background: var(--studio-soft) !important;
    border: 1px solid rgba(42, 30, 33, 0.07) !important;
    border-radius: 16px !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    overflow: hidden !important;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.24s ease !important;
}
.gradio-container [data-testid="audio"] *,
.gradio-container .waveform-container,
.gradio-container .waveform-container canvas {
    max-width: 100% !important;
    min-width: 0 !important;
}
.gradio-container [data-testid="audio"]:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(114, 27, 53, 0.26) !important;
    box-shadow: 0 8px 22px rgba(114, 27, 53, 0.08) !important;
}

.studio-aside-panel textarea {
    background: #FAF7F4 !important;
    border: 1px solid rgba(42, 30, 33, 0.08) !important;
    font-size: 12.5px !important;
    line-height: 1.6 !important;
}

/* 按钮丝绸流光与统一高度 */
button.primary {
    background: linear-gradient(110deg, #721B35 0%, #94294A 45%, #721B35 100%) !important;
    background-size: 220% 100% !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 10px 22px !important;
    min-height: 42px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    white-space: nowrap !important;
    box-shadow: 0 5px 16px rgba(114, 27, 53, 0.22) !important;
    overflow: hidden !important;
    animation: btnShimmer 6s ease-in-out infinite !important;
    transition: transform 0.2s ease, box-shadow 0.24s ease !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 26px rgba(114, 27, 53, 0.34) !important;
}
button.secondary {
    background: #EFECE8 !important;
    color: var(--studio-ink) !important;
    border: 1px solid rgba(42, 30, 33, 0.08) !important;
    border-radius: 999px !important;
    padding: 9px 16px !important;
    min-height: 42px !important;
    font-size: 13.5px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    transition: background 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease !important;
}
button.secondary:hover {
    background: #E5DFDA !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(42, 30, 33, 0.07) !important;
}
button.stop {
    background: #F6E4E7 !important;
    color: #8F1D3A !important;
    border: 1px solid rgba(143, 29, 58, 0.22) !important;
    border-radius: 999px !important;
    min-height: 40px !important;
    font-size: 13.5px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    transition: background 0.18s ease, transform 0.18s ease !important;
}
button.stop:hover {
    background: #EED2D7 !important;
    transform: translateY(-1px) !important;
}

.gradio-container .accordion {
    background: #F8F5F2 !important;
    border: 1px solid rgba(42, 30, 33, 0.07) !important;
    border-radius: 16px !important;
    padding: 12px 16px !important;
    margin-top: 4px !important;
    transition: border-color 0.22s ease, box-shadow 0.22s ease !important;
}
.gradio-container .accordion:hover {
    border-color: rgba(114, 27, 53, 0.16) !important;
    box-shadow: 0 4px 16px rgba(59, 26, 34, 0.04) !important;
}
.gradio-container .accordion > button {
    font-family: var(--studio-serif) !important;
    font-size: 15px !important;
    color: var(--studio-ink) !important;
    font-weight: 600 !important;
}

.file-checklist .wrap {
    max-height: 220px !important;
    overflow-y: auto !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
    padding: 8px 10px !important;
    background: #FAF7F4 !important;
    border: 1px solid rgba(42, 30, 33, 0.08) !important;
    border-radius: 12px !important;
}

#msst-mode-radio .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
}
#msst-mode-radio label {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 14px !important;
    padding: 12px 16px !important;
    transition: background 0.2s ease, border-color 0.2s ease, transform 0.18s ease !important;
}
#msst-mode-radio label:hover {
    background: #F5F1ED !important;
    transform: translateX(3px) !important;
}
#msst-mode-radio label.selected {
    background: #EFECE8 !important;
    border-color: rgba(114, 27, 53, 0.22) !important;
    font-weight: 600 !important;
}

.status-card {
    background: var(--studio-canvas);
    border: 1px solid rgba(114, 27, 53, 0.12);
    border-radius: 14px;
    padding: 14px 16px;
    color: var(--studio-ink) !important;
    font-size: 12.5px;
    line-height: 1.75;
    word-break: break-all;
}
.status-card code {
    background: #EFEAE6;
    color: var(--studio-wine) !important;
    padding: 2px 6px;
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
}

@media (max-width: 960px) {
    #studio-shell { flex-direction: column !important; flex-wrap: wrap !important; }
    #studio-sidebar {
        position: relative !important;
        height: auto !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        padding: 14px 12px 10px 12px !important;
        gap: 8px !important;
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        align-items: center !important;
    }
    #studio-sidebar > .html-container:first-child { width: 100% !important; }
    .brand-header { padding: 2px 4px 6px 4px !important; }
    #studio-sidebar button.nav-btn {
        flex: 1 1 calc(50% - 6px) !important;
        width: auto !important;
        justify-content: center !important;
        text-align: center !important;
        padding: 9px 10px !important;
        font-size: 14px !important;
        background: rgba(255, 255, 255, 0.36) !important;
    }
    #studio-sidebar button.nav-btn.nav-active {
        box-shadow: inset 0 -2.5px 0 var(--studio-wine) !important;
    }
    .sidebar-spacer { display: none !important; }
    .author-credit { width: auto !important; padding: 2px 6px !important; font-size: 12px !important; }
    #fx-cb-item { width: auto !important; padding: 2px 6px !important; }
    .studio-status-pill { padding: 7px 12px !important; font-size: 12px !important; }
    #studio-content-wrap {
        width: 100% !important;
        max-width: 100vw !important;
        min-width: 0 !important;
        padding: 0 8px 18px 8px !important;
        overflow-x: hidden !important;
    }
    .workspace-row {
        flex-direction: column !important;
        flex-wrap: wrap !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        min-height: auto !important;
        overflow-x: hidden !important;
    }
    .studio-main-card {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        border-radius: 18px !important;
        padding: 18px 14px 22px 14px !important;
        overflow-x: hidden !important;
    }
    .studio-main-card > *,
    .studio-aside-panel > * {
        max-width: 100% !important;
        min-width: 0 !important;
    }
    .studio-main-card::before, .studio-main-card::after { width: 200px !important; height: 200px !important; }
    .studio-aside-panel {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        border-radius: 18px !important;
        margin-top: 10px !important;
        padding: 20px 14px !important;
        border-left: none !important;
        overflow-x: hidden !important;
    }
    .page-title { font-size: 20px !important; }
    .aligned-row { flex-wrap: wrap !important; }
    .aligned-row > * { flex: 1 1 100% !important; }
    button.primary, button.secondary, button.stop { min-height: 44px !important; width: 100% !important; }
    .file-checklist .wrap { max-height: 180px !important; }
}
"""

UI_PREFS_PATH = BASE_DIR / ".ui_prefs.json"


def load_ui_prefs():
    default = {"enable_fx": True}
    if UI_PREFS_PATH.exists():
        try:
            data = json.loads(UI_PREFS_PATH.read_text(encoding="utf-8"))
            if isinstance(data.get("enable_fx"), bool):
                default["enable_fx"] = data["enable_fx"]
        except Exception:
            pass
    return default


def build_style_html(enable_fx: bool) -> str:
    if enable_fx:
        return """<style id="dynamic-fx-style">
        button.primary:hover { animation: btnShimmer 1.4s linear infinite !important; }
        </style>"""
    return """<style id="dynamic-fx-style">
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
    .studio-main-card {
        background-image: none !important;
        background: var(--studio-canvas) !important;
        border-top: none !important;
    }
    .studio-main-card::before,
    .studio-main-card::after {
        display: none !important;
    }
    .eq-bars span {
        transform: scaleY(0.55) !important;
    }
    .gradio-container [data-testid="audio"]:hover,
    button.primary:hover,
    button.secondary:hover,
    #studio-sidebar button.nav-btn:hover,
    #msst-mode-radio label:hover {
        transform: none !important;
    }
    </style>"""


def build_sidebar_status_html(enable_fx: bool) -> str:
    mode_text = "动效已开" if enable_fx else "静态极速"
    return f"""
    <div class="studio-status-pill">
        <span class="eq-bars"><span></span><span></span><span></span><span></span><span></span></span>
        <span>工作台就绪</span>
        <span class="pill-sub">{mode_text}</span>
    </div>
    """


def apply_ui_prefs(enable_fx: bool):
    try:
        UI_PREFS_PATH.write_text(json.dumps({"enable_fx": enable_fx}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return build_style_html(enable_fx), build_sidebar_status_html(enable_fx)



# ==================== 模型目录与步数解析（默认最高步数） ====================
def _extract_step_num(path_str: str) -> int:
    name = Path(path_str).stem
    nums = re.findall(r"(\d+)", name)
    return int(nums[-1]) if nums else -1


def list_model_dirs():
    dirs = []
    for root_dir, tag in [(MODELS_DIR, "模型库"), (EXP_DIR, "训练目录")]:
        if not root_dir.exists():
            continue
        for sub in sorted(root_dir.iterdir()):
            if not sub.is_dir():
                continue
            pts = [p for p in sub.glob("*.pt") if not (root_dir == EXP_DIR and p.name == "model_0.pt")]
            if pts:
                rel = str(sub.relative_to(BASE_DIR))
                dirs.append((f"{sub.name} ({tag})", rel))
    if not dirs and BASE_PRETRAIN_CKPT.exists():
        dirs.append(("reflow-test (官方底模)", "exp/reflow-test"))
    return dirs


def list_steps_in_dir(dir_rel: str):
    if not dir_rel:
        return []
    target = BASE_DIR / dir_rel
    if not target.exists():
        return []
    pts = list(target.glob("*.pt"))
    if target.resolve() == (EXP_DIR / "reflow-test").resolve() and len(pts) > 1:
        pts = [p for p in pts if p.name != "model_0.pt"]
    pts.sort(key=lambda p: _extract_step_num(p.name), reverse=True)
    return [(p.name, str(p.relative_to(BASE_DIR))) for p in pts]


def get_default_model_and_step():
    dirs = list_model_dirs()
    init_dir = dirs[0][1] if dirs else None
    steps = list_steps_in_dir(init_dir) if init_dir else []
    init_step = steps[0][1] if steps else None
    return dirs, init_dir, steps, init_step


def on_model_dir_change(dir_rel: str):
    steps = list_steps_in_dir(dir_rel)
    return gr.update(choices=steps, value=steps[0][1] if steps else None)


def refresh_all_models():
    dirs, init_dir, steps, init_step = get_default_model_and_step()
    return gr.update(choices=dirs, value=init_dir), gr.update(choices=steps, value=init_step)


def list_exp_dirs():
    dirs = []
    if EXP_DIR.exists():
        for sub in sorted(EXP_DIR.iterdir()):
            if sub.is_dir():
                dirs.append(str(sub.relative_to(BASE_DIR)))
    if "exp/reflow-test" not in dirs:
        dirs.insert(0, "exp/reflow-test")
    return dirs


def _safe_stem(p: str, max_len: int = 36) -> str:
    import re
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", Path(p).stem).strip("_")
    return (s[:max_len] if s else "audio")


def list_recent_results():
    items = []
    scan_dirs = [RESULTS_COVER_DIR, RESULTS_DIR]
    seen = set()
    for d in scan_dirs:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in {".wav", ".mp3", ".flac"} and not f.name.startswith("_"):
                rp = str(f.resolve())
                if rp in seen:
                    continue
                seen.add(rp)
                try:
                    mtime = f.stat().st_mtime
                except OSError:
                    mtime = 0
                items.append((mtime, f.name, str(f)))
    items.sort(key=lambda x: x[0], reverse=True)
    return [(name, path) for _, name, path in items[:50]]


def ensure_wav_44k(src_path: str, dst_path: str, mono: bool = False, sr: int = 44100, extra_af: str = ""):
    cmd = ["ffmpeg", "-y", "-i", str(src_path), "-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    if extra_af:
        cmd += ["-af", extra_af]
    cmd += ["-sample_fmt", "s16", str(dst_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not Path(dst_path).exists():
        raise RuntimeError(f"FFmpeg 转码失败：\n{res.stderr[-600:]}")


# ==================== MSST 引擎 ====================
def get_msst_models():
    presets = {}
    bs_ckpt = MSST_DIR / "pretrain" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    bs_cfg = MSST_DIR / "configs_custom" / "model_bs_roformer_ep_317_sdr_12.9755.yaml"
    kj_ckpt = MSST_DIR / "pretrain" / "melbandroformer_vocals_kj.safetensors"
    kj_cfg = MSST_DIR / "configs" / "KimberleyJensen" / "config_vocals_mel_band_roformer_kj.yaml"

    if bs_ckpt.exists() and bs_cfg.exists():
        presets["1. 人声/伴奏分离 (MSST BS-Roformer Viperx 12.97 推荐)"] = ("bs_roformer", str(bs_cfg), str(bs_ckpt))
        if kj_ckpt.exists() and kj_cfg.exists():
            presets["1b. 人声/伴奏分离-备选 (MSST Mel-Band-Roformer KJ)"] = ("mel_band_roformer", str(kj_cfg), str(kj_ckpt))
    elif kj_ckpt.exists() and kj_cfg.exists():
        presets["1. 人声/伴奏分离 (MSST Mel-Band-Roformer KJ)"] = ("mel_band_roformer", str(kj_cfg), str(kj_ckpt))

    kar_ckpt = MSST_DIR / "pretrain" / "mel_band_roformer_karaoke_becruily.ckpt"
    kar_cfg = MSST_DIR / "configs_custom" / "config_karaoke_becruily.yaml"
    if kar_ckpt.exists() and kar_cfg.exists():
        presets["2. 主唱/和声分离 (MSST Karaoke Becruily，分离主唱与和声)"] = ("mel_band_roformer", str(kar_cfg), str(kar_ckpt))

    dereverb_ckpt = MSST_DIR / "pretrain" / "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"
    dereverb_cfg = MSST_DIR / "configs_custom" / "dereverb_mel_band_roformer_anvuew.yaml"
    if dereverb_ckpt.exists() and dereverb_cfg.exists():
        presets["3. 人声去混响/去回声 (MSST Dereverb Anvuew)"] = ("mel_band_roformer", str(dereverb_cfg), str(dereverb_ckpt))

    pretrain_dir = MSST_DIR / "pretrain"
    cfg_custom = MSST_DIR / "configs_custom"
    known = {
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "melbandroformer_vocals_kj.safetensors",
        "mel_band_roformer_karaoke_becruily.ckpt",
        "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    }
    if pretrain_dir.exists():
        for ckpt in sorted(pretrain_dir.iterdir()):
            if ckpt.suffix in [".ckpt", ".pt", ".pth", ".safetensors"] and ckpt.name not in known:
                yaml_cand = cfg_custom / f"{ckpt.stem}.yaml"
                if yaml_cand.exists():
                    mtype = "bs_roformer" if "bs_roformer" in ckpt.name.lower() else "mel_band_roformer"
                    presets[f"自定义 ({ckpt.name})"] = (mtype, str(yaml_cand), str(ckpt))

    if not presets:
        presets[f"未检测到可用 MSST 模型（请检查 {MSST_DIR / 'pretrain'}）"] = ("", "", "")
    return presets


def _msst_is_single_target(cfg_path: str) -> bool:
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return bool(cfg.get("training", {}).get("target_instrument"))
    except Exception:
        return True


def _msst_stem_rank(wav_path: str) -> tuple:
    """严格只按输出文件名（不含父目录路径）排序：主轨(人声/主唱/去混响)=0，副轨(伴奏/和声/混响)=1"""
    stem = Path(wav_path).stem.lower()
    if stem in {"noreverb", "dry", "vocals", "vocal", "lead"}:
        return (0, stem)
    if "noreverb" in stem or "dry" in stem:
        return (0, stem)
    if "vocal" in stem and "inst" not in stem and "no_vocal" not in stem:
        return (0, stem)
    if stem in {"instrumental", "other", "reverb", "backing", "karaoke"} or "inst" in stem or "reverb" in stem or "other" in stem:
        return (1, stem)
    return (2, stem)


def run_msst_core(audio_path: str, model_type: str, cfg_path: str, ckpt_path: str, out_dir: Path, use_tta: bool = False, extract_inst: bool = True):
    in_tmp = out_dir / "_in"
    in_tmp.mkdir(parents=True, exist_ok=True)
    wav_in = in_tmp / (Path(audio_path).stem + ".wav")
    ensure_wav_44k(audio_path, str(wav_in), mono=False)

    # 双输出模型（target_instrument 为 null，如 Karaoke Becruily）本身已输出双轨，不能再传 --extract_instrumental
    need_extract_inst = bool(extract_inst and _msst_is_single_target(cfg_path))

    cmd = [
        sys.executable,
        str(MSST_DIR / "inference.py"),
        "--model_type", model_type,
        "--config_path", cfg_path,
        "--start_check_point", ckpt_path,
        "--input_folder", str(in_tmp),
        "--store_dir", str(out_dir),
    ]
    if use_tta:
        cmd.append("--use_tta")
    if need_extract_inst:
        cmd.append("--extract_instrumental")

    res = subprocess.run(cmd, cwd=str(MSST_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    shutil.rmtree(in_tmp, ignore_errors=True)
    if res.returncode != 0:
        raise RuntimeError(f"MSST 推理失败：\n{res.stderr[-800:]}")

    raw_wavs = [str(p) for p in out_dir.rglob("*.wav") if not p.name.startswith("_")]
    wavs = sorted(raw_wavs, key=_msst_stem_rank)
    if not wavs:
        raise RuntimeError("MSST 运行结束但未生成输出音轨。")
    return wavs, res.stdout


def _translate_raw_stem(raw_path: str, fallback: str) -> str:
    s = Path(raw_path).stem.lower()
    mapping = {
        "noreverb": "去混响干声",
        "dry": "去混响干声",
        "reverb": "混响残差",
        "vocals": "人声(vocals)",
        "vocal": "人声(vocal)",
        "lead": "主唱(lead)",
        "instrumental": "纯伴奏(instrumental)",
        "no_vocals": "纯伴奏(no_vocals)",
        "karaoke": "和声或伴奏(karaoke)",
        "other": "副轨(other)",
        "drums": "鼓组(drums)",
        "bass": "贝斯(bass)",
    }
    for k, v in mapping.items():
        if s == k or k in s:
            return v
    return f"{fallback}({Path(raw_path).stem})"


def _msst_step_and_labels(model_label: str, stems: list[str]) -> tuple[str, str, str]:
    if model_label.startswith("1.") or model_label.startswith("1b."):
        return "人声伴奏分离", "人声", "纯伴奏"
    if model_label.startswith("2."):
        return "主唱和声分离", "主唱", "原曲和声"
    if model_label.startswith("3."):
        return "去混响分离", "去混响干声", "混响残差"
    n1 = _translate_raw_stem(stems[0], "主轨") if len(stems) > 0 else "主轨"
    n2 = _translate_raw_stem(stems[1], "副轨") if len(stems) > 1 else "副轨"
    return "自定义模型分离", n1, n2


def run_msst_ui(audio_file, model_label, use_tta, extract_inst):
    if not audio_file:
        return None, None, "请先上传待处理的音频文件。"
    presets = get_msst_models()
    mtype, cfg_p, ckpt_p = presets[model_label]
    if not ckpt_p:
        return None, None, "未检测到可用 MSST 模型。"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    song_tag = _safe_stem(audio_file)
    step_tag, _, _ = _msst_step_and_labels(model_label, [])
    out_dir = RESULTS_MSST_DIR / f"{song_tag}_{step_tag}_{ts}"
    work_sub = out_dir / "_raw"
    work_sub.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        stems, _ = run_msst_core(audio_file, mtype, cfg_p, ckpt_p, work_sub, use_tta=use_tta, extract_inst=extract_inst)
        elapsed = time.time() - t0

        stem_name = Path(audio_file).stem
        _, n1, n2 = _msst_step_and_labels(model_label, stems)

        saved = []
        s1, s2 = None, None
        if len(stems) > 0:
            p1 = out_dir / f"{stem_name}_{n1}.wav"
            shutil.copy2(stems[0], p1)
            s1 = str(p1)
            saved.append(s1)
        if len(stems) > 1 and extract_inst:
            p2 = out_dir / f"{stem_name}_{n2}.wav"
            shutil.copy2(stems[1], p2)
            s2 = str(p2)
            saved.append(s2)

        shutil.rmtree(work_sub, ignore_errors=True)
        return s1, s2, f"MSST 分离完成（耗时 {elapsed:.1f}s，模型：{model_label}）\n保存目录：results/03_模型分离结果/{out_dir.name}/\n输出文件：\n" + "\n".join(saved)
    except Exception as e:
        shutil.rmtree(work_sub, ignore_errors=True)
        return None, None, str(e)


def merge_harmony_with_inst(harmony_file, inst_file, harmony_db, inst_db):
    if not harmony_file or not inst_file:
        return None, "请同时上传分离出的「和声音轨」和「纯伴奏音轨」。"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_wav = RESULTS_MIX_DIR / f"inst_with_harmony_{ts}.wav"
    filt = f"[0:a]volume={harmony_db}dB[h];[1:a]volume={inst_db}dB[i];[h][i]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,alimiter=limit=0.95[out]"
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(harmony_file), "-i", str(inst_file), "-filter_complex", filt, "-map", "[out]", "-ar", "44100", str(out_wav)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if res.returncode != 0 or not out_wav.exists():
        return None, f"合并失败：\n{res.stderr[-600:]}"
    return str(out_wav), f"已生成带和声伴奏：{out_wav}（可直接用于混音或下载）。"


def run_audio_format_clean(audio_file, target_sr, to_mono, do_highpass, do_loudnorm):
    if not audio_file:
        return None, "请先上传需要处理的音频文件。"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_wav = RESULTS_MIX_DIR / f"cleaned_{ts}.wav"
    filters = []
    if do_highpass:
        filters.append("highpass=f=80")
    if do_loudnorm:
        filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    af = ",".join(filters)
    try:
        ensure_wav_44k(audio_file, str(out_wav), mono=bool(to_mono), sr=int(target_sr), extra_af=af)
        return str(out_wav), f"清洗完成：{out_wav.name}"
    except Exception as e:
        return None, str(e)


def run_mix_audio(vocal_file, inst_file, vocal_db, inst_db, auto_loudnorm, add_reverb, add_comp, out_fmt):
    if not vocal_file or not inst_file:
        return None, "请同时上传「翻唱人声」与「伴奏音轨」。"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "mp3" if str(out_fmt).lower() == "mp3" else "wav"
    out_path = RESULTS_MIX_DIR / f"mix_{ts}.{ext}"
    v_chain = []
    if auto_loudnorm:
        v_chain.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    if add_comp:
        v_chain.append("acompressor=threshold=-18dB:ratio=3:attack=5:release=60")
    if add_reverb:
        v_chain.append("aecho=0.8:0.88:35|55:0.18|0.12")
    v_chain.append(f"volume={vocal_db}dB")
    v_flt = ",".join(v_chain)
    filt = f"[0:a]{v_flt}[v];[1:a]volume={inst_db}dB[i];[v][i]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,alimiter=limit=0.95[out]"
    cmd = ["ffmpeg", "-y", "-i", str(vocal_file), "-i", str(inst_file), "-filter_complex", filt, "-map", "[out]", "-ar", "44100"]
    if ext == "mp3":
        cmd += ["-b:a", "320k"]
    cmd.append(str(out_path))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not out_path.exists():
        return None, f"混音失败：\n{res.stderr[-600:]}"
    return str(out_path), f"混音完成：{out_path.name}"


# ==================== DDSP-SVC 推理与一条龙 ====================
def run_inference(audio_file, model_step_path, key_shift, formant_shift, reg_shift, pitch_extractor, infer_method, infer_step, t_start, threhold, f0_min, f0_max):
    if not audio_file:
        return None, "请先上传或载入干声音频。", gr.update()
    if not model_step_path:
        return None, "请先选择模型权重 (.pt)。", gr.update()
    ckpt_p = BASE_DIR / model_step_path
    if not ckpt_p.exists():
        return None, f"模型权重不存在：{ckpt_p}", gr.update()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    in_wav = RESULTS_TMP_DIR / f"_in_{ts}.wav"
    song_tag = _safe_stem(audio_file)
    out_wav = RESULTS_COVER_DIR / f"cover_{song_tag}_{ckpt_p.parent.name}_{ts}.wav"
    try:
        ensure_wav_44k(audio_file, str(in_wav), mono=True)
        cmd = [
            sys.executable, "main_reflow.py",
            "-m", str(ckpt_p),
            "-i", str(in_wav),
            "-o", str(out_wav),
            "-k", str(key_shift),
            "-f", str(formant_shift),
            "-v", str(reg_shift),
            "-pe", str(pitch_extractor),
            "-method", str(infer_method),
            "-step", str(infer_step),
            "-ts", str(t_start),
            "-th", str(threhold),
            "-fmin", str(f0_min),
            "-fmax", str(f0_max),
        ]
        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        in_wav.unlink(missing_ok=True)
        if res.returncode != 0 or not out_wav.exists():
            return None, f"推理失败：\n{res.stdout}\n{res.stderr}", gr.update()
        elapsed = time.time() - t0
        hist = list_recent_results()
        log_txt = f"推理完成（耗时 {elapsed:.1f}s）\n模型：{model_step_path}\n输出：{out_wav.name}\n{res.stdout[-500:]}"
        return str(out_wav), log_txt, gr.update(choices=hist, value=str(out_wav))
    except Exception as e:
        in_wav.unlink(missing_ok=True)
        return None, f"推理异常：{e}", gr.update()


def run_batch_inference(files, folder_path, model_step_path, key_shift, formant_shift, pitch_extractor, infer_method, infer_step):
    if not model_step_path:
        return None, "请先在上方选择模型目录与具体步数 (.pt)。"
    model_path = BASE_DIR / model_step_path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_in = RESULTS_TMP_DIR / f"_batch_in_{ts}"
    tmp_out = RESULTS_TMP_DIR / f"_batch_out_{ts}"
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    count = 0
    if files:
        for f in files:
            sp = Path(f if isinstance(f, str) else f.name)
            if sp.exists():
                ensure_wav_44k(str(sp), str(tmp_in / f"{sp.stem}.wav"), mono=True)
                count += 1
    elif folder_path and Path(folder_path).is_dir():
        for sp in sorted(Path(folder_path).iterdir()):
            if sp.is_file() and sp.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
                ensure_wav_44k(str(sp), str(tmp_in / f"{sp.stem}.wav"), mono=True)
                count += 1

    if count == 0:
        shutil.rmtree(tmp_in, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None, "未找到可处理的音频文件（请勾选、上传或填写有效目录）。"

    cmd = [
        sys.executable, "batch_infer.py",
        "-m", str(model_path),
        "-i", str(tmp_in),
        "-o", str(tmp_out),
        "-k", str(key_shift),
        "-f", str(formant_shift),
        "-pe", str(pitch_extractor),
        "-method", str(infer_method),
        "-step", str(infer_step),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    shutil.rmtree(tmp_in, ignore_errors=True)
    if res.returncode != 0:
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None, f"批量推理失败：\n{res.stderr[-800:]}"

    zip_path = RESULTS_COVER_DIR / f"batch_cover_{ts}.zip"
    outs = list(tmp_out.glob("*.wav"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in outs:
            zf.write(fp, fp.name)
    shutil.rmtree(tmp_out, ignore_errors=True)
    elapsed = time.time() - t0
    return str(zip_path), f"批量转换完成！共 {len(outs)} 个文件（耗时 {elapsed:.1f}s）。"


def list_oneclick_tasks():
    dirs = []
    if RESULTS_ONECLICK_DIR.exists():
        dirs.extend([d for d in RESULTS_ONECLICK_DIR.iterdir() if d.is_dir()])
    if RESULTS_DIR.exists():
        dirs.extend([d for d in RESULTS_DIR.glob("oneclick_*") if d.is_dir()])
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    items = []
    for d in dirs[:50]:
        cnt = len(list(d.glob("*.wav")))
        items.append((f"{d.name} ({cnt} 轨)", str(d)))
    return items


def load_oneclick_task_details(task_dir_str):
    if not task_dir_str:
        return gr.update(choices=[], value=None), None, ""
    d = Path(task_dir_str)
    if not d.exists():
        return gr.update(choices=[], value=None), None, "任务目录不存在。"
    wavs = sorted(d.glob("*.wav"), key=lambda f: f.name)
    choices = [(f.name, str(f)) for f in wavs]
    default_track = None
    for f in wavs:
        if f.name.startswith("05_"):
            default_track = str(f)
            break
    if not default_track and wavs:
        default_track = str(wavs[-1])
    log_file = d / "run_log.txt"
    log_text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else "未找到 run_log.txt。"
    return gr.update(choices=choices, value=default_track), default_track, log_text


def _run_step_with_heartbeat(fn, *args, **kwargs):
    import threading
    res_box = {}
    err_box = {}

    def _worker():
        try:
            res_box["val"] = fn(*args, **kwargs)
        except Exception as exc:
            err_box["err"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t, res_box, err_box


def run_one_click_cover(song_file, model_step_path, key_shift, pitch_extractor, harmony_mode, harmony_db, vocal_db, inst_db, auto_loudnorm, add_reverb):
    import re
    import traceback

    if not song_file:
        yield None, None, None, "请先上传带伴奏的原曲文件。", gr.update(), gr.update(), gr.update(), ""
        return
    if not model_step_path:
        yield None, None, None, "请先在上方选择推理模型与步数 (.pt)。", gr.update(), gr.update(), gr.update(), ""
        return
    presets = get_msst_models()
    k1 = next((k for k in presets if k.startswith("1.")), None)
    k2 = next((k for k in presets if k.startswith("2.")), None)
    k3 = next((k for k in presets if k.startswith("3.")), None)
    if not (k1 and k2 and k3):
        yield None, None, None, "MSST 三步模型不完整，无法执行一条龙翻唱。", gr.update(), gr.update(), gr.update(), ""
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    song_stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", Path(song_file).stem).strip("_")[:40] or "song"
    task_dir = RESULTS_ONECLICK_DIR / f"{song_stem}_{ts}"
    work_dir = task_dir / "_tmp"
    task_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_dir / "run_log.txt"

    logs = [
        f"=== 一条龙翻唱任务追溯日志 ===",
        f"任务目录: {task_dir}",
        f"原曲文件: {Path(song_file).name}",
        f"模型权重: {model_step_path}",
        f"参数: 升降调={key_shift}, 音高提取={pitch_extractor}, 和声模式={harmony_mode}, 主唱={vocal_db}dB, 和声={harmony_db}dB, 伴奏={inst_db}dB, 响度对齐={auto_loudnorm}, 混响={add_reverb}",
    ]

    def _flush_log(line=None):
        if line:
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
        log_text = "\n".join(logs)
        log_path.write_text(log_text, encoding="utf-8")
        return log_text

    def _save_stem(src_p, dest_name):
        if src_p and Path(src_p).exists():
            dst = task_dir / dest_name
            shutil.copy2(src_p, dst)
            return str(dst)
        return None

    t_all = time.time()
    p_03_dry = None
    p_04_vconv = None
    p_05_mix = None

    try:
        # Step 1: 人声与伴奏分离
        t0 = time.time()
        log_text = _flush_log("[1/5] 开始人声与伴奏分离...")
        t, res, err = _run_step_with_heartbeat(run_msst_core, song_file, *presets[k1], work_dir / "s1", use_tta=False, extract_inst=True)
        while t.is_alive():
            t.join(timeout=3.0)
            if t.is_alive():
                yield None, None, None, f"⏳ [1/5] 正在分离人声与伴奏...（已耗时 {int(time.time()-t0)}s）", gr.update(), gr.update(), gr.update(), log_text
        if "err" in err:
            raise err["err"]
        stems1, _ = res["val"]
        vocals, inst = stems1[0], (stems1[1] if len(stems1) > 1 else None)
        p_01_voc = _save_stem(vocals, "01_人声分离.wav")
        p_01_inst = _save_stem(inst, "01_纯伴奏.wav")
        log_text = _flush_log(f"[1/5] 人声分离完成（耗时 {time.time()-t0:.1f}s）：人声={bool(p_01_voc)}, 伴奏={bool(p_01_inst)}")

        # Step 2: 主唱与和声分离
        t0 = time.time()
        log_text = _flush_log("[2/5] 开始分离主唱与原曲和声...")
        yield None, None, None, "⏳ [2/5] 正在分离主唱与原曲和声...", gr.update(), gr.update(), gr.update(), log_text
        t, res, err = _run_step_with_heartbeat(run_msst_core, vocals, *presets[k2], work_dir / "s2", use_tta=False, extract_inst=True)
        while t.is_alive():
            t.join(timeout=3.0)
            if t.is_alive():
                yield None, None, None, f"⏳ [2/5] 正在分离主唱与原曲和声...（已耗时 {int(time.time()-t0)}s）", gr.update(), gr.update(), gr.update(), log_text
        if "err" in err:
            raise err["err"]
        stems2, _ = res["val"]
        lead, harm = stems2[0], (stems2[1] if len(stems2) > 1 else None)
        p_02_lead = _save_stem(lead, "02_主唱.wav")
        p_02_harm = _save_stem(harm, "02_原曲和声.wav")
        log_text = _flush_log(f"[2/5] 主唱/和声分离完成（耗时 {time.time()-t0:.1f}s）：主唱={bool(p_02_lead)}, 和声={bool(p_02_harm)}")

        # Step 3: 主唱去混响
        t0 = time.time()
        log_text = _flush_log("[3/5] 开始对主唱干声去混响...")
        yield None, None, None, "⏳ [3/5] 正在对主唱干声去混响...", gr.update(), gr.update(), gr.update(), log_text
        t, res, err = _run_step_with_heartbeat(run_msst_core, lead, *presets[k3], work_dir / "s3", use_tta=False, extract_inst=True)
        while t.is_alive():
            t.join(timeout=3.0)
            if t.is_alive():
                yield None, None, None, f"⏳ [3/5] 正在对主唱干声去混响...（已耗时 {int(time.time()-t0)}s）", gr.update(), gr.update(), gr.update(), log_text
        if "err" in err:
            raise err["err"]
        stems3, _ = res["val"]
        dry_lead = stems3[0]
        p_03_dry = _save_stem(dry_lead, "03_主唱去混响干声.wav")
        log_text = _flush_log(f"[3/5] 去混响完成（耗时 {time.time()-t0:.1f}s）")

        # Step 4: DDSP 主唱翻唱推理
        t0 = time.time()
        log_text = _flush_log("[4/5] 开始 DDSP-SVC 主唱翻唱推理...")
        yield p_03_dry, None, None, "⏳ [4/5] 正在执行 DDSP-SVC 主唱翻唱推理...", gr.update(), gr.update(), gr.update(), log_text
        t, res, err = _run_step_with_heartbeat(run_inference, p_03_dry, model_step_path, key_shift, 0, 0, pitch_extractor, "auto", "50", 0.0, -60, 50, 1100)
        while t.is_alive():
            t.join(timeout=3.0)
            if t.is_alive():
                yield p_03_dry, None, None, f"⏳ [4/5] 正在执行 DDSP-SVC 主唱翻唱推理...（已耗时 {int(time.time()-t0)}s）", gr.update(), gr.update(), gr.update(), log_text
        if "err" in err:
            raise err["err"]
        vconv_out, infer_msg, _ = res["val"]
        if not vconv_out:
            log_text = _flush_log(f"[4/5] 翻唱推理失败：{infer_msg}")
            shutil.rmtree(work_dir, ignore_errors=True)
            tasks = list_oneclick_tasks()
            tr_upd, tr_val, _ = load_oneclick_task_details(str(task_dir))
            yield p_03_dry, None, None, f"翻唱推理失败：{infer_msg}", gr.update(choices=tasks, value=str(task_dir)), tr_upd, tr_val, log_text
            return
        p_04_vconv = _save_stem(vconv_out, "04_翻唱主唱干声.wav")
        log_text = _flush_log(f"[4/5] 主唱翻唱完成（耗时 {time.time()-t0:.1f}s）")

        # Step 4b & 5: 和声处理 + 最终混音
        t0 = time.time()
        log_text = _flush_log("[5/5] 开始和声处理与最终混音...")
        yield p_03_dry, p_04_vconv, None, "⏳ [5/5] 正在处理和声并合成最终混音...", gr.update(), gr.update(), gr.update(), log_text

        final_inst = inst
        if inst and harm and harmony_mode == HARMONY_MODES[0]:
            merged_inst, _ = merge_harmony_with_inst(harm, inst, harmony_db, 0)
            if merged_inst:
                final_inst = merged_inst
            log_text = _flush_log("[5/5] 已保留原曲和声并合入伴奏。")
        elif inst and harm and harmony_mode == HARMONY_MODES[1]:
            log_text = _flush_log("[5/5] 正在对和声去混响并翻唱...")
            t, res, err = _run_step_with_heartbeat(run_msst_core, harm, *presets[k3], work_dir / "s3_harm", use_tta=False, extract_inst=False)
            while t.is_alive():
                t.join(timeout=3.0)
                if t.is_alive():
                    yield p_03_dry, p_04_vconv, None, f"⏳ [5/5] 正在对和声去混响...（已耗时 {int(time.time()-t0)}s）", gr.update(), gr.update(), gr.update(), log_text
            if "err" in err:
                raise err["err"]
            stems_h3, _ = res["val"]
            harm_dry = stems_h3[0]
            t, res, err = _run_step_with_heartbeat(run_inference, str(harm_dry), model_step_path, key_shift, 0, 0, pitch_extractor, "auto", "50", 0.0, -60, 50, 1100)
            while t.is_alive():
                t.join(timeout=3.0)
                if t.is_alive():
                    yield p_03_dry, p_04_vconv, None, f"⏳ [5/5] 正在翻唱和声...（已耗时 {int(time.time()-t0)}s）", gr.update(), gr.update(), gr.update(), log_text
            if "err" in err:
                raise err["err"]
            hconv_out, _, _ = res["val"]
            if hconv_out:
                _save_stem(hconv_out, "04_翻唱和声干声.wav")
                merged_inst, _ = merge_harmony_with_inst(hconv_out, inst, harmony_db, 0)
                if merged_inst:
                    final_inst = merged_inst
            log_text = _flush_log("[5/5] 翻唱和声已合入伴奏。")
        else:
            log_text = _flush_log("[5/5] 已去除和声，仅保留纯伴奏。")

        mix_out, mix_msg = run_mix_audio(p_04_vconv, final_inst, vocal_db, inst_db, auto_loudnorm, add_reverb, True, "wav")
        p_05_mix = _save_stem(mix_out, "05_最终混音成品.wav") or mix_out
        shutil.rmtree(work_dir, ignore_errors=True)
        total_sec = time.time() - t_all
        log_text = _flush_log(f"一条龙翻唱全部完成！总耗时 {total_sec:.1f}s。{mix_msg}")
        tasks = list_oneclick_tasks()
        tr_upd, tr_val, _ = load_oneclick_task_details(str(task_dir))
        yield p_03_dry, p_04_vconv, p_05_mix, f"一条龙翻唱完成（总耗时 {total_sec:.1f}s）！所有过程音轨已归档至：{task_dir.name}\n{mix_msg}", gr.update(choices=tasks, value=str(task_dir)), tr_upd, tr_val, log_text
    except Exception as e:
        err_tb = traceback.format_exc()
        log_text = _flush_log(f"一条龙处理出错：{e}\n{err_tb}")
        shutil.rmtree(work_dir, ignore_errors=True)
        tasks = list_oneclick_tasks()
        tr_upd, tr_val, _ = load_oneclick_task_details(str(task_dir))
        yield p_03_dry, p_04_vconv, p_05_mix, f"一条龙处理出错：{e}（详情见下方追溯日志）", gr.update(choices=tasks, value=str(task_dir)), tr_upd, tr_val, log_text

# ==================== 数据集与训练管理 ====================
def inspect_dataset():
    tr_wavs = list(TRAIN_AUDIO_DIR.rglob("*.wav")) if TRAIN_AUDIO_DIR.exists() else []
    val_wavs = list(VAL_AUDIO_DIR.rglob("*.wav")) if VAL_AUDIO_DIR.exists() else []
    f0_dir = TRAIN_DIR / "f0"
    f0_cnt = len(list(f0_dir.rglob("*.npy"))) if f0_dir.exists() else 0

    durs = []
    for w in tr_wavs:
        try:
            durs.append(sf.info(str(w)).duration)
        except Exception:
            pass
    total_sec = sum(durs)
    avg_sec = (total_sec / len(durs)) if durs else 0.0
    min_sec = min(durs) if durs else 0.0
    max_sec = max(durs) if durs else 0.0
    return (
        f"训练集 (data/train/audio): {len(tr_wavs)} 条 WAV | 总时长: {total_sec/60:.2f} 分钟 ({total_sec:.1f}s) | 平均 {avg_sec:.2f}s（最短 {min_sec:.2f}s / 最长 {max_sec:.2f}s）\n"
        f"验证集 (data/val/audio): {len(val_wavs)} 条 WAV | 已提取特征缓存 (train/f0): {f0_cnt} 条"
    )


def run_audio_slicer(files, threshold_db, min_len_ms, min_interval_ms, max_sil_ms, min_keep_sec):
    if not files:
        return "请先上传需要切片的干声音频文件（支持多选或 .zip）。"
    from slicer import Slicer

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    extract_dir = RESULTS_TMP_DIR / f"_slice_zip_{ts}"
    input_wavs = []
    for fpath in files:
        src = Path(fpath if isinstance(fpath, str) else fpath.name)
        if src.suffix.lower() == ".zip":
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(src, "r") as zf:
                zf.extractall(extract_dir)
            for p in sorted(extract_dir.rglob("*")):
                if p.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
                    input_wavs.append(p)
        else:
            input_wavs.append(src)

    total_saved, total_skipped, logs, sr = 0, 0, [], 44100
    for src_path in input_wavs:
        tmp_wav = RESULTS_TMP_DIR / f"_slice_tmp_{ts}.wav"
        try:
            ensure_wav_44k(str(src_path), str(tmp_wav), mono=True)
            audio, _ = librosa.load(str(tmp_wav), sr=sr, mono=True)
            slicer = Slicer(sr=sr, threshold=float(threshold_db), min_length=int(min_len_ms), min_interval=int(min_interval_ms), hop_size=20, max_sil_kept=int(max_sil_ms))
            chunks = slicer.slice(audio)
            saved_here = 0
            for idx, (_, chunk_info) in enumerate(chunks.items()):
                wav_chunk = chunk_info["slice"]
                if len(wav_chunk) / sr < float(min_keep_sec):
                    total_skipped += 1
                    continue
                sf.write(str(TRAIN_AUDIO_DIR / f"{src_path.stem}_s{idx:04d}.wav"), wav_chunk, sr)
                saved_here += 1
                total_saved += 1
            logs.append(f"{src_path.name}: 切出 {saved_here} 段")
        except Exception as e:
            logs.append(f"{src_path.name}: 切片失败 - {e}")
        finally:
            tmp_wav.unlink(missing_ok=True)

    shutil.rmtree(extract_dir, ignore_errors=True)
    return f"切片完成并已放入 data/train/audio！共保存 {total_saved} 段（过滤过短片段 {total_skipped} 段）。\n" + "\n".join(logs) + "\n\n" + inspect_dataset()


def upload_train_files(files):
    if not files:
        return inspect_dataset()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_unzip = RESULTS_TMP_DIR / f"_train_zip_{ts}"
    for f in files:
        src = Path(f if isinstance(f, str) else f.name)
        if src.suffix.lower() == ".zip":
            tmp_unzip.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(src, "r") as zf:
                zf.extractall(tmp_unzip)
            for p in sorted(tmp_unzip.rglob("*")):
                if p.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]:
                    ensure_wav_44k(str(p), str(TRAIN_AUDIO_DIR / (p.stem + ".wav")), mono=True)
        else:
            ensure_wav_44k(str(src), str(TRAIN_AUDIO_DIR / (src.stem + ".wav")), mono=True)
    shutil.rmtree(tmp_unzip, ignore_errors=True)
    return "已全部转为 44.1kHz 单声道 WAV 并放入训练集！\n\n" + inspect_dataset()


def run_batch_clean_train(do_highpass, do_loudnorm, do_trim_sil):
    wavs = sorted(TRAIN_AUDIO_DIR.rglob("*.wav"))
    if not wavs:
        return "data/train/audio 中没有找到 WAV 文件。"
    filters = []
    if do_highpass:
        filters.append("highpass=f=80")
    if do_trim_sil:
        filters.append("silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,areverse")
    if do_loudnorm:
        filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    af = ",".join(filters)

    ok_cnt, removed_short = 0, 0
    for w in wavs:
        tmp = w.with_suffix(".tmp.wav")
        try:
            ensure_wav_44k(str(w), str(tmp), mono=True, sr=44100, extra_af=af)
            if sf.info(str(tmp)).duration < 2.05:
                tmp.unlink(missing_ok=True)
                w.unlink(missing_ok=True)
                removed_short += 1
            else:
                shutil.move(str(tmp), str(w))
                ok_cnt += 1
        except Exception:
            tmp.unlink(missing_ok=True)
    return f"整库清洗完成！保留 {ok_cnt} 条有效音频，自动剔除短于 2.05s 的音频 {removed_short} 条。\n\n{inspect_dataset()}"


def run_draw_val(sample_pct):
    res = subprocess.run([sys.executable, "draw.py", "-r", str(float(sample_pct))], cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return f"划分验证集 (比例 {sample_pct}%) 返回码: {res.returncode}\n{res.stdout}\n{res.stderr}\n\n{inspect_dataset()}"


def restore_val_to_train():
    moved = 0
    for w in list(VAL_AUDIO_DIR.glob("*.wav")):
        shutil.move(str(w), str(TRAIN_AUDIO_DIR / w.name))
        moved += 1
    return f"已将 {moved} 条验证集音频退回训练集。\n\n{inspect_dataset()}"


def clear_dataset_and_cache(clear_audio, clear_features):
    removed = []
    if clear_features:
        for base in [TRAIN_DIR, VAL_DIR]:
            for sub in ["units", "f0", "volume", "aug_vol", "mel", "aug_mel", "skip"]:
                d = base / sub
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
                    removed.append(str(d.relative_to(BASE_DIR)))
    if clear_audio:
        for d in [TRAIN_AUDIO_DIR, VAL_AUDIO_DIR]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
                removed.append(str(d.relative_to(BASE_DIR)))
    return f"已清理：{', '.join(removed) if removed else '无'}\n\n{inspect_dataset()}"


def create_role_exp(role_name, init_mode, custom_step_ckpt):
    name = role_name.strip().replace(" ", "_")
    if not name:
        return gr.update(), "请输入角色目录名（例如 my_character）。"
    if not name.startswith("exp/"):
        name = f"exp/{name}"
    target_dir = BASE_DIR / name
    target_dir.mkdir(parents=True, exist_ok=True)

    if init_mode.startswith("从零训练") or init_mode.startswith("复制官方底模"):
        if not BASE_PRETRAIN_CKPT.exists():
            return gr.update(choices=list_exp_dirs()), f"未找到官方底模 {BASE_PRETRAIN_CKPT}"
        dst = target_dir / "model_0.pt"
        if not dst.exists():
            shutil.copy2(BASE_PRETRAIN_CKPT, dst)
        msg = f"已创建 {name} 并加载官方底模 model_0.pt。"
    elif init_mode.startswith("基于已有模型"):
        if not custom_step_ckpt:
            return gr.update(choices=list_exp_dirs()), "请先选择已有模型目录与具体步数 (.pt)。"
        src = BASE_DIR / custom_step_ckpt
        if not src.exists():
            return gr.update(choices=list_exp_dirs()), f"选择的权重不存在：{src}"
        shutil.copy2(src, target_dir / "model_0.pt")
        msg = f"已创建 {name}，并将 {custom_step_ckpt} 复制为起始权重 model_0.pt 用于微调。"
    else:
        msg = f"已创建空角色目录 {name}。"
    return gr.update(choices=list_exp_dirs(), value=name), msg


def recommend_train_params():
    vram_mb = 24000
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, text=True).stdout.strip().splitlines()[0]
        vram_mb = int(out.strip())
    except Exception:
        pass
    durs = []
    for w in TRAIN_AUDIO_DIR.rglob("*.wav"):
        try:
            durs.append(sf.info(str(w)).duration)
        except Exception:
            pass
    mins = sum(durs) / 60.0
    bs = 48 if vram_mb >= 20000 else (24 if vram_mb >= 11000 else 12)
    lr = 0.0001 if mins < 8.0 else 0.0002
    cdev = "cuda" if vram_mb >= 20000 and mins <= 25.0 else "cpu"
    return bs, lr, "fp16", cdev, 4000, f"已按 {vram_mb}MB 显存与 {mins:.1f} 分钟素材填入推荐参数。"


def save_train_config(
    expdir, vocoder_choice, f0_ext, duration, use_pitch_aug,
    batch_size, lr, decay_step, gamma, epochs, lambda_ddsp,
    amp_dtype, cache_all, cache_dev, num_workers, interval_val, interval_save,
):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["env"]["expdir"] = str(expdir)
    cfg["data"]["f0_extractor"] = str(f0_ext)
    cfg["data"]["duration"] = float(duration)
    cfg["model"]["use_pitch_aug"] = bool(use_pitch_aug)
    if "pc_nsf_hifigan" in str(vocoder_choice):
        cfg["vocoder"]["type"] = "pc_nsf_hifigan"
        cfg["vocoder"]["ckpt"] = "pretrain/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt"
    else:
        cfg["vocoder"]["type"] = "nsf-hifigan"
        cfg["vocoder"]["ckpt"] = "pretrain/nsf_hifigan/model"
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["train"]["lr"] = float(lr)
    cfg["train"]["decay_step"] = int(decay_step)
    cfg["train"]["gamma"] = float(gamma)
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["lambda_ddsp"] = float(lambda_ddsp)
    cfg["train"]["amp_dtype"] = str(amp_dtype)
    cfg["train"]["cache_all_data"] = bool(cache_all)
    cfg["train"]["cache_device"] = str(cache_dev)
    cfg["train"]["num_workers"] = int(num_workers)
    cfg["train"]["interval_val"] = int(interval_val)
    cfg["train"]["interval_force_save"] = int(interval_save)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return f"已保存训练配置到 {CONFIG_PATH}（expdir={expdir}）。"


def run_preprocess():
    res = subprocess.run([sys.executable, "preprocess.py", "-c", "configs/reflow.yaml"], cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return f"特征提取完成（返回码 {res.returncode}）：\n{res.stdout[-1200:]}\n{res.stderr[-600:]}"


def format_status_banner():
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
    except Exception:
        smi = "无法获取 GPU 信息"

    running = TRAIN_PROC is not None and TRAIN_PROC.poll() is None
    status = "🟢 训练运行中" if running else "⚪ 未运行"

    step_str, loss_str, val_str = "-", "-", "-"
    if TRAIN_LOG_PATH.exists():
        lines = TRAIN_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]
        for ln in reversed(lines):
            if step_str == "-":
                m_step = re.search(r"step[:\s]+(\d+)", ln, re.I)
                m_loss = re.search(r"loss[:\s=]+([0-9.]+)", ln, re.I)
                if m_step:
                    step_str = m_step.group(1)
                if m_loss:
                    loss_str = m_loss.group(1)
            if val_str == "-" and "valid" in ln.lower():
                m_v = re.search(r"loss[:\s=]+([0-9.]+)", ln, re.I)
                if m_v:
                    val_str = m_v.group(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cur_exp = yaml.safe_load(f).get("env", {}).get("expdir", "exp/reflow-test")
    latest_pts = list_steps_in_dir(cur_exp)
    latest_ckpt = latest_pts[0][0] if latest_pts else "暂无（仅底模或空）"

    return (
        f"<div class='status-card'>"
        f"<b>训练状态：</b>{status} &nbsp;|&nbsp; <b>当前实验目录：</b><code>{cur_exp}</code>（最新权重：<code>{latest_ckpt}</code>）<br>"
        f"<b>实时指标：</b>Step = <code>{step_str}</code> &nbsp;|&nbsp; Train Loss = <code>{loss_str}</code> &nbsp;|&nbsp; Val Loss = <code>{val_str}</code><br>"
        f"<b>GPU 硬件：</b><code>{smi}</code>"
        f"</div>"
    )


def get_gpu_and_train_status():
    logs = TRAIN_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[-60:] if TRAIN_LOG_PATH.exists() else ["暂无训练日志。"]
    return format_status_banner(), "\n".join(logs)


def start_training(resume=False):
    global TRAIN_PROC
    if TRAIN_PROC is not None and TRAIN_PROC.poll() is None:
        return format_status_banner(), "训练已在运行中，请勿重复启动。"
    if not list(TRAIN_AUDIO_DIR.glob("*.wav")):
        return format_status_banner(), "训练集 data/train/audio 为空，请先完成第 1 步。"
    if not list(VAL_AUDIO_DIR.glob("*.wav")):
        return format_status_banner(), "验证集 data/val/audio 为空，请先点击「3. 划分验证集」。"
    if not (TRAIN_DIR / "f0").exists():
        return format_status_banner(), "未检测到特征缓存，请先点击「1. 提取特征」。"

    log_f = open(TRAIN_LOG_PATH, "w", encoding="utf-8")
    TRAIN_PROC = subprocess.Popen([sys.executable, "train_reflow.py", "-c", "configs/reflow.yaml"], cwd=str(BASE_DIR), stdout=log_f, stderr=subprocess.STDOUT)
    time.sleep(1.5)
    return format_status_banner(), f"训练已启动 (PID={TRAIN_PROC.pid})，可点击刷新查看实时指标或前往 6006 端口 TensorBoard 试听。"


def stop_training():
    global TRAIN_PROC
    if TRAIN_PROC is not None and TRAIN_PROC.poll() is None:
        TRAIN_PROC.terminate()
        try:
            TRAIN_PROC.wait(timeout=8)
        except Exception:
            TRAIN_PROC.kill()
        return format_status_banner(), "训练进程已停止。"
    return format_status_banner(), "当前没有运行中的训练进程。"


def compare_checkpoints(audio_file, ckpt_list, key_shift, pitch_extractor):
    if not audio_file:
        return None, None, None, "请上传一段测试干声。"
    if not ckpt_list:
        return None, None, None, "请勾选至少 1 个 Checkpoint（最多对比前 3 个）。"
    selected = ckpt_list[:3]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_in = RESULTS_TMP_DIR / f"_cmp_in_{ts}.wav"
    ensure_wav_44k(audio_file, str(tmp_in), mono=False)

    outs, logs = [None, None, None], []
    for i, ckpt_rel in enumerate(selected):
        out_p, msg, _ = run_inference(str(tmp_in), ckpt_rel, key_shift, 0, 0, pitch_extractor, "auto", "50", 0.7, -60, 50, 1100)
        outs[i] = out_p
        logs.append(f"轨道 {i+1} ({Path(ckpt_rel).name}): {'成功' if out_p else msg}")
    tmp_in.unlink(missing_ok=True)
    return outs[0], outs[1], outs[2], "\n".join(logs)


def archive_or_export_model(ckpt_rel, role_alias):
    if not ckpt_rel:
        return None, "请先选择要归档/导出的 Checkpoint。"
    ckpt_p = BASE_DIR / ckpt_rel
    cfg_p = ckpt_p.parent / "config.yaml"
    if not ckpt_p.exists():
        return None, f"文件不存在：{ckpt_p}"

    alias = (role_alias or ckpt_p.parent.name).strip().replace(" ", "_")
    dst_dir = MODELS_DIR / alias
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_p, dst_dir / ckpt_p.name)
    if cfg_p.exists():
        shutil.copy2(cfg_p, dst_dir / "config.yaml")

    zip_path = RESULTS_DIR / f"model_{alias}_{ckpt_p.stem}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(dst_dir / ckpt_p.name, f"{alias}/{ckpt_p.name}")
        if (dst_dir / "config.yaml").exists():
            zf.write(dst_dir / "config.yaml", f"{alias}/config.yaml")
    return str(zip_path), f"已精简归档到 models/{alias}/，并生成打包下载文件。"


def import_external_model_zip(zip_file, custom_name):
    if not zip_file:
        return "请先上传包含 .pt 和 config.yaml 的 .zip 模型包。", gr.update(), gr.update()
    zp = Path(zip_file if isinstance(zip_file, str) else zip_file.name)
    folder_name = (custom_name or zp.stem).strip().replace(" ", "_")
    dst_dir = MODELS_DIR / folder_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    imported = []
    with zipfile.ZipFile(zp, "r") as zf:
        for info in zf.infolist():
            fn = Path(info.filename).name
            if fn.endswith((".pt", ".yaml", ".yml")):
                with zf.open(info) as s, open(dst_dir / fn, "wb") as d:
                    shutil.copyfileobj(s, d)
                imported.append(fn)
    d_upd, s_upd = refresh_all_models()
    return f"已导入到 models/{folder_name}/（{', '.join(imported) if imported else '无有效文件'}）", d_upd, s_upd


def reset_to_clean_image():
    if TRAIN_PROC is not None and TRAIN_PROC.poll() is None:
        return "当前有训练进程在运行，请先停止训练再重置！"
    cleaned = []
    for target in [RESULTS_DIR, CACHE_DIR]:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)
            cleaned.append(str(target.relative_to(BASE_DIR)))
    for base in [TRAIN_DIR, VAL_DIR]:
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
        (base / "audio").mkdir(parents=True, exist_ok=True)
        cleaned.append(str(base.relative_to(BASE_DIR)))
    TRAIN_LOG_PATH.unlink(missing_ok=True)
    return f"已重置为纯净镜像状态！清理目录：{', '.join(cleaned)}（保留了所有底模与 models/ 模型库）。"

# ==================== 容器目录扫描、一键分离与文件/模型管理辅助 ====================
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
MANAGED_DIRS = {
    "待处理音频目录 (raw/)": RAW_DIR,
    "01 · 翻唱结果 (results/01_翻唱结果/)": RESULTS_COVER_DIR,
    "02 · 一条龙翻唱 (results/02_一条龙翻唱/)": RESULTS_ONECLICK_DIR,
    "03 · 模型分离结果 (results/03_模型分离结果/)": RESULTS_MSST_DIR,
    "04 · 混音与后期处理 (results/04_混音与后期处理/)": RESULTS_MIX_DIR,
    "全部输出总目录 (results/)": RESULTS_DIR,
    "训练集干声 (data/train/audio/)": TRAIN_AUDIO_DIR,
    "验证集干声 (data/val/audio/)": VAL_AUDIO_DIR,
}
ONE_CLICK_MSST_KEY = "__ONE_CLICK_3STEP__"


def list_container_audio_choices():
    items = []
    for root_dir, tag in [(RAW_DIR, "raw"), (RESULTS_DIR, "results")]:
        if not root_dir.exists():
            continue
        for f in root_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS and not f.name.startswith("_"):
                rel_parts = f.relative_to(root_dir).parts
                if any(part.startswith("_") for part in rel_parts):
                    continue
                try:
                    mtime = f.stat().st_mtime
                except OSError:
                    mtime = 0
                rel = str(f.relative_to(root_dir))
                items.append((mtime, f"[{tag}] {rel}", str(f)))
    items.sort(key=lambda x: x[0], reverse=True)
    return [(label, path) for _, label, path in items[:150]]


def save_audio_to_raw(audio_path):
    if not audio_path:
        c = list_container_audio_choices()
        return gr.update(choices=c), gr.update(choices=c), "请先在上方上传或载入一段音频。"
    src_p = Path(audio_path)
    if not src_p.exists():
        c = list_container_audio_choices()
        return gr.update(choices=c), gr.update(choices=c), "音频文件不存在，无法保存。"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dst_p = RAW_DIR / src_p.name
    if dst_p.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_p = RAW_DIR / f"{src_p.stem}_{ts}{src_p.suffix}"
    shutil.copy2(src_p, dst_p)
    c = list_container_audio_choices()
    return gr.update(choices=c, value=str(dst_p)), gr.update(choices=c), f"已保存到容器目录：raw/{dst_p.name}"


def upload_to_managed_dir(dir_key: str, files):
    target = MANAGED_DIRS.get(dir_key, RAW_DIR)
    target.mkdir(parents=True, exist_ok=True)
    if not files:
        return gr.update(choices=list_managed_files(dir_key), value=[]), "请先选择要上传的音频或 .zip 压缩包。"
    saved = 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_unzip = RESULTS_TMP_DIR / f"_upload_zip_{ts}"
    for f in files:
        sp = Path(f if isinstance(f, str) else f.name)
        if sp.suffix.lower() == ".zip":
            tmp_unzip.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(sp, "r") as zf:
                zf.extractall(tmp_unzip)
            for item in sorted(tmp_unzip.rglob("*")):
                if item.is_file() and item.suffix.lower() in AUDIO_EXTS:
                    dst = target / item.name
                    if dst.exists():
                        dst = target / f"{item.stem}_{ts}{item.suffix}"
                    shutil.copy2(item, dst)
                    saved += 1
        elif sp.is_file():
            dst = target / sp.name
            if dst.exists():
                dst = target / f"{sp.stem}_{ts}{sp.suffix}"
            shutil.copy2(sp, dst)
            saved += 1
    shutil.rmtree(tmp_unzip, ignore_errors=True)
    return gr.update(choices=list_managed_files(dir_key), value=[]), f"已成功导入 {saved} 个文件到「{dir_key}」。"


def _run_msst_3step_single(audio_file: str, out_dir: Path, use_tta: bool):
    presets = get_msst_models()
    k1 = next((k for k in presets if k.startswith("1.")), None)
    k2 = next((k for k in presets if k.startswith("2.")), None)
    k3 = next((k for k in presets if k.startswith("3.")), None)
    if not (k1 and k2 and k3):
        raise RuntimeError(f"未检测到完整的 MSST 三步模型（人声分离 / 主唱和声 / 去混响），请检查 {MSST_DIR / 'pretrain'}。")

    out_dir.mkdir(parents=True, exist_ok=True)
    s1_dir = out_dir / "_step1_vocals"
    s2_dir = out_dir / "_step2_karaoke"
    s3_dir = out_dir / "_step3_dereverb"

    stems1, _ = run_msst_core(audio_file, *presets[k1], s1_dir, use_tta=use_tta, extract_inst=True)
    vocals = stems1[0]
    inst = stems1[1] if len(stems1) > 1 else None

    stems2, _ = run_msst_core(vocals, *presets[k2], s2_dir, use_tta=use_tta, extract_inst=True)
    lead_vocal = stems2[0]
    harmony = stems2[1] if len(stems2) > 1 else None

    stems3, _ = run_msst_core(lead_vocal, *presets[k3], s3_dir, use_tta=use_tta, extract_inst=True)
    dry_lead = stems3[0]

    stem_name = Path(audio_file).stem
    final_dry = out_dir / f"{stem_name}_纯净主唱干声.wav"
    shutil.copy2(dry_lead, final_dry)
    saved_files = [str(final_dry)]

    final_inst = None
    if inst and Path(inst).exists():
        final_inst = out_dir / f"{stem_name}_纯伴奏.wav"
        shutil.copy2(inst, final_inst)
        saved_files.append(str(final_inst))
    if harmony and Path(harmony).exists():
        final_harm = out_dir / f"{stem_name}_原曲和声.wav"
        shutil.copy2(harmony, final_harm)
        saved_files.append(str(final_harm))

    for d in [s1_dir, s2_dir, s3_dir]:
        shutil.rmtree(d, ignore_errors=True)

    return str(final_dry), (str(final_inst) if final_inst else None), saved_files


def list_msst_history():
    items = []
    if RESULTS_MSST_DIR.exists():
        for f in RESULTS_MSST_DIR.rglob("*"):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS and not f.name.startswith("_"):
                rel_parts = f.relative_to(RESULTS_MSST_DIR).parts
                if any(part.startswith("_") for part in rel_parts):
                    continue
                try:
                    mtime = f.stat().st_mtime
                except OSError:
                    mtime = 0
                rel = str(f.relative_to(RESULTS_MSST_DIR))
                items.append((mtime, rel, str(f)))
    if RESULTS_DIR.exists():
        for sub in RESULTS_DIR.iterdir():
            if sub.is_dir() and sub.name.startswith("msst_"):
                for f in sub.rglob("*"):
                    if f.is_file() and f.suffix.lower() in AUDIO_EXTS and not f.name.startswith("_"):
                        try:
                            mtime = f.stat().st_mtime
                        except OSError:
                            mtime = 0
                        rel = str(f.relative_to(RESULTS_DIR))
                        items.append((mtime, rel, str(f)))
    items.sort(key=lambda x: x[0], reverse=True)
    return [(label, path) for _, label, path in items[:120]]


def run_msst_enhanced(audio_file, model_label, use_tta, extract_inst):
    if not audio_file:
        yield None, None, "请先上传或载入待分离的音频文件。", gr.update(), gr.update(), gr.update()
        return
    if model_label == ONE_CLICK_MSST_KEY:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        song_tag = _safe_stem(audio_file)
        out_dir = RESULTS_MSST_DIR / f"{song_tag}_三步全分离_{ts}"
        try:
            t0 = time.time()
            yield None, None, "⏳ 正在执行一键三步分离（人声分离 → 去和声 → 去混响）...", gr.update(), gr.update(), gr.update()
            t, res, err = _run_step_with_heartbeat(_run_msst_3step_single, audio_file, out_dir, use_tta=use_tta)
            while t.is_alive():
                t.join(timeout=3.0)
                if t.is_alive():
                    yield None, None, f"⏳ 正在执行一键三步分离...（已耗时 {int(time.time()-t0)}s）", gr.update(), gr.update(), gr.update()
            if "err" in err:
                raise err["err"]
            dry_p, inst_p, all_files = res["val"]
            elapsed = time.time() - t0
            c = list_container_audio_choices()
            mh = list_msst_history()
            msg = f"一键三步全分离完成（耗时 {elapsed:.1f}s）！\n已依次执行：人声分离 → 去和声 → 去混响\n输出文件：\n" + "\n".join(all_files)
            yield dry_p, inst_p, msg, gr.update(choices=c), gr.update(choices=c), gr.update(choices=mh)
            return
        except Exception as e:
            yield None, None, f"一键三步分离失败：{e}", gr.update(), gr.update(), gr.update()
            return
    t0 = time.time()
    yield None, None, f"⏳ 正在运行 MSST 模型（{model_label}）...", gr.update(), gr.update(), gr.update()
    t, res, err = _run_step_with_heartbeat(run_msst_ui, audio_file, model_label, use_tta, extract_inst)
    while t.is_alive():
        t.join(timeout=3.0)
        if t.is_alive():
            yield None, None, f"⏳ 正在运行 MSST 模型...（已耗时 {int(time.time()-t0)}s）", gr.update(), gr.update(), gr.update()
    if "err" in err:
        yield None, None, f"MSST 分离失败：{err['err']}", gr.update(), gr.update(), gr.update()
        return
    s1, s2, msg = res["val"]
    c = list_container_audio_choices()
    mh = list_msst_history()
    yield s1, s2, msg, gr.update(choices=c), gr.update(choices=c), gr.update(choices=mh)


def run_msst_batch_zip(files, model_label, use_tta, extract_inst):
    if not files:
        return None, "请上传一个或多个音频文件或 .zip 压缩包。", gr.update(), gr.update(), gr.update()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    extract_dir = RESULTS_TMP_DIR / f"_msst_unzip_{ts}"
    step_tag = "三步全分离" if model_label == ONE_CLICK_MSST_KEY else _msst_step_and_labels(model_label, [])[0]
    batch_out = RESULTS_MSST_DIR / f"批量分离_{step_tag}_{ts}"
    batch_out.mkdir(parents=True, exist_ok=True)

    audio_list = []
    for f in files:
        sp = Path(f if isinstance(f, str) else f.name)
        if sp.suffix.lower() == ".zip":
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(sp, "r") as zf:
                zf.extractall(extract_dir)
            for item in sorted(extract_dir.rglob("*")):
                if item.is_file() and item.suffix.lower() in AUDIO_EXTS:
                    audio_list.append(item)
        elif sp.is_file() and sp.suffix.lower() in AUDIO_EXTS:
            audio_list.append(sp)

    if not audio_list:
        shutil.rmtree(extract_dir, ignore_errors=True)
        return None, "未在上传文件/压缩包中找到支持的音频（WAV/FLAC/MP3/M4A/OGG）。", gr.update(), gr.update(), gr.update()

    presets = get_msst_models()
    logs, produced = [], []
    t0 = time.time()
    for idx, ap in enumerate(audio_list, 1):
        sub_out = batch_out / f"{idx:02d}_{ap.stem}"
        try:
            if model_label == ONE_CLICK_MSST_KEY:
                _, _, outs = _run_msst_3step_single(str(ap), sub_out, use_tta=use_tta)
            else:
                mtype, cfg_p, ckpt_p = presets[model_label]
                work_sub = sub_out / "_raw"
                raw_outs, _ = run_msst_core(str(ap), mtype, cfg_p, ckpt_p, work_sub, use_tta=use_tta, extract_inst=extract_inst)
                _, n1, n2 = _msst_step_and_labels(model_label, raw_outs)
                outs = []
                if len(raw_outs) > 0:
                    p1 = sub_out / f"{ap.stem}_{n1}.wav"
                    shutil.copy2(raw_outs[0], p1)
                    outs.append(str(p1))
                if len(raw_outs) > 1 and extract_inst:
                    p2 = sub_out / f"{ap.stem}_{n2}.wav"
                    shutil.copy2(raw_outs[1], p2)
                    outs.append(str(p2))
                shutil.rmtree(work_sub, ignore_errors=True)
            produced.extend(outs)
            logs.append(f"[{idx}/{len(audio_list)}] {ap.name}：成功（{len(outs)} 轨）")
        except Exception as e:
            logs.append(f"[{idx}/{len(audio_list)}] {ap.name}：失败 - {e}")

    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path = RESULTS_MSST_DIR / f"批量分离_{step_tag}_{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in produced:
            p_obj = Path(fp)
            if p_obj.exists():
                zf.write(p_obj, p_obj.relative_to(batch_out))

    elapsed = time.time() - t0
    c = list_container_audio_choices()
    mh = list_msst_history()
    summary = f"批量分离完成！共处理 {len(audio_list)} 首，输出 {len(produced)} 轨（耗时 {elapsed:.1f}s）。\n结果已保存至 results/03_模型分离结果/{batch_out.name}/ 并打包下载。\n" + "\n".join(logs)
    return str(zip_path), summary, gr.update(choices=c), gr.update(choices=c), gr.update(choices=mh)


def list_managed_files(dir_key: str):
    target = MANAGED_DIRS.get(dir_key, RAW_DIR)
    if not target.exists():
        return []
    files = []
    for f in target.rglob("*"):
        if f.is_file() and not f.name.startswith(".") and not f.name.startswith("_"):
            rel_parts = f.relative_to(target).parts
            if any(part.startswith("_") for part in rel_parts):
                continue
            try:
                sz_mb = f.stat().st_size / (1024 * 1024)
                mtime = f.stat().st_mtime
            except OSError:
                sz_mb, mtime = 0.0, 0
            rel = str(f.relative_to(target))
            files.append((mtime, f"{rel}  ({sz_mb:.2f} MB)", str(f)))
    files.sort(key=lambda x: x[0], reverse=True)
    return [(label, path) for _, label, path in files[:200]]


def delete_selected_files(dir_key: str, selected_paths):
    if not selected_paths:
        return gr.update(choices=list_managed_files(dir_key), value=[]), "未勾选任何文件。"
    removed, failed = 0, 0
    base_resolved = BASE_DIR.resolve()
    for sp in selected_paths:
        try:
            fp = Path(sp).resolve()
            if base_resolved in fp.parents and fp.is_file():
                fp.unlink()
                removed += 1
        except Exception:
            failed += 1
    new_choices = list_managed_files(dir_key)
    return gr.update(choices=new_choices, value=[]), f"已永久删除 {removed} 个文件" + (f"（失败 {failed} 个）" if failed else "。")


def list_deletable_models():
    items = []
    for root_dir, tag in [(MODELS_DIR, "模型库 models"), (EXP_DIR, "训练目录 exp")]:
        if not root_dir.exists():
            continue
        for sub in sorted(root_dir.iterdir()):
            if not sub.is_dir():
                continue
            rel = str(sub.relative_to(BASE_DIR))
            if rel == "exp/reflow-test":
                continue
            pts = list(sub.glob("*.pt"))
            items.append((f"[{tag}] {sub.name}  ({len(pts)} 个权重文件)", rel))
    return items


def download_selected_files(dir_key: str, selected_paths):
    if not selected_paths:
        return None, "请先勾选要下载的文件。"
    base_resolved = BASE_DIR.resolve()
    valid = []
    for sp in selected_paths:
        try:
            fp = Path(sp).resolve()
            if base_resolved in fp.parents and fp.is_file():
                valid.append(fp)
        except Exception:
            pass
    if not valid:
        return None, "未找到可下载的有效文件。"
    if len(valid) == 1:
        return str(valid[0]), f"已准备好下载单文件：{valid[0].name}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = RESULTS_TMP_DIR / f"_dl_files_{ts}.zip"
    target_root = MANAGED_DIRS.get(dir_key, RAW_DIR).resolve()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in valid:
            arc = fp.relative_to(target_root) if target_root in fp.parents else fp.name
            zf.write(fp, str(arc))
    return str(zip_path), f"已打包 {len(valid)} 个文件为 {zip_path.name}，点击下方文件即可下载。"


def download_selected_models(selected_rels):
    if not selected_rels:
        return None, "请先勾选要下载的模型目录。"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = RESULTS_TMP_DIR / f"_dl_models_{ts}.zip"
    file_cnt = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in selected_rels:
            target = (BASE_DIR / rel).resolve()
            if not (target.is_dir() and (MODELS_DIR.resolve() in target.parents or EXP_DIR.resolve() in target.parents)):
                continue
            for fp in sorted(target.rglob("*")):
                if fp.is_file() and (fp.suffix.lower() in {".pt", ".yaml", ".yml", ".json", ".txt"}):
                    zf.write(fp, f"{target.name}/{fp.relative_to(target)}")
                    file_cnt += 1
    if file_cnt == 0:
        zip_path.unlink(missing_ok=True)
        return None, "所选模型目录中未找到 .pt / config.yaml 文件。"
    return str(zip_path), f"已打包 {len(selected_rels)} 个模型目录（共 {file_cnt} 个权重/配置文件），点击下方文件即可下载。"


def delete_selected_models(selected_rels):
    if TRAIN_PROC is not None and TRAIN_PROC.poll() is None:
        return gr.update(), "当前正在训练中，请先停止训练再删除模型目录！"
    if not selected_rels:
        return gr.update(choices=list_deletable_models(), value=[]), "未勾选任何模型目录。"
    removed = []
    for rel in selected_rels:
        if rel == "exp/reflow-test":
            continue
        target = (BASE_DIR / rel).resolve()
        if target.is_dir() and (MODELS_DIR.resolve() in target.parents or EXP_DIR.resolve() in target.parents):
            shutil.rmtree(target, ignore_errors=True)
            removed.append(rel)
    return gr.update(choices=list_deletable_models(), value=[]), f"已删除模型目录：{', '.join(removed) if removed else '无'}"


def run_batch_with_picker(selected_container_files, uploaded_files, folder_path, model_step_path, key_shift, formant_shift, pitch_extractor, infer_method, infer_step):
    merged = list(uploaded_files) if uploaded_files else []
    if selected_container_files:
        merged.extend(selected_container_files)
    return run_batch_inference(merged if merged else None, folder_path, model_step_path, key_shift, formant_shift, pitch_extractor, infer_method, infer_step)


def archive_and_refresh_models(ckpt_rel, role_alias):
    zip_out, msg = archive_or_export_model(ckpt_rel, role_alias)
    dirs_upd, steps_upd = refresh_all_models()
    return zip_out, msg + "（已自动同步到「语音翻唱」模型列表，切过去即可直接使用）", dirs_upd, steps_upd


# ==================== 构建 Gradio 界面（暖白纸感三栏工作室风格） ====================
init_dirs, init_dir, init_steps, init_step = get_default_model_and_step()
init_prefs = load_ui_prefs()
init_container_audios = list_container_audio_choices()
msst_presets = get_msst_models()
msst_keys = list(msst_presets.keys())
msst_choices = [
    ("一键三步全分离  ·  3 轨（自动完成人声分离 → 去和声 → 去混响，直出纯净干声）", ONE_CLICK_MSST_KEY),
]
for k in msst_keys:
    if k.startswith("1."):
        msst_choices.append(("人声与伴奏  ·  2 轨（BS-Roformer 12.97 推荐）", k))
    elif k.startswith("1b."):
        msst_choices.append(("人声与伴奏  ·  2 轨（Mel-Band-Roformer KJ 备选）", k))
    elif k.startswith("2."):
        msst_choices.append(("主唱与和声  ·  2 轨（Karaoke Becruily 分离主唱与和声）", k))
    elif k.startswith("3."):
        msst_choices.append(("去除混响  ·  2 轨（Dereverb Anvuew 去除人声混响）", k))
    else:
        msst_choices.append((k, k))

STEP_CHOICES = [
    ("30 步（快速试听）", "30"),
    ("50 步（默认推荐）", "50"),
    ("80 步（更细腻）", "80"),
    ("100 步（极细腻）", "100"),
]

with gr.Blocks(
    title="DDSP-SVC 6.2 语音工作室",
    theme=STUDIO_THEME,
    css=CUSTOM_CSS,
    js=FORCE_LIGHT_JS,
) as demo:
    fx_style_box = gr.HTML(build_style_html(init_prefs["enable_fx"]), elem_id="fx-style-holder")
    pending_ds_action = gr.State("")

    with gr.Row(elem_id="studio-shell"):
        # -------------------- 左侧边栏 --------------------
        with gr.Column(scale=0, min_width=220, elem_id="studio-sidebar"):
            gr.HTML(
                """
                <div class="brand-header">
                    <div class="brand-mark"></div>
                    <div>
                        <div class="brand-title">DDSP-SVC</div>
                        <div class="brand-sub">6.2 · 语音翻唱工作室</div>
                    </div>
                </div>
                """
            )
            nav_infer = gr.Button("♪  语音翻唱", elem_classes=["nav-btn", "nav-active"])
            nav_msst = gr.Button("▤  音轨分离", elem_classes=["nav-btn"])
            nav_train = gr.Button("◈  模型训练", elem_classes=["nav-btn"])
            nav_sys = gr.Button("⚙  系统维护", elem_classes=["nav-btn"])

            gr.HTML('<div class="sidebar-spacer"></div>')
            gr.HTML('<div class="author-credit">作者：<span>bilibili-无法访问em</span></div>')
            fx_cb = gr.Checkbox(value=init_prefs["enable_fx"], label="动态效果（柔光与声波）", elem_id="fx-cb-item")
            sidebar_pill = gr.HTML(build_sidebar_status_html(init_prefs["enable_fx"]))

        # -------------------- 右侧内容区 --------------------
        with gr.Column(scale=1, min_width=0, elem_id="studio-content-wrap"):

            # ===== 视图 1：语音翻唱 =====
            with gr.Row(visible=True, elem_classes=["workspace-row"]) as view_infer:
                with gr.Column(scale=7, min_width=0, elem_classes=["studio-main-card"]):
                    gr.HTML(
                        """
                        <div class="page-header">
                            <span class="page-title">语音翻唱</span>
                            <span class="page-meta">推荐真无损 WAV 干声 · 支持直接选用或保存到容器目录</span>
                        </div>
                        """
                    )
                    in_audio = gr.Audio(label="点击上传或拖动干声音频至此", type="filepath")

                    with gr.Row(elem_classes=["aligned-row"]):
                        infer_pick_dd = gr.Dropdown(
                            label="从容器目录选用（raw/ 与分离输出）",
                            choices=init_container_audios,
                            value=None,
                            scale=5,
                        )
                        infer_pick_ref = gr.Button("刷新", scale=1)
                        infer_pick_load = gr.Button("载入", scale=1)
                        infer_save_raw = gr.Button("存到 raw/", scale=2)

                    with gr.Row(elem_classes=["aligned-row"]):
                        model_dir_dd = gr.Dropdown(label="角色模型", choices=init_dirs, value=init_dir, scale=4)
                        model_step_dd = gr.Dropdown(label="权重步数", choices=init_steps, value=init_step, scale=4)
                        refresh_btn = gr.Button("刷新模型", scale=2)

                    key_slider = gr.Slider(
                        -24, 24, value=0, step=1,
                        label="变调（半音）",
                        info="男转女常用 +12，女转男常用 -12，原调保持 0。",
                    )

                    infer_btn = gr.Button("开始翻唱", variant="primary")

                    with gr.Accordion("高级调参（音高提取器 / 共振峰 / 声区偏移 / 采样步数）", open=False):
                        with gr.Row():
                            formant_slider = gr.Slider(-6, 6, value=0, step=0.5, label="共振峰偏移", info="正值偏细，负值偏厚")
                            reg_slider = gr.Slider(-24, 24, value=0, step=0.5, label="声区偏移", info="仅 pc_nsf_hifigan 生效")
                        with gr.Row():
                            pe_dd = gr.Dropdown(PE_CHOICES, value="rmvpe", label="音高提取器")
                            method_dd = gr.Dropdown(METHOD_CHOICES, value="auto", label="采样方法")
                        with gr.Row():
                            step_tb = gr.Dropdown(choices=STEP_CHOICES, value="50", allow_custom_value=True, label="采样步数")
                            t_start_num = gr.Number(value=0.7, label="t_start (0~1)", info="越大越保留原唱咬字细节，默认 0.7")
                        with gr.Row():
                            th_num = gr.Number(value=-60, label="静音阈值 (dB)")
                            fmin_num = gr.Number(value=50, label="最低 F0 (Hz)")
                            fmax_num = gr.Number(value=1100, label="最高 F0 (Hz)")

                    with gr.Accordion("批量干声推理（支持容器目录多选或本地批量上传）", open=False):
                        b_pick_cg = gr.CheckboxGroup(
                            choices=init_container_audios,
                            label="勾选容器内音频（raw/ 与 results/）",
                            elem_classes=["file-checklist"],
                        )
                        with gr.Row():
                            b_pick_ref = gr.Button("刷新列表")
                            b_pick_all = gr.Button("全选")
                            b_pick_none = gr.Button("取消全选")
                        b_files = gr.File(label="或本地批量上传干声（多选）", file_count="multiple")
                        b_dir = gr.Textbox(label="或填写容器内目录路径（以上均未选时生效）", placeholder=str(RAW_DIR))
                        with gr.Row():
                            b_key = gr.Slider(-24, 24, value=0, step=1, label="变调")
                            b_formant = gr.Slider(-6, 6, value=0, step=0.5, label="共振峰")
                            b_pe = gr.Dropdown(PE_CHOICES, value="rmvpe", label="音高提取器")
                            b_method = gr.Dropdown(METHOD_CHOICES, value="auto", label="采样方法")
                            b_step = gr.Dropdown(choices=STEP_CHOICES, value="50", allow_custom_value=True, label="采样步数")
                        b_btn = gr.Button("开始批量转换", variant="primary")
                        with gr.Row():
                            b_zip = gr.File(label="下载结果 (.zip)")
                            b_log = gr.Textbox(label="批量日志", lines=3)

                    with gr.Accordion("原曲一条龙快速试听（自动分离 + 翻唱 + 混音，精细翻唱建议分步操作）", open=False):
                        gr.HTML('<div class="soft-guide-box">适合快速听 Demo：上传带伴奏的完整原曲，自动完成 <b>人声分离 → 和声处理 → 去混响 → 翻唱 → 混音</b>。想追求最好音质仍建议用左侧「音轨分离」分步精调。</div>')
                        oc_song = gr.Audio(label="上传带伴奏原曲（推荐真无损 WAV）", type="filepath")
                        with gr.Row():
                            oc_key = gr.Slider(-24, 24, value=0, step=1, label="变调（半音）")
                            oc_pe = gr.Dropdown(PE_CHOICES, value="rmvpe", label="音高提取器")
                        oc_hmode = gr.Radio(HARMONY_MODES, value=HARMONY_MODES[0], label="和声处理方式")
                        with gr.Row():
                            oc_vdb = gr.Slider(-12, 12, value=0, step=0.5, label="主唱音量 (dB)")
                            oc_hdb = gr.Slider(-18, 6, value=-4, step=0.5, label="和声音量 (dB)")
                            oc_idb = gr.Slider(-12, 12, value=-2, step=0.5, label="伴奏音量 (dB)")
                        with gr.Row():
                            oc_ln = gr.Checkbox(value=True, label="人声响度对齐")
                            oc_rev = gr.Checkbox(value=True, label="加轻量空间混响")
                        oc_btn = gr.Button("开始一条龙试听", variant="primary")
                        with gr.Row():
                            oc_vraw = gr.Audio(label="提取后的主唱干声", type="filepath")
                            oc_vconv = gr.Audio(label="翻唱后干声", type="filepath")
                            oc_mix = gr.Audio(label="最终混音成品", type="filepath")
                        oc_log = gr.Textbox(label="执行状态", lines=3)
                        with gr.Accordion("一条龙历史追溯 & 过程音轨排查", open=False):
                            with gr.Row(elem_classes=["aligned-row"]):
                                oc_task_dd = gr.Dropdown(label="选择一条龙历史任务", choices=list_oneclick_tasks(), scale=4)
                                oc_task_ref = gr.Button("刷新记录", scale=1)
                            oc_track_dd = gr.Dropdown(label="选择过程音轨试听（01人声/伴奏 → 02主唱/和声 → 03去混响 → 04翻唱 → 05混音）", choices=[])
                            oc_trace_player = gr.Audio(label="过程音轨回放", type="filepath")
                            oc_trace_log = gr.Textbox(label="完整执行追溯日志 (run_log.txt)", lines=8)

                with gr.Column(scale=4, min_width=0, elem_classes=["studio-aside-panel"]):
                    gr.HTML(
                        """
                        <div class="aside-title">翻唱记录</div>
                        <div class="aside-desc">转换完成的人声音频会呈现在这里，支持回听与历史对比。</div>
                        """
                    )
                    out_audio = gr.Audio(label="当前输出（WAV）", type="filepath")
                    with gr.Accordion("历史试听对比", open=False):
                        with gr.Row(elem_classes=["aligned-row"]):
                            hist_dd = gr.Dropdown(label="选择历史记录", choices=list_recent_results(), scale=4)
                            hist_ref_btn = gr.Button("刷新", scale=1)
                        hist_player = gr.Audio(label="历史试听", type="filepath")
                    with gr.Accordion("运行日志", open=False):
                        infer_log = gr.Textbox(label="", lines=8, show_label=False)

            # ===== 视图 2：音轨分离与混音 =====
            with gr.Row(visible=False, elem_classes=["workspace-row"]) as view_msst:
                with gr.Column(scale=7, min_width=0, elem_classes=["studio-main-card"]):
                    gr.HTML(
                        """
                        <div class="page-header">
                            <span class="page-title">音轨分离</span>
                            <span class="page-meta">MSST Mel-Band-Roformer · 支持一键三步直出干声与压缩包批量分离</span>
                        </div>
                        """
                    )
                    msst_in = gr.Audio(label="点击上传或拖动原曲 / 人声至此", type="filepath")
                    with gr.Row(elem_classes=["aligned-row"]):
                        msst_pick_dd = gr.Dropdown(
                            label="从容器目录选用（raw/ 与分离输出）",
                            choices=init_container_audios,
                            value=None,
                            scale=5,
                        )
                        msst_pick_ref = gr.Button("刷新", scale=1)
                        msst_pick_load = gr.Button("载入", scale=1)
                        msst_save_raw = gr.Button("存到 raw/", scale=2)

                    msst_btn = gr.Button("开始分离", variant="primary")

                    gr.HTML('<div class="section-label">分离方式</div>')
                    msst_model = gr.Radio(
                        choices=msst_choices,
                        value=ONE_CLICK_MSST_KEY,
                        label="",
                        show_label=False,
                        elem_id="msst-mode-radio",
                    )
                    with gr.Row():
                        msst_inst = gr.Checkbox(value=True, label="同时输出伴奏 / 和声 / 混响残差")
                        msst_tta = gr.Checkbox(value=False, label="TTA 增强（耗时约 3 倍）")

                    with gr.Accordion("批量分离（导入 .zip 压缩包或多首音频一键分离）", open=False):
                        gr.HTML('<div class="soft-guide-box">可直接上传包含多首歌曲的 <b>.zip 压缩包</b> 或多选音频，自动按上方选中的「分离方式」（支持<b>一键三步全分离</b>）批量处理并打包下载，结果同时留在 <code>results/</code> 方便后续直接训练或翻唱。</div>')
                        msst_batch_files = gr.File(label="上传多首音频或 .zip 压缩包", file_count="multiple")
                        msst_batch_btn = gr.Button("开始批量分离并打包", variant="primary")
                        with gr.Row():
                            msst_batch_zip = gr.File(label="下载批量分离结果 (.zip)")
                            msst_batch_log = gr.Textbox(label="批量分离日志", lines=4)

                    with gr.Accordion("翻唱人声与伴奏混音", open=False):
                        with gr.Row():
                            mix_v = gr.Audio(label="翻唱干声", type="filepath")
                            mix_i = gr.Audio(label="伴奏音轨", type="filepath")
                        with gr.Row():
                            mix_vdb = gr.Slider(-12, 12, value=0, step=0.5, label="人声音量 (dB)")
                            mix_idb = gr.Slider(-12, 12, value=-2, step=0.5, label="伴奏音量 (dB)")
                            mix_fmt = gr.Dropdown(["wav", "mp3"], value="wav", label="输出格式")
                        with gr.Row():
                            mix_ln = gr.Checkbox(value=True, label="响度对齐 (-14 LUFS)")
                            mix_rev = gr.Checkbox(value=True, label="轻量空间混响")
                            mix_comp = gr.Checkbox(value=True, label="动态压缩")
                        mix_btn = gr.Button("合并混音", variant="primary")
                        mix_out = gr.Audio(label="混音成品", type="filepath")
                        mix_log = gr.Textbox(label="混音状态", lines=2)

                    with gr.Accordion("更多工具（和声并入伴奏 / 单文件清洗）", open=False):
                        gr.Markdown("**和声 + 纯伴奏 → 带和声伴奏**（翻唱完主唱后保留原曲饱满和声）")
                        with gr.Row():
                            mh_h = gr.Audio(label="原曲和声", type="filepath")
                            mh_i = gr.Audio(label="纯伴奏", type="filepath")
                        with gr.Row(elem_classes=["aligned-row"]):
                            mh_hdb = gr.Slider(-12, 12, value=0, step=0.5, label="和声 (dB)")
                            mh_idb = gr.Slider(-12, 12, value=0, step=0.5, label="伴奏 (dB)")
                            mh_btn = gr.Button("合并生成")
                        mh_out = gr.Audio(label="带和声伴奏", type="filepath")
                        mh_log = gr.Textbox(label="合并状态", lines=2)

                        gr.Markdown("**单文件格式转换与清洗 (FFmpeg)**（用于翻唱前给单条干声去低频底噪、统一音量和转 44.1kHz WAV）")
                        clean_in = gr.Audio(label="待清洗音频", type="filepath")
                        with gr.Row():
                            clean_sr = gr.Dropdown([44100, 48000, 32000, 16000], value=44100, label="采样率")
                            clean_mono = gr.Checkbox(value=True, label="单声道")
                            clean_hp = gr.Checkbox(value=True, label="80Hz 去低噪")
                            clean_ln = gr.Checkbox(value=True, label="响度归一化")
                        clean_btn = gr.Button("开始清洗")
                        clean_out = gr.Audio(label="清洗后 WAV", type="filepath")
                        clean_log = gr.Textbox(label="处理结果", lines=2)

                with gr.Column(scale=4, min_width=0, elem_classes=["studio-aside-panel"]):
                    gr.HTML(
                        """
                        <div class="aside-title">分离记录</div>
                        <div class="aside-desc">分离出的音轨会自动保存到容器 results/ 目录，可直接去「语音翻唱」页选用。</div>
                        """
                    )
                    msst_out1 = gr.Audio(label="主音轨（纯净主唱干声 / 人声）", type="filepath")
                    msst_out2 = gr.Audio(label="副音轨（纯伴奏 / 和声 / 混响）", type="filepath")
                    with gr.Accordion("历史分离记录（回听与下载）", open=True):
                        with gr.Row(elem_classes=["aligned-row"]):
                            msst_hist_dd = gr.Dropdown(label="选择历史分离音轨", choices=list_msst_history(), scale=4)
                            msst_hist_ref = gr.Button("刷新", scale=1)
                        msst_hist_player = gr.Audio(label="历史音轨试听与下载", type="filepath")
                    with gr.Accordion("分离日志", open=False):
                        msst_log = gr.Textbox(label="", lines=6, show_label=False)

            # ===== 视图 3：模型训练 =====
            with gr.Row(visible=False, elem_classes=["workspace-row"]) as view_train:
                with gr.Column(scale=7, min_width=0, elem_classes=["studio-main-card"]):
                    gr.HTML(
                        """
                        <div class="page-header">
                            <span class="page-title">模型训练</span>
                            <span class="page-meta">建议准备 10~30 分钟无伴奏、无混响的纯净人声</span>
                        </div>
                        """
                    )

                    with gr.Accordion("第 1 步 · 准备数据集（自动转码 → 清洗 → 划分验证集）", open=True):
                        gr.HTML(
                            """
                            <div class="soft-guide-box">
                                <b>自动转码说明：</b>支持直接上传 <b>WAV / FLAC / MP3 / M4A / .zip</b>，入库时会自动转为 <b>44.1kHz 单声道 WAV</b>，无需提前手动转格式（但仍强烈推荐<b>真无损 WAV</b>，有损格式丢失的高频无法恢复）。<br>
                                <b>操作顺序：</b>① 长干声点「1. 切片入库」（已切好的短片段直接传右侧框）→ ② 点「2. 一键清洗」（去低噪 + 统一音量 + 剔除过短报错片段）→ ③ 点「3. 划分验证集」（抽 2% 当试听考题，<b>训练前必点</b>）。
                            </div>
                            """
                        )
                        with gr.Row():
                            slice_files = gr.File(label="上传完整长干声（自动按静音切段）", file_count="multiple")
                            tr_upload = gr.File(label="或直接上传已切好的短片段 / .zip", file_count="multiple")
                        with gr.Row():
                            sl_btn = gr.Button("1. 切片入库", variant="primary")
                            bc_btn = gr.Button("2. 一键清洗")
                            btn_draw = gr.Button("3. 划分验证集", variant="primary")

                        with gr.Accordion("数据集清理与细节参数", open=False):
                            gr.HTML(
                                """
                                <div class="soft-guide-box">
                                    <b>清理说明：</b><br>
                                    · <b>清空特征缓存</b>：仅删提取特征产生的中间文件，<b>保留你的干声 WAV</b>（改了音高提取器或声码器想重新提特征时用）。<br>
                                    · <b>清空全部训练干声</b>：删除训练集和验证集里的所有音频文件（<b>换新角色重新炼丹</b>时用）。
                                </div>
                                """
                            )
                            with gr.Row():
                                btn_ask_clr_feat = gr.Button("清空特征缓存（保留干声）", variant="stop")
                                btn_ask_clr_audio = gr.Button("清空全部训练干声", variant="stop")
                                btn_restore_val = gr.Button("退回验证集到训练集")
                                tr_check_btn = gr.Button("刷新体检")

                            with gr.Column(visible=False, elem_classes=["danger-confirm-box"]) as ds_confirm_box:
                                ds_confirm_msg = gr.HTML("")
                                with gr.Row():
                                    btn_ds_confirm_yes = gr.Button("确认永久删除", variant="stop")
                                    btn_ds_confirm_no = gr.Button("取消")

                            with gr.Row():
                                sl_th = gr.Slider(-60, -10, value=-40, step=1, label="切片静音阈值 (dB)")
                                sl_minlen = gr.Number(value=5000, label="最小切片 (ms)")
                                sl_minint = gr.Number(value=300, label="静音间隔 (ms)")
                            with gr.Row():
                                sl_maxsil = gr.Number(value=500, label="保留静音 (ms)")
                                sl_keep = gr.Number(value=2.2, label="过滤短于 (秒)")
                                tr_draw_pct = gr.Slider(1, 20, value=2, step=1, label="验证集抽取比例 (%)")
                            with gr.Row():
                                bc_hp = gr.Checkbox(value=True, label="80Hz 高通去低噪")
                                bc_ln = gr.Checkbox(value=True, label="响度归一化 (-14 LUFS)")
                                bc_trim = gr.Checkbox(value=True, label="裁剪首尾静音")

                    with gr.Accordion("第 2 步 · 新建角色与训练参数", open=False):
                        with gr.Row(elem_classes=["aligned-row"]):
                            role_name_tb = gr.Textbox(placeholder="例如: my_character", label="新角色名（自动建于 exp/ 下）", scale=3)
                            role_init_dd = gr.Dropdown(
                                [
                                    "从零训练新角色（推荐 · 自动加载官方底模）",
                                    "基于已有模型继续微调",
                                ],
                                value="从零训练新角色（推荐 · 自动加载官方底模）",
                                label="训练方式",
                                scale=4,
                            )
                            btn_create_role = gr.Button("创建角色", variant="primary", scale=2)
                        with gr.Row(visible=False) as ft_pick_row:
                            ft_dir_dd = gr.Dropdown(choices=init_dirs, value=init_dir, label="选择已有模型目录")
                            ft_step_dd = gr.Dropdown(choices=init_steps, value=init_step, label="选择微调起始步数")
                        role_msg = gr.Textbox(label="角色状态", lines=1)

                        with gr.Row(elem_classes=["aligned-row"]):
                            cfg_expdir = gr.Dropdown(choices=list_exp_dirs(), value="exp/reflow-test", allow_custom_value=True, label="当前训练目录 (expdir)", scale=4)
                            btn_auto_cfg = gr.Button("按显存与时长自动配参", scale=3)
                            btn_savecfg = gr.Button("保存训练配置", variant="primary", scale=2)
                        cfg_msg = gr.Textbox(label="配置状态", lines=1)

                        with gr.Accordion("高级超参（configs/reflow.yaml，不懂可保持默认）", open=False):
                            with gr.Row():
                                cfg_voc = gr.Dropdown(["pc_nsf_hifigan (推荐 2025.02)", "nsf_hifigan (原版)"], value="pc_nsf_hifigan (推荐 2025.02)", label="声码器")
                                cfg_f0 = gr.Dropdown(PE_CHOICES, value="rmvpe", label="音高提取器")
                                cfg_dur = gr.Number(value=2, label="窗长 (秒)")
                                cfg_paug = gr.Checkbox(value=True, label="音高增强 (use_pitch_aug)")
                            with gr.Row():
                                cfg_bs = gr.Number(value=48, precision=0, label="Batch Size")
                                cfg_lr = gr.Number(value=0.0002, label="学习率 (lr)")
                                cfg_decay = gr.Number(value=50000, precision=0, label="衰减步数")
                                cfg_gamma = gr.Number(value=0.5, label="gamma")
                                cfg_epochs = gr.Number(value=100000, precision=0, label="最大 Epochs")
                                cfg_lddsp = gr.Number(value=1.0, label="lambda_ddsp")
                            with gr.Row():
                                cfg_amp = gr.Dropdown(["fp16", "bf16", "fp32"], value="fp16", label="混合精度")
                                cfg_cache = gr.Checkbox(value=True, label="缓存全部数据")
                                cfg_cdev = gr.Dropdown(["cpu", "cuda"], value="cpu", label="缓存设备")
                                cfg_nw = gr.Number(value=2, precision=0, label="num_workers")
                                cfg_ival = gr.Number(value=2000, precision=0, label="验证间隔")
                                cfg_isave = gr.Number(value=4000, precision=0, label="保存间隔")

                    with gr.Accordion("第 3 步 · 提取特征并启动训练", open=False):
                        gr.HTML('<div class="soft-guide-box">先点击 <b>1. 提取特征</b>，完成后再点 <b>2. 启动训练</b>。训练期间可打开 <b>6006 端口 TensorBoard → Audio</b> 试听，声音清晰自然即可随时停止。</div>')
                        with gr.Row():
                            btn_prep = gr.Button("1. 提取特征", variant="primary")
                            btn_start_tr = gr.Button("2. 启动训练", variant="primary")
                            btn_stop_tr = gr.Button("停止训练", variant="stop")

                    with gr.Accordion("第 4 步 · 多步数试听对比 & 一键导入模型库 / 下载", open=False):
                        with gr.Row(elem_classes=["aligned-row"]):
                            cmp_exp = gr.Dropdown(choices=list_exp_dirs(), value="exp/reflow-test", label="实验目录", scale=4)
                            cmp_ref_btn = gr.Button("扫描权重", scale=2)
                        cmp_ckpts = gr.CheckboxGroup(choices=list_steps_in_dir("exp/reflow-test"), label="勾选待对比步数（最多 3 个）")
                        with gr.Row():
                            cmp_audio = gr.Audio(label="测试干声", type="filepath")
                            cmp_key = gr.Slider(-24, 24, value=0, step=1, label="变调")
                            cmp_pe = gr.Dropdown(PE_CHOICES, value="rmvpe", label="音高提取器")
                        cmp_btn = gr.Button("对比试听", variant="primary")
                        with gr.Row():
                            cmp_o1 = gr.Audio(label="轨道 1", type="filepath")
                            cmp_o2 = gr.Audio(label="轨道 2", type="filepath")
                            cmp_o3 = gr.Audio(label="轨道 3", type="filepath")
                        cmp_log = gr.Textbox(label="对比日志", lines=2)

                        gr.HTML('<div class="soft-guide-box" style="margin-top:8px;"><b>直接导入与导出：</b>点击下方按钮会同时把精简后的模型<b>直接导入到「语音翻唱」模型库</b>（自动刷新列表立即可用），并生成可下载的 <code>.zip</code> 备份包。</div>')
                        with gr.Row(elem_classes=["aligned-row"]):
                            exp_ckpt_dd = gr.Dropdown(choices=list_steps_in_dir("exp/reflow-test"), label="定稿权重", scale=3)
                            exp_alias_tb = gr.Textbox(placeholder="角色名", label="导入模型库的名称", scale=3)
                            exp_btn = gr.Button("导入模型库并打包下载", variant="primary", scale=3)
                        with gr.Row():
                            exp_zip = gr.File(label="精简模型包 (.zip)")
                            exp_log = gr.Textbox(label="导入与导出结果", lines=2)

                with gr.Column(scale=4, min_width=0, elem_classes=["studio-aside-panel"]):
                    gr.HTML('<div class="aside-title">训练监控</div>')
                    status_banner = gr.HTML(format_status_banner())
                    btn_refresh_log = gr.Button("刷新 GPU 与训练状态")
                    tr_ds_info = gr.Textbox(label="数据集体检", value=inspect_dataset(), lines=5)
                    tr_log_box = gr.Textbox(label="实时日志", lines=10)

            # ===== 视图 4：系统维护 =====
            with gr.Row(visible=False, elem_classes=["workspace-row"]) as view_sys:
                with gr.Column(scale=7, min_width=0, elem_classes=["studio-main-card"]):
                    gr.HTML(
                        """
                        <div class="page-header">
                            <span class="page-title">系统维护</span>
                            <span class="page-meta">容器文件上传与批量清理 · 模型管理 · 镜像纯净重置</span>
                        </div>
                        """
                    )

                    with gr.Accordion("容器音频文件管理（上传 / 下载 / 全选 / 批量删除）", open=True):
                        with gr.Row(elem_classes=["aligned-row"]):
                            fm_dir_dd = gr.Dropdown(
                                choices=list(MANAGED_DIRS.keys()),
                                value="待处理音频目录 (raw/)",
                                label="选择要管理的目录",
                                scale=4,
                            )
                            fm_ref_btn = gr.Button("刷新列表", scale=1)
                        with gr.Row(elem_classes=["aligned-row"]):
                            fm_upload_files = gr.File(label="上传音频或 .zip 直接保存到上方目录", file_count="multiple", scale=4)
                            fm_upload_btn = gr.Button("保存到该目录", variant="primary", scale=2)
                        fm_files_cg = gr.CheckboxGroup(
                            choices=list_managed_files("待处理音频目录 (raw/)"),
                            label="文件列表（勾选后可打包下载或批量删除）",
                            elem_classes=["file-checklist"],
                        )
                        with gr.Row():
                            fm_all_btn = gr.Button("全选")
                            fm_none_btn = gr.Button("取消全选")
                            fm_dl_btn = gr.Button("下载所选文件", variant="primary")
                            fm_ask_del_btn = gr.Button("删除所选文件", variant="stop")
                        fm_dl_file = gr.File(label="所选文件下载（多选自动打包为 .zip）")
                        with gr.Column(visible=False, elem_classes=["danger-confirm-box"]) as fm_confirm_box:
                            fm_confirm_msg = gr.HTML("")
                            with gr.Row():
                                fm_confirm_yes = gr.Button("确认永久删除所选文件", variant="stop")
                                fm_confirm_no = gr.Button("取消")

                    with gr.Accordion("已训练模型管理（打包下载 / 清理，保护官方底模）", open=False):
                        gr.HTML('<div class="soft-guide-box">可勾选<b>打包下载备份</b>或删除不再需要的角色模型目录（<code>models/</code> 与 <code>exp/</code>）。系统已自动隐藏并保护官方底模 <code>exp/reflow-test</code> 防止误删。</div>')
                        mm_models_cg = gr.CheckboxGroup(
                            choices=list_deletable_models(),
                            label="角色模型目录（勾选后可下载或删除）",
                            elem_classes=["file-checklist"],
                        )
                        with gr.Row():
                            mm_ref_btn = gr.Button("刷新列表")
                            mm_all_btn = gr.Button("全选")
                            mm_none_btn = gr.Button("取消全选")
                            mm_dl_btn = gr.Button("打包下载所选模型", variant="primary")
                            mm_ask_del_btn = gr.Button("删除所选模型", variant="stop")
                        mm_dl_file = gr.File(label="所选模型下载包 (.zip)")
                        with gr.Column(visible=False, elem_classes=["danger-confirm-box"]) as mm_confirm_box:
                            mm_confirm_msg = gr.HTML("")
                            with gr.Row():
                                mm_confirm_yes = gr.Button("确认永久删除所选模型", variant="stop")
                                mm_confirm_no = gr.Button("取消")

                    with gr.Accordion("导入外部模型包 (.zip)", open=False):
                        with gr.Row(elem_classes=["aligned-row"]):
                            imp_zip = gr.File(label="上传含 .pt 与 config.yaml 的 .zip", scale=3)
                            imp_name = gr.Textbox(label="导入后文件夹名（留空取压缩包名）", placeholder="my_singer", scale=3)
                            imp_btn = gr.Button("导入到模型库", variant="primary", scale=2)

                    with gr.Accordion("社区镜像发布前：一键重置为纯净状态", open=False):
                        gr.HTML('<div class="soft-guide-box">清空临时试听、训练集切片、特征缓存与日志，<b>完整保留所有预训练底模与示例模型</b>。</div>')
                        reset_ask_btn = gr.Button("一键重置为纯净镜像状态", variant="stop")
                        with gr.Column(visible=False, elem_classes=["danger-confirm-box"]) as reset_confirm_box:
                            gr.HTML('<div class="danger-warn-text">⚠️ 警告：即将清空 results/、cache/、data/ 下全部试听文件与训练数据（无法恢复），但保留预训练底模与 models/ 模型库。确认执行？</div>')
                            with gr.Row():
                                reset_confirm_yes = gr.Button("确认重置镜像", variant="stop")
                                reset_confirm_no = gr.Button("取消")

                with gr.Column(scale=4, min_width=0, elem_classes=["studio-aside-panel"]):
                    gr.HTML(
                        """
                        <div class="aside-title">操作反馈</div>
                        <div class="aside-desc">文件保存、清理、模型导入与镜像重置的结果会显示在这里。</div>
                        """
                    )
                    sys_op_log = gr.Textbox(label="文件与模型管理反馈", lines=4)
                    imp_log = gr.Textbox(label="模型导入结果", lines=3)
                    reset_log = gr.Textbox(label="镜像重置与空间占用", lines=6)

    # ==================== 侧边栏导航切换 ====================
    def _switch_page(idx: int):
        vis = [gr.update(visible=(i == idx)) for i in range(4)]
        cls = [gr.update(elem_classes=["nav-btn", "nav-active"] if i == idx else ["nav-btn"]) for i in range(4)]
        return vis + cls

    nav_outputs = [view_infer, view_msst, view_train, view_sys, nav_infer, nav_msst, nav_train, nav_sys]
    nav_infer.click(lambda: _switch_page(0), outputs=nav_outputs)
    nav_msst.click(lambda: _switch_page(1), outputs=nav_outputs)
    nav_train.click(lambda: _switch_page(2), outputs=nav_outputs)
    nav_sys.click(lambda: _switch_page(3), outputs=nav_outputs)

    # 动态效果开关（持久化）
    fx_cb.change(apply_ui_prefs, inputs=[fx_cb], outputs=[fx_style_box, sidebar_pill])

    # ==================== 容器目录直选与保存事件 ====================
    infer_pick_ref.click(lambda: gr.update(choices=list_container_audio_choices()), outputs=infer_pick_dd)
    infer_pick_load.click(lambda p: p if p else gr.update(), inputs=infer_pick_dd, outputs=in_audio)
    infer_save_raw.click(save_audio_to_raw, inputs=in_audio, outputs=[infer_pick_dd, msst_pick_dd, infer_log])

    msst_pick_ref.click(lambda: gr.update(choices=list_container_audio_choices()), outputs=msst_pick_dd)
    msst_pick_load.click(lambda p: p if p else gr.update(), inputs=msst_pick_dd, outputs=msst_in)
    msst_save_raw.click(save_audio_to_raw, inputs=msst_in, outputs=[msst_pick_dd, infer_pick_dd, msst_log])

    def _refresh_batch_list():
        return gr.update(choices=list_container_audio_choices(), value=[])

    def _select_all_batch():
        c = list_container_audio_choices()
        return gr.update(choices=c, value=[v for _, v in c])

    b_pick_ref.click(_refresh_batch_list, outputs=b_pick_cg, queue=False)
    b_pick_all.click(_select_all_batch, outputs=b_pick_cg, queue=False)
    b_pick_none.click(lambda: gr.update(value=[]), outputs=b_pick_cg, queue=False)

    # ==================== 语音翻唱与一条龙绑定 ====================
    model_dir_dd.change(on_model_dir_change, inputs=model_dir_dd, outputs=model_step_dd, queue=False)
    refresh_btn.click(refresh_all_models, outputs=[model_dir_dd, model_step_dd], queue=False)
    hist_ref_btn.click(lambda: gr.update(choices=list_recent_results()), outputs=hist_dd, queue=False)
    hist_dd.change(lambda p: p, inputs=hist_dd, outputs=hist_player, queue=False)
    infer_btn.click(
        run_inference,
        inputs=[in_audio, model_step_dd, key_slider, formant_slider, reg_slider, pe_dd, method_dd, step_tb, t_start_num, th_num, fmin_num, fmax_num],
        outputs=[out_audio, infer_log, hist_dd],
    )
    b_btn.click(
        run_batch_with_picker,
        inputs=[b_pick_cg, b_files, b_dir, model_step_dd, b_key, b_formant, b_pe, b_method, b_step],
        outputs=[b_zip, b_log],
    )
    oc_btn.click(
        run_one_click_cover,
        inputs=[oc_song, model_step_dd, oc_key, oc_pe, oc_hmode, oc_hdb, oc_vdb, oc_idb, oc_ln, oc_rev],
        outputs=[oc_vraw, oc_vconv, oc_mix, oc_log, oc_task_dd, oc_track_dd, oc_trace_player, oc_trace_log],
    )
    oc_task_ref.click(lambda: gr.update(choices=list_oneclick_tasks()), outputs=oc_task_dd, queue=False)
    oc_task_dd.change(load_oneclick_task_details, inputs=oc_task_dd, outputs=[oc_track_dd, oc_trace_player, oc_trace_log], queue=False)
    oc_track_dd.change(lambda p: p, inputs=oc_track_dd, outputs=oc_trace_player, queue=False)

    # ==================== 音轨分离（含一键三步、压缩包批量与历史记录）与混音绑定 ====================
    msst_btn.click(
        run_msst_enhanced,
        inputs=[msst_in, msst_model, msst_tta, msst_inst],
        outputs=[msst_out1, msst_out2, msst_log, infer_pick_dd, msst_pick_dd, msst_hist_dd],
    )
    msst_batch_btn.click(
        run_msst_batch_zip,
        inputs=[msst_batch_files, msst_model, msst_tta, msst_inst],
        outputs=[msst_batch_zip, msst_batch_log, infer_pick_dd, msst_pick_dd, msst_hist_dd],
    )
    msst_hist_ref.click(lambda: gr.update(choices=list_msst_history()), outputs=msst_hist_dd, queue=False)
    msst_hist_dd.change(lambda p: p, inputs=msst_hist_dd, outputs=msst_hist_player, queue=False)
    mh_btn.click(merge_harmony_with_inst, inputs=[mh_h, mh_i, mh_hdb, mh_idb], outputs=[mh_out, mh_log])
    clean_btn.click(run_audio_format_clean, inputs=[clean_in, clean_sr, clean_mono, clean_hp, clean_ln], outputs=[clean_out, clean_log])
    mix_btn.click(run_mix_audio, inputs=[mix_v, mix_i, mix_vdb, mix_idb, mix_ln, mix_rev, mix_comp, mix_fmt], outputs=[mix_out, mix_log])

    # ==================== 模型训练与二次确认删除绑定 ====================
    sl_btn.click(run_audio_slicer, inputs=[slice_files, sl_th, sl_minlen, sl_minint, sl_maxsil, sl_keep], outputs=tr_ds_info)
    tr_upload.upload(upload_train_files, inputs=tr_upload, outputs=tr_ds_info)
    bc_btn.click(run_batch_clean_train, inputs=[bc_hp, bc_ln, bc_trim], outputs=tr_ds_info)
    tr_check_btn.click(inspect_dataset, outputs=tr_ds_info, queue=False)
    btn_draw.click(run_draw_val, inputs=tr_draw_pct, outputs=tr_ds_info)
    btn_restore_val.click(restore_val_to_train, outputs=tr_ds_info)

    def _ask_clear_feat():
        return (
            "feat",
            gr.update(visible=True),
            '<div class="danger-warn-text">⚠️ 警告：即将删除已提取的特征缓存（f0 / units / mel / volume 等中间文件），你的干声 WAV 会保留。确认清空特征缓存？</div>',
        )

    def _ask_clear_audio():
        return (
            "audio",
            gr.update(visible=True),
            '<div class="danger-warn-text">⚠️ 高危警告：即将永久删除训练集（data/train/audio）和验证集（data/val/audio）里的全部干声音频文件，删除后无法恢复！确认清空？</div>',
        )

    def _exec_ds_clear(action_kind):
        if action_kind == "feat":
            res = clear_dataset_and_cache(False, True)
        elif action_kind == "audio":
            res = clear_dataset_and_cache(True, False)
        else:
            res = inspect_dataset()
        return "", gr.update(visible=False), res

    btn_ask_clr_feat.click(_ask_clear_feat, outputs=[pending_ds_action, ds_confirm_box, ds_confirm_msg], queue=False)
    btn_ask_clr_audio.click(_ask_clear_audio, outputs=[pending_ds_action, ds_confirm_box, ds_confirm_msg], queue=False)
    btn_ds_confirm_yes.click(_exec_ds_clear, inputs=[pending_ds_action], outputs=[pending_ds_action, ds_confirm_box, tr_ds_info])
    btn_ds_confirm_no.click(lambda: ("", gr.update(visible=False)), outputs=[pending_ds_action, ds_confirm_box], queue=False)

    role_init_dd.change(lambda m: gr.update(visible=m.startswith("基于已有模型")), inputs=role_init_dd, outputs=ft_pick_row, queue=False)
    ft_dir_dd.change(on_model_dir_change, inputs=ft_dir_dd, outputs=ft_step_dd, queue=False)
    btn_create_role.click(create_role_exp, inputs=[role_name_tb, role_init_dd, ft_step_dd], outputs=[cfg_expdir, role_msg])
    btn_auto_cfg.click(recommend_train_params, outputs=[cfg_bs, cfg_lr, cfg_amp, cfg_cdev, cfg_isave, cfg_msg])
    btn_savecfg.click(
        save_train_config,
        inputs=[cfg_expdir, cfg_voc, cfg_f0, cfg_dur, cfg_paug, cfg_bs, cfg_lr, cfg_decay, cfg_gamma, cfg_epochs, cfg_lddsp, cfg_amp, cfg_cache, cfg_cdev, cfg_nw, cfg_ival, cfg_isave],
        outputs=cfg_msg,
    )
    btn_prep.click(run_preprocess, outputs=tr_log_box)
    btn_start_tr.click(lambda: start_training(False), outputs=[status_banner, tr_log_box])
    btn_stop_tr.click(stop_training, outputs=[status_banner, tr_log_box])
    btn_refresh_log.click(get_gpu_and_train_status, outputs=[status_banner, tr_log_box], queue=False)

    def _refresh_exp_ckpts(exp_rel):
        c = list_steps_in_dir(exp_rel)
        vals = [v for _, v in c]
        return gr.update(choices=c, value=vals[:3]), gr.update(choices=c, value=vals[0] if vals else None)

    cmp_ref_btn.click(_refresh_exp_ckpts, inputs=cmp_exp, outputs=[cmp_ckpts, exp_ckpt_dd], queue=False)
    cmp_btn.click(compare_checkpoints, inputs=[cmp_audio, cmp_ckpts, cmp_key, cmp_pe], outputs=[cmp_o1, cmp_o2, cmp_o3, cmp_log])
    exp_btn.click(archive_and_refresh_models, inputs=[exp_ckpt_dd, exp_alias_tb], outputs=[exp_zip, exp_log, model_dir_dd, model_step_dd])

    # ==================== 系统维护：上传保存 / 文件管理 / 模型清理 / 重置 ====================
    fm_dir_dd.change(lambda d: (gr.update(choices=list_managed_files(d), value=[]), gr.update(visible=False)), inputs=fm_dir_dd, outputs=[fm_files_cg, fm_confirm_box], queue=False)
    fm_ref_btn.click(lambda d: gr.update(choices=list_managed_files(d), value=[]), inputs=fm_dir_dd, outputs=fm_files_cg, queue=False)
    fm_upload_btn.click(upload_to_managed_dir, inputs=[fm_dir_dd, fm_upload_files], outputs=[fm_files_cg, sys_op_log])
    fm_all_btn.click(lambda d: gr.update(choices=list_managed_files(d), value=[v for _, v in list_managed_files(d)]), inputs=fm_dir_dd, outputs=fm_files_cg, queue=False)
    fm_none_btn.click(lambda: gr.update(value=[]), outputs=fm_files_cg, queue=False)

    def _ask_del_files(dir_key, sel):
        if not sel:
            return gr.update(visible=False), "", "请先勾选要删除的文件。"
        return (
            gr.update(visible=True),
            f'<div class="danger-warn-text">⚠️ 警告：即将永久删除「{dir_key}」中勾选的 <b>{len(sel)}</b> 个文件，删除后无法恢复！确认执行？</div>',
            f"等待确认删除 {len(sel)} 个文件...",
        )

    def _exec_del_files(dir_key, sel):
        cg_upd, msg = delete_selected_files(dir_key, sel)
        return cg_upd, gr.update(visible=False), msg

    fm_dl_btn.click(download_selected_files, inputs=[fm_dir_dd, fm_files_cg], outputs=[fm_dl_file, sys_op_log])
    fm_ask_del_btn.click(_ask_del_files, inputs=[fm_dir_dd, fm_files_cg], outputs=[fm_confirm_box, fm_confirm_msg, sys_op_log], queue=False)
    fm_confirm_yes.click(_exec_del_files, inputs=[fm_dir_dd, fm_files_cg], outputs=[fm_files_cg, fm_confirm_box, sys_op_log])
    fm_confirm_no.click(lambda: (gr.update(visible=False), "已取消删除。"), outputs=[fm_confirm_box, sys_op_log], queue=False)

    mm_ref_btn.click(lambda: gr.update(choices=list_deletable_models(), value=[]), outputs=mm_models_cg, queue=False)
    mm_all_btn.click(lambda: gr.update(choices=list_deletable_models(), value=[v for _, v in list_deletable_models()]), outputs=mm_models_cg, queue=False)
    mm_none_btn.click(lambda: gr.update(value=[]), outputs=mm_models_cg, queue=False)
    mm_dl_btn.click(download_selected_models, inputs=[mm_models_cg], outputs=[mm_dl_file, sys_op_log])

    def _ask_del_models(sel):
        if not sel:
            return gr.update(visible=False), "", "请先勾选要删除的模型目录。"
        return (
            gr.update(visible=True),
            f'<div class="danger-warn-text">⚠️ 高危警告：即将永久删除以下 <b>{len(sel)}</b> 个模型目录（含全部权重文件，无法恢复）：<code>{", ".join(sel)}</code>。确认删除？</div>',
            f"等待确认删除 {len(sel)} 个模型目录...",
        )

    def _exec_del_models(sel):
        cg_upd, msg = delete_selected_models(sel)
        dirs_upd, steps_upd = refresh_all_models()
        return cg_upd, gr.update(visible=False), msg, dirs_upd, steps_upd

    mm_ask_del_btn.click(_ask_del_models, inputs=[mm_models_cg], outputs=[mm_confirm_box, mm_confirm_msg, sys_op_log], queue=False)
    mm_confirm_yes.click(_exec_del_models, inputs=[mm_models_cg], outputs=[mm_models_cg, mm_confirm_box, sys_op_log, model_dir_dd, model_step_dd])
    mm_confirm_no.click(lambda: (gr.update(visible=False), "已取消删除模型。"), outputs=[mm_confirm_box, sys_op_log], queue=False)

    imp_btn.click(import_external_model_zip, inputs=[imp_zip, imp_name], outputs=[imp_log, model_dir_dd, model_step_dd])

    reset_ask_btn.click(lambda: gr.update(visible=True), outputs=reset_confirm_box, queue=False)
    reset_confirm_no.click(lambda: (gr.update(visible=False), "已取消镜像重置。"), outputs=[reset_confirm_box, reset_log], queue=False)
    reset_confirm_yes.click(lambda: (gr.update(visible=False), reset_to_clean_image()), outputs=[reset_confirm_box, reset_log])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8, max_size=64).launch(server_name="0.0.0.0", server_port=7860, share=False)

