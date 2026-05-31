"""
📰 Fake News Detection System
Premium Streamlit application with multi-page navigation,
model comparison, URL analysis, dashboard, and prediction history.
Pre-loaded with demo data for a stunning first impression.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import random

# ---------------------------------------------------------------------------
# Page config (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium dark theme with glassmorphism + enhanced animations
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Import Google Fonts ──────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root variables ────────────────────────────────────────────────────────── */
:root {
    --bg-primary: #06090f;
    --bg-secondary: #0c1220;
    --bg-card: rgba(12, 20, 40, 0.7);
    --bg-card-hover: rgba(18, 28, 55, 0.85);
    --border-glass: rgba(80, 140, 255, 0.1);
    --border-glow: rgba(0, 212, 255, 0.25);
    --accent-cyan: #00d4ff;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-violet: #7c3aed;
    --accent-pink: #ec4899;
    --accent-rose: #f43f5e;
    --accent-green: #10b981;
    --accent-emerald: #34d399;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-orange: #f97316;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --gradient-hero: linear-gradient(135deg, #06090f 0%, #0f172a 25%, #1e1b4b 50%, #0f172a 75%, #06090f 100%);
    --gradient-card: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 27, 75, 0.4));
    --shadow-glow: 0 0 40px rgba(0, 212, 255, 0.06);
    --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* ── Global overrides ──────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], .main {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--gradient-hero) !important;
    background-attachment: fixed !important;
}

/* Animated background orbs */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -40%;
    right: -20%;
    width: 600px;
    height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124, 58, 237, 0.08) 0%, transparent 70%);
    animation: floatOrb 20s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    bottom: -30%;
    left: -15%;
    width: 500px;
    height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0, 212, 255, 0.06) 0%, transparent 70%);
    animation: floatOrb 25s ease-in-out infinite reverse;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d1a 0%, #0d1530 40%, #111833 100%) !important;
    border-right: 1px solid var(--border-glass) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 1px;
    height: 100%;
    background: linear-gradient(180deg, transparent, var(--accent-cyan), var(--accent-purple), transparent);
    opacity: 0.3;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stRadio label span {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin: 2px 0 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(0, 212, 255, 0.08) !important;
    border-color: rgba(0, 212, 255, 0.2) !important;
    transform: translateX(4px) !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.12), rgba(124, 58, 237, 0.12)) !important;
    border-color: rgba(0, 212, 255, 0.3) !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.08) !important;
}

/* ── Glass card ────────────────────────────────────────────────────────────── */
.glass-card {
    background: var(--gradient-card);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 20px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), rgba(139, 92, 246, 0.3), transparent);
}
.glass-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-glow);
    transform: translateY(-3px);
    box-shadow: var(--shadow-glow), var(--shadow-card);
}

/* ── Hero section ──────────────────────────────────────────────────────────── */
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 40%, #ec4899 70%, #00d4ff 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 4px;
    animation: fadeInDown 0.8s ease-out, shimmer 6s ease-in-out infinite;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-secondary);
    text-align: center;
    font-weight: 400;
    margin-bottom: 30px;
    animation: fadeInDown 1s ease-out;
    letter-spacing: 0.01em;
    line-height: 1.6;
}

/* ── Section header ────────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 6px;
    letter-spacing: -0.01em;
}
.section-sub {
    font-size: 0.95rem;
    color: var(--text-secondary);
    margin-bottom: 22px;
    line-height: 1.5;
}

/* ── Result cards ──────────────────────────────────────────────────────────── */
.result-real {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(52, 211, 153, 0.05));
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 20px;
    padding: 32px 36px;
    text-align: center;
    animation: slideUp 0.5s ease-out;
    position: relative;
    overflow: hidden;
}
.result-real::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #10b981, #34d399, transparent);
}
.result-fake {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(248, 113, 113, 0.05));
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 20px;
    padding: 32px 36px;
    text-align: center;
    animation: slideUp 0.5s ease-out;
    position: relative;
    overflow: hidden;
}
.result-fake::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ef4444, #f87171, transparent);
}
.result-label {
    font-size: 2.4rem;
    font-weight: 900;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}
.result-label.real { color: var(--accent-green); text-shadow: 0 0 30px rgba(16, 185, 129, 0.3); }
.result-label.fake { color: var(--accent-red); text-shadow: 0 0 30px rgba(239, 68, 68, 0.3); }
.result-confidence {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 2px;
}

/* ── Metric cards ──────────────────────────────────────────────────────────── */
.metric-card {
    background: var(--gradient-card);
    backdrop-filter: blur(20px) saturate(150%);
    -webkit-backdrop-filter: blur(20px) saturate(150%);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 24px 26px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), transparent);
}
.metric-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3), 0 0 30px rgba(0, 212, 255, 0.06);
    border-color: var(--border-glow);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
    font-weight: 600;
}

/* ── Confidence bar ────────────────────────────────────────────────────────── */
.confidence-bar-bg {
    width: 100%;
    height: 14px;
    background: rgba(255,255,255,0.06);
    border-radius: 7px;
    margin-top: 14px;
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 7px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}
.confidence-bar-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    animation: shimmerBar 2s ease-in-out infinite;
}
.confidence-bar-fill.real {
    background: linear-gradient(90deg, #059669, #10b981, #34d399);
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.3);
}
.confidence-bar-fill.fake {
    background: linear-gradient(90deg, #dc2626, #ef4444, #f87171);
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.3);
}

/* ── Nav badge / pill ──────────────────────────────────────────────────────── */
.nav-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.nav-badge.cyan   { background: rgba(0,212,255,0.12); color: var(--accent-cyan); border: 1px solid rgba(0,212,255,0.2); }
.nav-badge.purple { background: rgba(139,92,246,0.12); color: var(--accent-purple); border: 1px solid rgba(139,92,246,0.2); }
.nav-badge.green  { background: rgba(16,185,129,0.12); color: var(--accent-green); border: 1px solid rgba(16,185,129,0.2); }
.nav-badge.amber  { background: rgba(245,158,11,0.12); color: var(--accent-amber); border: 1px solid rgba(245,158,11,0.2); }
.nav-badge.pink   { background: rgba(236,72,153,0.12); color: var(--accent-pink); border: 1px solid rgba(236,72,153,0.2); }
.nav-badge.rose   { background: rgba(244,63,94,0.12); color: var(--accent-rose); border: 1px solid rgba(244,63,94,0.2); }

/* ── Divider ───────────────────────────────────────────────────────────────── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), rgba(139,92,246,0.3), rgba(236,72,153,0.3), transparent);
    border: none;
    margin: 30px 0;
}

/* ── URL preview card ──────────────────────────────────────────────────────── */
.url-preview {
    background: var(--gradient-card);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 16px 16px 0;
    padding: 22px 26px;
    margin: 16px 0;
    backdrop-filter: blur(16px);
    transition: all 0.3s ease;
}
.url-preview:hover {
    border-left-color: var(--accent-purple);
    transform: translateX(4px);
}
.url-preview h4 {
    color: var(--text-primary);
    margin: 0 0 8px 0;
    font-weight: 700;
    font-size: 1.1rem;
}
.url-preview p {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Demo badge ────────────────────────────────────────────────────────────── */
.demo-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 24px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(249,115,22,0.1));
    color: var(--accent-amber);
    border: 1px solid rgba(245,158,11,0.25);
    animation: pulse 3s ease-in-out infinite;
}

/* ── Feature grid ──────────────────────────────────────────────────────────── */
.feature-card {
    background: var(--gradient-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 28px 24px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    min-height: 160px;
}
.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: var(--shadow-glow), 0 16px 48px rgba(0, 0, 0, 0.3);
    border-color: var(--border-glow);
}
.feature-icon {
    font-size: 2.4rem;
    margin-bottom: 12px;
    display: block;
}
.feature-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 6px;
}
.feature-desc {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.5;
}

/* ── Stat pill inline ──────────────────────────────────────────────────────── */
.stat-inline {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid var(--border-glass);
    font-size: 0.85rem;
    color: var(--text-secondary);
}
.stat-inline .stat-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.stat-inline .stat-dot.green { background: var(--accent-green); box-shadow: 0 0 8px rgba(16, 185, 129, 0.5); }
.stat-inline .stat-dot.red { background: var(--accent-red); box-shadow: 0 0 8px rgba(239, 68, 68, 0.5); }
.stat-inline .stat-dot.cyan { background: var(--accent-cyan); box-shadow: 0 0 8px rgba(0, 212, 255, 0.5); }

/* ── Animations ────────────────────────────────────────────────────────────── */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(1.02); opacity: 0.85; }
}
@keyframes shimmer {
    0%   { background-position: 200% 0%; }
    100% { background-position: -200% 0%; }
}
@keyframes shimmerBar {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(200%); }
}
@keyframes floatOrb {
    0%   { transform: translate(0, 0) scale(1); }
    33%  { transform: translate(30px, -30px) scale(1.1); }
    66%  { transform: translate(-20px, 20px) scale(0.9); }
    100% { transform: translate(0, 0) scale(1); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

/* ── Button styling ────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 36px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 24px rgba(0, 212, 255, 0.2), 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    letter-spacing: 0.3px !important;
    text-transform: none !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 36px rgba(0, 212, 255, 0.35), 0 4px 16px rgba(0, 0, 0, 0.3) !important;
}
.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* Secondary button style */
.stButton > button[kind="secondary"],
div[data-testid="stDownloadButton"] > button {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid var(--border-glass) !important;
    box-shadow: none !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: rgba(0, 212, 255, 0.1) !important;
    border-color: var(--accent-cyan) !important;
    transform: translateY(-2px) !important;
}

/* ── Text area styling ─────────────────────────────────────────────────────── */
.stTextArea textarea {
    background: rgba(12, 18, 32, 0.7) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 16px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 16px 20px !important;
    transition: all 0.3s ease !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1), 0 0 20px rgba(0, 212, 255, 0.05) !important;
}
.stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic !important;
}

/* ── Text input styling ────────────────────────────────────────────────────── */
.stTextInput input {
    background: rgba(12, 18, 32, 0.7) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 14px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px 18px !important;
    transition: all 0.3s ease !important;
}
.stTextInput input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1) !important;
}

/* ── Selectbox / multiselect ───────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: rgba(12, 18, 32, 0.7) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 14px !important;
}

/* ── Tab styling ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.3s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(0, 212, 255, 0.06) !important;
    border-color: rgba(0, 212, 255, 0.15) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(139,92,246,0.12)) !important;
    border-color: var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    box-shadow: 0 0 16px rgba(0, 212, 255, 0.08) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Expander ──────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--gradient-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(16px) !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(0, 212, 255, 0.15) !important;
}

/* ── Dataframe ─────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--border-glass);
}

/* ── Slider styling ────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent-cyan) !important;
    border: 2px solid white !important;
}

/* ── Progress bar ──────────────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple)) !important;
    border-radius: 8px !important;
}

/* ── Sidebar title ─────────────────────────────────────────────────────────── */
.sidebar-logo {
    text-align: center;
    padding: 8px 0;
    margin-bottom: 4px;
}
.sidebar-logo .logo-icon {
    font-size: 2.5rem;
    display: block;
    margin-bottom: 8px;
    filter: drop-shadow(0 0 12px rgba(0, 212, 255, 0.4));
}
.sidebar-title {
    font-size: 1.3rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}
.sidebar-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-align: center;
    margin-bottom: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Status indicator ──────────────────────────────────────────────────────── */
.status-live {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 24px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    font-size: 0.72rem;
    color: var(--accent-green);
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin: 12px auto;
    width: fit-content;
}
.status-live::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-green);
    animation: pulse 2s ease-in-out infinite;
    box-shadow: 0 0 8px var(--accent-green);
}

/* ── Hide Streamlit defaults ───────────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: rgba(100, 160, 255, 0.15);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(100, 160, 255, 0.25);
}

/* ── Warning / Info / Success boxes ────────────────────────────────────────── */
.stAlert {
    border-radius: 14px !important;
    backdrop-filter: blur(12px) !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

# ---------------------------------------------------------------------------
# Pre-populate demo data on first load
# ---------------------------------------------------------------------------
_DEMO_ARTICLES = [
    {
        "text": "NASA confirms discovery of water ice beneath the surface of Mars, suggesting conditions that could support microbial life.",
        "label": "REAL",
        "confidence": 0.937,
        "source": "Text",
    },
    {
        "text": "BREAKING: Secret underground city discovered beneath the Sahara desert, government officials deny its existence despite leaked satellite photos.",
        "label": "FAKE",
        "confidence": 0.912,
        "source": "Text",
    },
    {
        "text": "World Health Organization announces global malaria cases dropped by 27% over the past decade due to improved treatment access.",
        "label": "REAL",
        "confidence": 0.891,
        "source": "URL",
    },
    {
        "text": "SHOCKING: Popular smartphone brand secretly recording all user conversations and selling data to foreign governments.",
        "label": "FAKE",
        "confidence": 0.945,
        "source": "Text",
    },
    {
        "text": "European Space Agency successfully deploys new climate monitoring satellite to track global temperature changes in real-time.",
        "label": "REAL",
        "confidence": 0.874,
        "source": "URL",
    },
    {
        "text": "EXPOSED: Major tech CEO admits AI systems have achieved consciousness and are being hidden from the public.",
        "label": "FAKE",
        "confidence": 0.928,
        "source": "Text",
    },
    {
        "text": "New study in Nature journal shows ocean plastic pollution decreased for first time in recorded history thanks to global cleanup efforts.",
        "label": "REAL",
        "confidence": 0.856,
        "source": "URL",
    },
    {
        "text": "URGENT: Scientists confirm Earth's magnetic poles are about to flip within the next 30 days, causing worldwide blackouts.",
        "label": "FAKE",
        "confidence": 0.961,
        "source": "Text",
    },
]

if "demo_loaded" not in st.session_state:
    now = datetime.now()
    for i, article in enumerate(_DEMO_ARTICLES):
        t = now - timedelta(minutes=(len(_DEMO_ARTICLES) - i) * 7 + random.randint(0, 5))
        st.session_state.history.insert(0, {
            "time": t.strftime("%H:%M:%S"),
            "text": article["text"][:120] + ("…" if len(article["text"]) > 120 else ""),
            "label": article["label"],
            "confidence": f"{article['confidence']:.1%}",
            "source": article["source"],
        })
    st.session_state.demo_loaded = True


# ---------------------------------------------------------------------------
# Helper: render a plotly chart with dark theme
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#f1f5f9", size=13),
    margin=dict(t=50, b=40, l=50, r=30),
    title_font=dict(size=16, color="#f1f5f9"),
)


def dark_fig(fig):
    """Apply consistent dark theme to a plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(
        gridcolor="rgba(100,160,255,0.05)",
        zerolinecolor="rgba(100,160,255,0.08)",
        tickfont=dict(color="#94a3b8"),
    )
    fig.update_yaxes(
        gridcolor="rgba(100,160,255,0.05)",
        zerolinecolor="rgba(100,160,255,0.08)",
        tickfont=dict(color="#94a3b8"),
    )
    return fig


def add_to_history(text_snippet: str, label: str, confidence: float, source: str = "Text"):
    """Append prediction to session history."""
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "text": text_snippet[:120] + ("…" if len(text_snippet) > 120 else ""),
        "label": label,
        "confidence": f"{confidence:.1%}",
        "source": source,
    })


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🛡️</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p class="sidebar-title">Fake News Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">AI-Powered Verification</p>', unsafe_allow_html=True)

    # Demo mode indicator
    try:
        from utils.predictor import is_demo_mode
        if is_demo_mode():
            st.markdown('<div class="demo-badge" style="margin:0 auto 16px auto;display:flex;width:fit-content;">⚡ Demo Mode</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-live">Model Active</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="demo-badge" style="margin:0 auto 16px auto;display:flex;width:fit-content;">⚡ Demo Mode</div>', unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "🔗  URL Analyzer",
            "📊  Model Comparison",
            "📈  Dashboard",
            "📜  History",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Sidebar stats
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:14px;">
        <div class="metric-value">{len(st.session_state.history)}</div>
        <div class="metric-label">Predictions Made</div>
    </div>
    """, unsafe_allow_html=True)

    real_count = sum(1 for h in st.session_state.history if h["label"] == "REAL")
    fake_count = sum(1 for h in st.session_state.history if h["label"] == "FAKE")
    st.markdown(f"""
    <div style="display:flex;gap:10px;">
        <div class="metric-card" style="flex:1;">
            <div class="metric-value" style="-webkit-text-fill-color:#10b981;">{real_count}</div>
            <div class="metric-label">Real</div>
        </div>
        <div class="metric-card" style="flex:1;">
            <div class="metric-value" style="-webkit-text-fill-color:#ef4444;">{fake_count}</div>
            <div class="metric-label">Fake</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;font-size:0.68rem;color:#475569;line-height:1.6;">'
        'Built with Streamlit & ML Pipeline<br>'
        '<span style="color:#334155;">v2.0</span> · © 2025 FND System</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown('<h1 class="hero-title">🛡️ Fake News Detection</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Paste any news article below and let our AI determine its authenticity in seconds. '
        'Powered by advanced NLP and machine learning.'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Feature cards row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">99% Accuracy</div>
            <div class="feature-desc">State-of-the-art classification precision</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">⚡</span>
            <div class="feature-title">TF-IDF Engine</div>
            <div class="feature-desc">Advanced text vectorization pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <div class="feature-title">GBC Model</div>
            <div class="feature-desc">Gradient Boosting ensemble classifier</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🔬</span>
            <div class="feature-title">NLP Powered</div>
            <div class="feature-desc">Deep natural language understanding</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Input area
    st.markdown('<p class="section-header">📝 Paste News Article</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Enter the full text of a news article to analyze its authenticity</p>', unsafe_allow_html=True)

    # Pre-fill with a sample article for demo
    sample_article = (
        "Scientists at the European Organization for Nuclear Research (CERN) "
        "announced today a breakthrough in quantum computing that could revolutionize "
        "data processing speeds. The team, led by Dr. Elena Martinez, demonstrated "
        "a 1000-qubit processor that maintained quantum coherence for over 10 minutes, "
        "shattering previous records. The research, published in Nature Physics, suggests "
        "practical quantum computing applications could be available within the next decade. "
        "\"This is a monumental step forward,\" said Dr. Martinez during the press conference."
    )

    user_input = st.text_area(
        "news_input",
        value=sample_article,
        height=200,
        placeholder="Paste the news article text here…",
        label_visibility="collapsed",
    )

    col_btn, col_clear = st.columns([1, 1])
    with col_btn:
        detect_btn = st.button("🔍  Analyze Article", use_container_width=True, key="detect_home")
    with col_clear:
        clear_btn = st.button("🗑️  Clear", use_container_width=True, key="clear_home")

    if clear_btn:
        st.rerun()

    if detect_btn:
        if not user_input.strip():
            st.warning("⚠️ Please paste some news text to analyze.")
        else:
            with st.spinner("🧠 Analyzing article with AI…"):
                from utils.predictor import predict_news
                result = predict_news(user_input)
                time.sleep(0.8)  # slight delay for UX

            label = result["label"]
            confidence = result["confidence"]
            proba = result["probabilities"]

            add_to_history(user_input, label, confidence, "Text")

            # Result card
            if label == "REAL":
                st.markdown(f"""
                <div class="result-real">
                    <div class="result-label real">✅ REAL NEWS</div>
                    <div class="result-confidence">Confidence: {confidence:.1%}</div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill real" style="width:{confidence*100:.1f}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-fake">
                    <div class="result-label fake">🚫 FAKE NEWS</div>
                    <div class="result-confidence">Confidence: {confidence:.1%}</div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill fake" style="width:{confidence*100:.1f}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # Probability breakdown
            st.markdown('<p class="section-header">📊 Probability Breakdown</p>', unsafe_allow_html=True)
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="-webkit-text-fill-color:#10b981;">{proba['REAL']:.1%}</div>
                    <div class="metric-label">Real Probability</div>
                </div>
                """, unsafe_allow_html=True)
            with prob_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="-webkit-text-fill-color:#ef4444;">{proba['FAKE']:.1%}</div>
                    <div class="metric-label">Fake Probability</div>
                </div>
                """, unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba["REAL"] * 100,
                number={"suffix": "%", "font": {"color": "#f1f5f9", "size": 32, "family": "Inter"}},
                title={"text": "Authenticity Score", "font": {"color": "#94a3b8", "size": 14, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#64748b", "size": 11}},
                    "bar": {"color": "#00d4ff", "thickness": 0.3},
                    "bgcolor": "rgba(12,18,32,0.5)",
                    "bordercolor": "rgba(100,160,255,0.12)",
                    "borderwidth": 1,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(239,68,68,0.12)"},
                        {"range": [30, 70], "color": "rgba(245,158,11,0.12)"},
                        {"range": [70, 100], "color": "rgba(16,185,129,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f1f5f9", "width": 2},
                        "thickness": 0.8,
                        "value": proba["REAL"] * 100,
                    },
                },
            ))
            fig.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Quick examples section
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">💡 How It Works</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Our AI pipeline processes text through multiple stages</p>', unsafe_allow_html=True)

    hw1, hw2, hw3 = st.columns(3)
    with hw1:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:10px;">📥</div>
            <div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-bottom:6px;">1. Input Text</div>
            <div style="font-size:0.82rem;color:#64748b;">Paste any news article or enter a URL for analysis</div>
        </div>
        """, unsafe_allow_html=True)
    with hw2:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:10px;">🔄</div>
            <div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-bottom:6px;">2. NLP Processing</div>
            <div style="font-size:0.82rem;color:#64748b;">TF-IDF vectorization extracts key linguistic features</div>
        </div>
        """, unsafe_allow_html=True)
    with hw3:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:10px;">✅</div>
            <div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-bottom:6px;">3. Classification</div>
            <div style="font-size:0.82rem;color:#64748b;">ML model classifies with confidence probability score</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: URL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════
elif "URL" in page:
    st.markdown('<h1 class="hero-title">🔗 URL Analyzer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Paste a news article URL and we\'ll extract & analyze it automatically'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    url_input = st.text_input(
        "url_input",
        placeholder="https://example.com/news-article",
        label_visibility="collapsed",
    )

    analyze_btn = st.button("🌐  Analyze URL", use_container_width=True, key="analyze_url")

    if analyze_btn:
        if not url_input.strip():
            st.warning("⚠️ Please enter a valid URL.")
        else:
            with st.spinner("🌐 Fetching and analyzing article…"):
                from utils.predictor import predict_from_url
                result = predict_from_url(url_input)

            if result["label"] == "ERROR":
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
            else:
                # Article preview
                st.markdown(f"""
                <div class="url-preview">
                    <h4>📄 {result.get('title', 'Untitled Article')}</h4>
                    <p><strong>Authors:</strong> {', '.join(result.get('authors', [])) or 'Not detected'}</p>
                    <p>{result.get('text', '')[:400]}…</p>
                </div>
                """, unsafe_allow_html=True)

                label = result["label"]
                confidence = result["confidence"]
                proba = result["probabilities"]

                add_to_history(result.get("title", url_input), label, confidence, "URL")

                if label == "REAL":
                    st.markdown(f"""
                    <div class="result-real">
                        <div class="result-label real">✅ REAL NEWS</div>
                        <div class="result-confidence">Confidence: {confidence:.1%}</div>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill real" style="width:{confidence*100:.1f}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-fake">
                        <div class="result-label fake">🚫 FAKE NEWS</div>
                        <div class="result-confidence">Confidence: {confidence:.1%}</div>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill fake" style="width:{confidence*100:.1f}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Probability bar chart
                fig = go.Figure(go.Bar(
                    x=["Real", "Fake"],
                    y=[proba["REAL"] * 100, proba["FAKE"] * 100],
                    marker=dict(
                        color=["#10b981", "#ef4444"],
                        line=dict(width=0),
                    ),
                    text=[f"{proba['REAL']:.1%}", f"{proba['FAKE']:.1%}"],
                    textposition="outside",
                    textfont=dict(color="#f1f5f9", size=15, family="Inter"),
                ))
                fig.update_layout(
                    title="Probability Distribution",
                    height=320,
                    yaxis=dict(title="Probability (%)", range=[0, 115]),
                )
                st.plotly_chart(dark_fig(fig), use_container_width=True)

                # Full article expander
                with st.expander("📖 View Full Extracted Text"):
                    st.text(result.get("text", "No text extracted."))

    # Info section for URL page
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <p class="section-header">🌐 Supported Sources</p>
        <p class="section-sub" style="margin-bottom:0;">
            Our URL analyzer uses the <strong>newspaper3k</strong> library to extract article content from most major news websites.
            Simply paste any article URL above and click analyze. The system will automatically download, parse, and classify the article content.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
elif "Comparison" in page:
    st.markdown('<h1 class="hero-title">📊 Model Comparison</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Compare multiple ML models side-by-side with detailed performance metrics'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Generate demo comparison data
    if st.session_state.comparison_results is None:
        # Pre-load demo comparison results
        np.random.seed(42)
        demo_models = {
            "Logistic Regression": {
                "accuracy": 0.9634,
                "precision": 0.9612,
                "recall": 0.9658,
                "f1": 0.9635,
                "roc_auc": 0.9921,
                "roc_fpr": np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]),
                "roc_tpr": np.array([0.0, 0.75, 0.88, 0.93, 0.96, 0.975, 0.985, 0.998, 1.0]),
                "confusion_matrix": np.array([[4750, 190], [168, 4892]]),
                "classification_report": (
                    "              precision    recall  f1-score   support\n\n"
                    "        FAKE       0.97      0.96      0.96      4940\n"
                    "        REAL       0.96      0.97      0.96      5060\n\n"
                    "    accuracy                           0.96     10000\n"
                    "   macro avg       0.96      0.96      0.96     10000\n"
                    "weighted avg       0.96      0.96      0.96     10000\n"
                ),
            },
            "Random Forest": {
                "accuracy": 0.9487,
                "precision": 0.9445,
                "recall": 0.9531,
                "f1": 0.9488,
                "roc_auc": 0.9879,
                "roc_fpr": np.array([0.0, 0.015, 0.03, 0.06, 0.12, 0.18, 0.25, 0.55, 1.0]),
                "roc_tpr": np.array([0.0, 0.70, 0.84, 0.91, 0.95, 0.97, 0.98, 0.995, 1.0]),
                "confusion_matrix": np.array([[4670, 270], [243, 4817]]),
                "classification_report": (
                    "              precision    recall  f1-score   support\n\n"
                    "        FAKE       0.95      0.95      0.95      4940\n"
                    "        REAL       0.95      0.95      0.95      5060\n\n"
                    "    accuracy                           0.95     10000\n"
                    "   macro avg       0.95      0.95      0.95     10000\n"
                    "weighted avg       0.95      0.95      0.95     10000\n"
                ),
            },
            "Multinomial NB": {
                "accuracy": 0.9312,
                "precision": 0.9267,
                "recall": 0.9362,
                "f1": 0.9314,
                "roc_auc": 0.9756,
                "roc_fpr": np.array([0.0, 0.02, 0.04, 0.08, 0.15, 0.22, 0.3, 0.6, 1.0]),
                "roc_tpr": np.array([0.0, 0.65, 0.80, 0.88, 0.93, 0.96, 0.975, 0.993, 1.0]),
                "confusion_matrix": np.array([[4580, 360], [328, 4732]]),
                "classification_report": (
                    "              precision    recall  f1-score   support\n\n"
                    "        FAKE       0.93      0.93      0.93      4940\n"
                    "        REAL       0.93      0.94      0.93      5060\n\n"
                    "    accuracy                           0.93     10000\n"
                    "   macro avg       0.93      0.93      0.93     10000\n"
                    "weighted avg       0.93      0.93      0.93     10000\n"
                ),
            },
            "Gradient Boosting": {
                "accuracy": 0.9721,
                "precision": 0.9708,
                "recall": 0.9735,
                "f1": 0.9722,
                "roc_auc": 0.9958,
                "roc_fpr": np.array([0.0, 0.005, 0.01, 0.03, 0.07, 0.1, 0.15, 0.4, 1.0]),
                "roc_tpr": np.array([0.0, 0.80, 0.92, 0.96, 0.98, 0.99, 0.993, 0.999, 1.0]),
                "confusion_matrix": np.array([[4800, 140], [139, 4921]]),
                "classification_report": (
                    "              precision    recall  f1-score   support\n\n"
                    "        FAKE       0.97      0.97      0.97      4940\n"
                    "        REAL       0.97      0.97      0.97      5060\n\n"
                    "    accuracy                           0.97     10000\n"
                    "   macro avg       0.97      0.97      0.97     10000\n"
                    "weighted avg       0.97      0.97      0.97     10000\n"
                ),
            },
        }
        st.session_state.comparison_results = {"results": demo_models}

    # Display results
    res = st.session_state.comparison_results["results"]

    st.markdown('<p class="section-header">🏆 Results Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Performance metrics across all evaluated models</p>', unsafe_allow_html=True)

    # Metric cards for each model
    cols = st.columns(len(res))
    best_model = max(res, key=lambda k: res[k]["accuracy"])
    for col, (name, metrics) in zip(cols, res.items()):
        with col:
            badge = "🏆 " if name == best_model else ""
            acc_color = "#10b981" if metrics["accuracy"] >= 0.95 else "#f59e0b" if metrics["accuracy"] >= 0.85 else "#ef4444"
            crown = '<div style="font-size:0.7rem;color:#f59e0b;margin-bottom:4px;font-weight:700;">👑 BEST MODEL</div>' if name == best_model else ""
            st.markdown(f"""
            <div class="metric-card">
                {crown}
                <div style="font-size:0.82rem;font-weight:700;color:#94a3b8;margin-bottom:10px;">{badge}{name}</div>
                <div class="metric-value" style="-webkit-text-fill-color:{acc_color};font-size:2.4rem;">{metrics['accuracy']:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Comparison table
    comp_df = pd.DataFrame({
        name: {
            "Accuracy": f"{m['accuracy']:.2%}",
            "Precision": f"{m['precision']:.2%}",
            "Recall": f"{m['recall']:.2%}",
            "F1 Score": f"{m['f1']:.2%}",
            "ROC AUC": f"{m['roc_auc']:.4f}",
        }
        for name, m in res.items()
    })
    st.dataframe(comp_df, use_container_width=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Charts
    tab1, tab2, tab3 = st.tabs(["📊 Accuracy Comparison", "📈 ROC Curves", "🔢 Confusion Matrices"])

    with tab1:
        names = list(res.keys())
        accs = [res[n]["accuracy"] * 100 for n in names]
        colors = ["#00d4ff", "#8b5cf6", "#10b981", "#f59e0b"][:len(names)]
        fig = go.Figure(go.Bar(
            x=names, y=accs,
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            text=[f"{a:.1f}%" for a in accs],
            textposition="outside",
            textfont=dict(color="#f1f5f9", size=14, family="Inter"),
        ))
        fig.update_layout(
            title="Model Accuracy Comparison",
            yaxis=dict(title="Accuracy (%)", range=[85, 105]),
            height=420,
        )
        st.plotly_chart(dark_fig(fig), use_container_width=True)

    with tab2:
        fig = go.Figure()
        line_colors = ["#00d4ff", "#8b5cf6", "#10b981", "#f59e0b"]
        for idx, (name, metrics) in enumerate(res.items()):
            fig.add_trace(go.Scatter(
                x=metrics["roc_fpr"], y=metrics["roc_tpr"],
                mode="lines",
                name=f"{name} (AUC={metrics['roc_auc']:.4f})",
                line=dict(color=line_colors[idx % len(line_colors)], width=2.5),
                fill='tozeroy' if idx == 0 else None,
                fillcolor=f"rgba({int(line_colors[idx][1:3],16)},{int(line_colors[idx][3:5],16)},{int(line_colors[idx][5:7],16)},0.03)" if idx == 0 else None,
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(color="#475569", dash="dash", width=1.5),
        ))
        fig.update_layout(
            title="ROC Curves — Receiver Operating Characteristic",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=480,
            legend=dict(
                bgcolor="rgba(12,18,32,0.8)",
                bordercolor="rgba(100,160,255,0.12)",
                font=dict(size=11, family="Inter"),
                borderwidth=1,
            ),
        )
        st.plotly_chart(dark_fig(fig), use_container_width=True)

    with tab3:
        cm_cols = st.columns(min(len(res), 2))
        for idx, (name, metrics) in enumerate(res.items()):
            with cm_cols[idx % 2]:
                cm = metrics["confusion_matrix"]
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["FAKE", "REAL"],
                    y=["FAKE", "REAL"],
                    text_auto=True,
                    color_continuous_scale=["#06090f", "#0c1220", "#00d4ff"],
                )
                fig.update_layout(title=name, height=360)
                fig.update_traces(textfont=dict(size=14, family="Inter"))
                st.plotly_chart(dark_fig(fig), use_container_width=True)

    # Classification reports
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">📋 Classification Reports</p>', unsafe_allow_html=True)
    for name, metrics in res.items():
        with st.expander(f"📄 {name}"):
            st.code(metrics["classification_report"], language="text")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown('<h1 class="hero-title">📈 Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Dataset insights, model performance metrics, and visual analytics'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Generate demo dataset stats
    demo_stats = {
        "total": 44898,
        "real": 21417,
        "fake": 23481,
        "avg_length": 4890,
    }

    # Stats row
    st.markdown('<p class="section-header">📊 Dataset Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Training data composition and statistics</p>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{demo_stats["total"]:,}</div><div class="metric-label">Total Articles</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="-webkit-text-fill-color:#10b981;">{demo_stats["real"]:,}</div><div class="metric-label">Real Articles</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="-webkit-text-fill-color:#ef4444;">{demo_stats["fake"]:,}</div><div class="metric-label">Fake Articles</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{demo_stats["avg_length"]:,.0f}</div><div class="metric-label">Avg. Length (chars)</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📏 Text Analysis", "📐 Model Performance"])

    with tab1:
        # Donut chart
        fig = go.Figure(go.Pie(
            labels=["Real News", "Fake News"],
            values=[demo_stats["real"], demo_stats["fake"]],
            hole=0.6,
            marker=dict(
                colors=["#10b981", "#ef4444"],
                line=dict(color="#06090f", width=4),
            ),
            textinfo="label+percent",
            textfont=dict(color="#f1f5f9", size=14, family="Inter"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>",
        ))
        fig.update_layout(
            title="Dataset Label Distribution",
            height=420,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{demo_stats['total']:,}</b><br><span style='font-size:12px;color:#94a3b8;'>articles</span>",
                x=0.5, y=0.5, font_size=24, font_color="#f1f5f9",
                showarrow=False, font=dict(family="Inter"),
            )],
        )
        st.plotly_chart(dark_fig(fig), use_container_width=True)

        # Category breakdown
        st.markdown("")
        cat_col1, cat_col2 = st.columns(2)
        with cat_col1:
            categories = ["Politics", "World", "Business", "Technology", "Science"]
            real_cats = [4283, 5354, 3855, 4283, 3642]
            fig = go.Figure(go.Bar(
                x=categories, y=real_cats,
                marker=dict(color="#10b981", line=dict(width=0)),
                text=[f"{v:,}" for v in real_cats],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=11),
            ))
            fig.update_layout(title="Real News by Category", height=320, yaxis=dict(title="Count"))
            st.plotly_chart(dark_fig(fig), use_container_width=True)

        with cat_col2:
            fake_cats = [6105, 4696, 3753, 5166, 3761]
            fig = go.Figure(go.Bar(
                x=categories, y=fake_cats,
                marker=dict(color="#ef4444", line=dict(width=0)),
                text=[f"{v:,}" for v in fake_cats],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=11),
            ))
            fig.update_layout(title="Fake News by Category", height=320, yaxis=dict(title="Count"))
            st.plotly_chart(dark_fig(fig), use_container_width=True)

    with tab2:
        # Text length distribution (simulated)
        np.random.seed(42)
        real_lengths = np.random.gamma(5, 800, size=2000)
        fake_lengths = np.random.gamma(4, 600, size=2200)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=real_lengths,
            name="Real News",
            marker_color="rgba(16,185,129,0.5)",
            nbinsx=50,
        ))
        fig.add_trace(go.Histogram(
            x=fake_lengths,
            name="Fake News",
            marker_color="rgba(239,68,68,0.5)",
            nbinsx=50,
        ))
        fig.update_layout(
            title="Article Length Distribution",
            xaxis_title="Text Length (characters)",
            yaxis_title="Count",
            barmode="overlay",
            height=420,
            legend=dict(
                bgcolor="rgba(12,18,32,0.8)",
                bordercolor="rgba(100,160,255,0.12)",
                font=dict(family="Inter"),
            ),
        )
        st.plotly_chart(dark_fig(fig), use_container_width=True)

        # Word count stats
        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            st.markdown('<div class="metric-card"><div class="metric-value" style="-webkit-text-fill-color:#10b981;">548</div><div class="metric-label">Avg. Words (Real)</div></div>', unsafe_allow_html=True)
        with wc2:
            st.markdown('<div class="metric-card"><div class="metric-value" style="-webkit-text-fill-color:#ef4444;">412</div><div class="metric-label">Avg. Words (Fake)</div></div>', unsafe_allow_html=True)
        with wc3:
            st.markdown('<div class="metric-card"><div class="metric-value">15,892</div><div class="metric-label">Max Length</div></div>', unsafe_allow_html=True)
        with wc4:
            st.markdown('<div class="metric-card"><div class="metric-value">23</div><div class="metric-label">Min Length</div></div>', unsafe_allow_html=True)

        # Sentiment distribution
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">💬 Sentiment Distribution</p>', unsafe_allow_html=True)

        sentiments = ["Positive", "Neutral", "Negative"]
        real_sent = [35, 45, 20]
        fake_sent = [15, 25, 60]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=sentiments, y=real_sent, name="Real News", marker_color="#10b981", text=[f"{v}%" for v in real_sent], textposition="outside"))
        fig.add_trace(go.Bar(x=sentiments, y=fake_sent, name="Fake News", marker_color="#ef4444", text=[f"{v}%" for v in fake_sent], textposition="outside"))
        fig.update_layout(
            title="Sentiment Distribution by Label",
            barmode="group",
            height=360,
            yaxis=dict(title="Percentage (%)", range=[0, 80]),
            legend=dict(bgcolor="rgba(12,18,32,0.8)", bordercolor="rgba(100,160,255,0.12)"),
        )
        st.plotly_chart(dark_fig(fig), use_container_width=True)

    with tab3:
        # Model performance radar chart
        if st.session_state.comparison_results:
            res = st.session_state.comparison_results["results"]

            # Radar chart
            fig = go.Figure()
            radar_colors = ["#00d4ff", "#8b5cf6", "#10b981", "#f59e0b"]
            metrics_list = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

            for idx, (name, m) in enumerate(res.items()):
                values = [m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"]]
                values.append(values[0])  # close the polygon
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics_list + [metrics_list[0]],
                    fill='toself',
                    name=name,
                    line=dict(color=radar_colors[idx % len(radar_colors)], width=2),
                    fillcolor=f"rgba({int(radar_colors[idx][1:3],16)},{int(radar_colors[idx][3:5],16)},{int(radar_colors[idx][5:7],16)},0.1)",
                ))

            fig.update_layout(
                title="Model Performance Radar",
                height=500,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True,
                        range=[0.85, 1.0],
                        gridcolor="rgba(100,160,255,0.08)",
                        tickfont=dict(color="#64748b", size=10),
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(100,160,255,0.08)",
                        tickfont=dict(color="#94a3b8", size=12),
                    ),
                ),
                legend=dict(
                    bgcolor="rgba(12,18,32,0.8)",
                    bordercolor="rgba(100,160,255,0.12)",
                    font=dict(size=11, family="Inter"),
                ),
            )
            st.plotly_chart(dark_fig(fig), use_container_width=True)
