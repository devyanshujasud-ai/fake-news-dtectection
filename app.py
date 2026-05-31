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
