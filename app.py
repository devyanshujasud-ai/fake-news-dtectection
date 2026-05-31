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
