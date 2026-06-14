#!/usr/bin/env python3
"""
VTS V20 — Daily Monitoring Dashboard  (v2)
Run with:  streamlit run VTS_V18/monitoring/dashboard.py
"""

import json, os
from datetime import datetime, date, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
MON_OUT   = os.path.join(ROOT, "data", "monitoring")
RPT_OUT   = os.path.join(ROOT, "data", "v20_report")
SIGNAL_F  = os.path.join(MON_OUT, "latest_signal.json")
HISTORY_F = os.path.join(MON_OUT, "signal_history.csv")

# ── Regime config ──────────────────────────────────────────────────────────────
REGIME_COLOR = {"Calm":"#2E7D32","Defensive":"#E65C00","High-Stress":"#BF360C","Extreme":"#B71C1C"}
REGIME_BG    = {"Calm":"#E8F5E9","Defensive":"#FFF3E0","High-Stress":"#FBE9E7","Extreme":"#FFEBEE"}
THRESHOLDS   = [("Calm", 0, 60), ("Defensive", 60, 80),
                ("High-Stress", 80, 95), ("Extreme", 95, 101)]
INSTRUMENTS  = {"Calm":{"TV":"SVXY","DR":"QLD","STR":"SPY"},
                "Defensive":{"TV":"IAU","DR":"XLU","STR":"IYR"},
                "High-Stress":{"TV":"IAU","DR":"Cash","STR":"Cash"},
                "Extreme":{"TV":"Cash","DR":"Cash","STR":"Cash"}}

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VTS V20", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
    .block-container{padding-top:.8rem;padding-bottom:.5rem}
    .alert-trade{background:#B71C1C;color:white;padding:14px 20px;border-radius:8px;
        font-size:1.35rem;font-weight:700;text-align:center;margin-bottom:10px}
    .alert-ok{background:#1B5E20;color:white;padding:10px 20px;border-radius:8px;
        font-size:1.05rem;text-align:center;margin-bottom:10px}
    .kpi-box{background:#F5F7FA;border-radius:8px;padding:10px 14px;text-align:center}
    .kpi-val{font-size:1.6rem;font-weight:700;line-height:1.1}
    .kpi-lbl{font-size:.78rem;color:#666;margin-top:2px}
    .holding-row{display:flex;justify-content:space-between;
        padding:5px 0;border-bottom:1px solid #EEE}
    .stop-ok{color:#2E7D32;font-weight:600}
    .stop-warn{color:#E65C00;font-weight:600}
    .stop-active{color:#B71C1C;font-weight:700}
    h3{font-size:1rem!important;margin-bottom:4px!important;color:#333}
</style>""", unsafe_allow_html=True)

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_signal():
    if not os.path.exists(SIGNAL_F): return None
    with open(SIGNAL_F) as f: return json.load(f)

@st.cache_data(ttl=300)
def load_history():
    if not os.path.exists(HISTORY_F): return pd.DataFrame()
    return pd.read_csv(HISTORY_F, parse_dates=["date"])

@st.cache_data(ttl=3600)
def load_perf():
    """Returns dict with ytd, mtd, cagr, sharpe, max_dd, spy_ytd."""
    out = {}
    # Full metrics
    mf = os.path.join(RPT_OUT, "00_full_metrics.json")
    if os.path.exists(mf):
        m = json.load(open(mf))
        out["cagr"]   = m.get("cagr")
        out["sharpe"] = m.get("sharpe")
        out["max_dd"] = m.get("max_dd")
        out["report_generated"] = m.get("report_generated")
        out["data_through"]     = m.get("data_through")
    # YTD from annual returns
    af = os.path.join(RPT_OUT, "01_annual_returns.csv")
    if os.path.exists(af):
        ann = pd.read_csv(af)
        cy = date.today().year
        row = ann[ann["year"] == cy]
        if not row.empty:
            out["ytd"]     = float(row["vts_v20"].iloc[0])
            out["spy_ytd"] = float(row["spy"].iloc[0]) if "spy" in row.columns else None
    # MTD from monthly heatmap
    hf = os.path.join(RPT_OUT, "02_monthly_returns_heatmap.csv")
    if os.path.exists(hf):
        month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"Mei":5,"Jun":6,
                     "Jul":7,"Aug":8,"Sep":9,"Okt":10,"Nov":11,"Dec":12,
                     "May":5,"Oct":10}
        df_h = pd.read_csv(hf, index_col=0)
        cy, cm = date.today().year, date.today().month
        if cy in df_h.index:
            for col, mo in month_map.items():
                if mo == cm and col in df_h.columns:
                    v = df_h.loc[cy, col]
                    if pd.notna(v): out["mtd"] = float(v)
    return out

@st.cache_data(ttl=3600)
def load_equity():
    path = os.path.join(RPT_OUT, "02_monthly_returns_heatmap.csv")
    if not os.path.exists(path): return None
    month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"Mei":5,"Jun":6,
                 "Jul":7,"Aug":8,"Sep":9,"Okt":10,"Nov":11,"Dec":12,
                 "May":5,"Oct":10}
    df = pd.read_csv(path, index_col=0)
    rows = []
    for yr in df.index:
        for col, mo in month_map.items():
            if col not in df.columns: continue
            v = df.loc[yr, col]
            if pd.notna(v) and v != "":
                rows.append({"year":int(yr),"month":mo,"ret":float(v)/100})
    if not rows: return None
    s = pd.DataFrame(rows).sort_values(["year","month"])
    s["date"] = pd.to_datetime(
        s["year"].astype(str)+"-"+s["month"].astype(str).str.zfill(2)+"-01")
    s = s.set_index("date")["ret"]
    s.index = s.index + pd.offsets.MonthEnd(0)
    return (1 + s).cumprod() * 100_000

# ── Load ───────────────────────────────────────────────────────────────────────
sig     = load_signal()
history = load_history()
perf    = load_perf()
equity  = load_equity()

# ── Header ─────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([6, 1])
c1.markdown("### 📊 VTS V20 — Signal Dashboard")
if c2.button("🔄 Refresh"):
    st.cache_data.clear(); st.rerun()

if sig is None:
    st.error("No signal file. Run `python VTS_V18/monitoring/run_daily_signal.py`")
    st.stop()

gen_dt  = datetime.fromisoformat(sig["generated_at"])
if gen_dt.tzinfo is None:                       # older signals were tz-naive (UTC)
    gen_dt = gen_dt.replace(tzinfo=timezone.utc)
age_hrs = (datetime.now(timezone.utc) - gen_dt).total_seconds() / 3600
fresh   = "🟢 Fresh" if age_hrs < 6 else ("🟡 Today" if age_hrs < 20 else "🔴 Stale — rerun signal")
st.caption(f"Based on US close **{sig['as_of']}** · "
           f"Computed {gen_dt.strftime('%H:%M')} UTC · {fresh} · "
           f"1-day lag: positions effective tomorrow")

# ── Trade alert banner ─────────────────────────────────────────────────────────
if sig["trade_needed_tomorrow"]:
    td = sig["trade_details"]
    trades = "  |  ".join(f"{c['sleeve']}: {c['sell']} → {c['buy']}"
                          for c in td["changes"])
    st.markdown(
        f'<div class="alert-trade">🚨 TRADE TOMORROW · {td["reason"].upper()}<br>'
        f'<span style="font-size:.95rem;font-weight:400">{trades}</span><br>'
        f'<span style="font-size:.8rem;font-weight:400">Execute at EU open · 09:00 CET</span>'
        f'</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert-ok">✅  No trade tomorrow — hold current positions</div>',
                unsafe_allow_html=True)

# ── KPI strip ──────────────────────────────────────────────────────────────────
def kpi(label, value, color="#1A3D6E", suffix=""):
    return (f'<div class="kpi-box">'
            f'<div class="kpi-val" style="color:{color}">{value}{suffix}</div>'
            f'<div class="kpi-lbl">{label}</div></div>')

def pct_color(v):
    if v is None: return "#888"
    return "#2E7D32" if v >= 0 else "#B71C1C"

def fmt_pct(v, decimals=1):
    if v is None: return "—"
    return f"{v:+.{decimals}f}%"

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.markdown(kpi("YTD Return", fmt_pct(perf.get("ytd")),
                color=pct_color(perf.get("ytd"))), unsafe_allow_html=True)
k2.markdown(kpi("MTD Return", fmt_pct(perf.get("mtd")),
                color=pct_color(perf.get("mtd"))), unsafe_allow_html=True)
k3.markdown(kpi("SPY YTD",   fmt_pct(perf.get("spy_ytd")),
                color=pct_color(perf.get("spy_ytd"))), unsafe_allow_html=True)
k4.markdown(kpi("CAGR (full)", f"{perf.get('cagr','—')}%", color="#1A3D6E"), unsafe_allow_html=True)
k5.markdown(kpi("Sharpe", f"{perf.get('sharpe','—')}", color="#1A3D6E"), unsafe_allow_html=True)
k6.markdown(kpi("Max DD", f"{perf.get('max_dd','—')}%", color="#B71C1C"), unsafe_allow_html=True)

# Backtest provenance — these KPIs are static backtest figures, not live values.
_rg = perf.get("report_generated")
_dt = perf.get("data_through")
if _rg or _dt:
    st.markdown(
        f"<div style='font-size:.72rem;color:#999;text-align:center;margin-top:4px'>"
        f"Performance figures above are from the backtest"
        + (f" through <b>{_dt}</b>" if _dt else "")
        + (f" · regenerated {_rg}" if _rg else "")
        + " — not live. The barometer, regime and signal below are live.</div>",
        unsafe_allow_html=True)

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

# ── Main columns ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 2.2])

with left:
    baro    = sig["barometer"]
    regime  = sig["regime"]
    rc      = REGIME_COLOR[regime]

    # ── Gauge ────────────────────────────────────────────────────────────────
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=baro,
        number={"font":{"size":40,"color":rc},"valueformat":".1f"},
        gauge={
            "axis":{
                "range":[0,100],
                "tickvals":[0,60,80,95,100],
                "ticktext":["0","60","80","95","100"],
                "tickwidth":1,"tickcolor":"#555","tickfont":{"size":10},
            },
            "bar":{"color":rc,"thickness":0.2},
            "bgcolor":"white",
            "steps":[
                {"range":[0, 60],"color":"#E8F5E9"},
                {"range":[60,80],"color":"#FFF3E0"},
                {"range":[80,95],"color":"#FBE9E7"},
                {"range":[95,100],"color":"#FFCDD2"},
            ],
            "threshold":{"line":{"color":rc,"width":4},"thickness":0.85,"value":baro},
        },
        domain={"x":[0,1],"y":[0,1]},
    ))
    fig_g.update_layout(height=210, margin=dict(l=10,r=10,t=15,b=5),
                        paper_bgcolor="white")
    st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})

    # Regime badge
    st.markdown(
        f'<div style="text-align:center;margin-top:-10px;margin-bottom:6px">'
        f'<span style="background:{REGIME_BG[regime]};color:{rc};padding:5px 18px;'
        f'border-radius:20px;font-weight:700;font-size:1.05rem">{regime.upper()}</span>'
        f'</div>', unsafe_allow_html=True)

    if sig.get("regime_changed"):
        st.warning(f"⚡ Changed from **{sig['previous_regime']}** today")

    # ── Distance to thresholds ────────────────────────────────────────────────
    next_up   = next(((lbl, lo) for lbl,lo,hi in THRESHOLDS if lo > baro), None)
    next_down = next(((lbl, hi) for lbl,lo,hi in reversed(THRESHOLDS) if hi <= baro), None)

    if next_up:
        gap = next_up[1] - baro
        st.markdown(f'<div style="text-align:center;font-size:.85rem;color:#666;margin-bottom:4px">'
                    f'▲ <b>{gap:.1f} pts</b> to <b>{next_up[0]}</b> '
                    f'(threshold: {next_up[1]})</div>', unsafe_allow_html=True)
    if next_down:
        gap = baro - next_down[1]
        st.markdown(f'<div style="text-align:center;font-size:.85rem;color:#666;margin-bottom:8px">'
                    f'▼ <b>{gap:.1f} pts</b> below <b>{next_down[0]}</b></div>',
                    unsafe_allow_html=True)

    # Days in current regime
    if not history.empty:
        reg_col = history.sort_values("date", ascending=False)
        last_change_idx = reg_col[reg_col["regime"] != regime].index
        if len(last_change_idx):
            days_in = (reg_col.index.get_loc(last_change_idx[0]))
        else:
            days_in = len(reg_col)
        st.markdown(f'<div style="text-align:center;font-size:.82rem;color:#888;margin-bottom:10px">'
                    f'In {regime} for <b>{days_in}</b> trading day{"s" if days_in!=1 else ""}'
                    f'</div>', unsafe_allow_html=True)

    st.divider()

    # ── Current holdings ─────────────────────────────────────────────────────
    st.markdown("**Current Holdings**")
    pos = sig["positions"]
    for sl, lbl in [("TV","Tactical Vol"),("DR","Defensive Rot."),("STR","Strategic Ret.")]:
        ticker = pos[sl]
        color  = "#B71C1C" if "Cash" in ticker else "#1A3D6E"
        st.markdown(
            f'<div class="holding-row">'
            f'<span style="color:#555;font-size:.9rem"><b>{sl}</b> {lbl}</span>'
            f'<span style="color:{color};font-weight:700">{ticker}</span>'
            f'</div>', unsafe_allow_html=True)

    # Next regime → what would change
    if next_up:
        next_pos = INSTRUMENTS.get(next_up[0], {})
        changes  = [f"{sl}: {pos[sl]}→{next_pos.get(sl,'?')}"
                    for sl in ["TV","DR","STR"] if next_pos.get(sl) != pos[sl]]
        if changes:
            st.markdown(
                f'<div style="font-size:.78rem;color:#888;margin-top:6px">'
                f'If barometer reaches {next_up[1]}: '
                + " · ".join(changes) + "</div>", unsafe_allow_html=True)

    st.divider()

    # ── SVXY stop ────────────────────────────────────────────────────────────
    if pos["TV"] in ("SVXY", "Cash (stop)"):
        st.markdown("**SVXY Trailing Stop**")
        dd      = sig["svxy_dd_from_peak_pct"]
        peak    = sig["svxy_peak_price"]
        price   = sig["svxy_price"]
        active  = sig["svxy_stop_active"]
        pct_to  = min(max(-dd / 12, 0), 1.0)

        if active:
            cls, lbl = "stop-active", "🛑 STOP ACTIVE"
        elif dd < -8:
            cls, lbl = "stop-warn", "⚠️ WARNING"
        else:
            cls, lbl = "stop-ok", "✅ OK"

        col_a, col_b = st.columns(2)
        col_a.metric("Price", f"${price:.2f}")
        col_b.metric("Peak", f"${peak:.2f}")

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:.85rem;margin-bottom:4px">'
            f'<span class="{cls}">{lbl}</span>'
            f'<span style="color:#555">DD: <b>{dd:+.1f}%</b> / stop at −12%</span>'
            f'</div>', unsafe_allow_html=True)

        # Progress bar: green → red
        bar_color = "#2E7D32" if pct_to < 0.5 else ("#E65C00" if pct_to < 0.85 else "#B71C1C")
        st.markdown(
            f'<div style="background:#EEE;border-radius:4px;height:10px">'
            f'<div style="background:{bar_color};width:{pct_to*100:.0f}%;'
            f'height:10px;border-radius:4px"></div></div>'
            f'<div style="font-size:.75rem;color:#888;text-align:right">'
            f'{pct_to*100:.0f}% toward stop</div>', unsafe_allow_html=True)

        st.divider()

    # ── VIX ──────────────────────────────────────────────────────────────────
    st.markdown("**Market Conditions**")
    col_a, col_b = st.columns(2)
    col_a.metric("VIX", f"{sig['vix']:.1f}")
    col_b.metric("VIX pct (252d)", f"{sig['vix_252d_percentile']:.0f}%")

# ── Right panel ────────────────────────────────────────────────────────────────
with right:

    # ── Barometer 60 days ─────────────────────────────────────────────────────
    st.markdown("**Barometer — Last 60 Days**")
    baro_60 = pd.DataFrame(sig.get("barometer_history_60d", []))
    if not baro_60.empty:
        baro_60["date"] = pd.to_datetime(baro_60["date"])

        # Colour config per regime
        _seg_color = {
            "Calm":        "#2E7D32",
            "Defensive":   "#E65C00",
            "High-Stress": "#BF360C",
            "Extreme":     "#B71C1C",
        }
        _bg_color = {
            "Calm":        "rgba(46,125,50,0.07)",
            "Defensive":   "rgba(230,81,0,0.09)",
            "High-Stress": "rgba(191,54,12,0.09)",
            "Extreme":     "rgba(183,28,28,0.12)",
        }

        # ── Split into per-regime segments so the line changes colour ──────────
        # Build list of (start_idx, end_idx, regime_name)
        segments = []
        cur_regime = baro_60["regime"].iloc[0]
        seg_start  = 0
        for i in range(1, len(baro_60)):
            if baro_60["regime"].iloc[i] != cur_regime:
                segments.append((seg_start, i, cur_regime))
                seg_start  = i
                cur_regime = baro_60["regime"].iloc[i]
        segments.append((seg_start, len(baro_60) - 1, cur_regime))

        fig_b = go.Figure()

        # Horizontal regime bands
        for lo, hi, lbl, col in [
            (0,  60, "Calm",        "rgba(46,125,50,0.06)"),
            (60, 80, "Defensive",   "rgba(230,81,0,0.08)"),
            (80, 95, "High-Stress", "rgba(191,54,12,0.08)"),
            (95,100, "Extreme",     "rgba(183,28,28,0.11)"),
        ]:
            fig_b.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0,
                            annotation_text=lbl, annotation_position="right",
                            annotation_font_size=9, annotation_font_color="#888")

        # Threshold dashed lines
        for thresh, col in [(60,"#F9A825"),(80,"#E65C00"),(95,"#B71C1C")]:
            fig_b.add_hline(y=thresh, line_dash="dot", line_color=col, line_width=1.2)

        # Coloured line segments (one trace per segment, bridge with next point)
        shown = set()
        for s_start, s_end, seg_reg in segments:
            # Include the next point to avoid gaps between segments
            end_idx   = min(s_end + 1, len(baro_60) - 1)
            seg_df    = baro_60.iloc[s_start : end_idx + 1]
            col       = _seg_color.get(seg_reg, "#555")
            show_leg  = seg_reg not in shown
            shown.add(seg_reg)
            fig_b.add_trace(go.Scatter(
                x=seg_df["date"], y=seg_df["barometer"],
                mode="lines",
                line={"color": col, "width": 2.5},
                fill="tozeroy",
                fillcolor=_bg_color.get(seg_reg, "rgba(0,0,0,0.05)"),
                name=seg_reg,
                legendgroup=seg_reg,
                showlegend=show_leg,
            ))

        # Vertical regime-change markers
        for i in range(1, len(baro_60)):
            if baro_60["regime"].iloc[i] != baro_60["regime"].iloc[i - 1]:
                chg_date = baro_60["date"].iloc[i]
                new_reg  = baro_60["regime"].iloc[i]
                old_reg  = baro_60["regime"].iloc[i - 1]
                col      = _seg_color.get(new_reg, "#555")
                fig_b.add_vline(
                    x=chg_date,
                    line_dash="dash", line_color=col, line_width=1.5,
                )
                fig_b.add_annotation(
                    x=chg_date,
                    y=103,
                    text=f"→{new_reg[:3]}",
                    showarrow=False,
                    font={"size": 8, "color": col, "family": "monospace"},
                    xanchor="left",
                )

        # Today's value dot
        fig_b.add_trace(go.Scatter(
            x=[baro_60["date"].iloc[-1]], y=[baro_60["barometer"].iloc[-1]],
            mode="markers+text",
            marker={"size": 11, "color": rc, "line": {"color": "white", "width": 2}},
            text=[f" {baro:.1f}"], textposition="middle right",
            textfont={"color": rc, "size": 11, "family": "monospace"},
            name="Today", showlegend=False,
        ))

        fig_b.update_layout(
            height=255, margin=dict(l=0, r=95, t=10, b=25),
            xaxis={"showgrid": False, "tickformat": "%b %d"},
            yaxis={"range": [0, 107], "showgrid": True, "gridcolor": "#EEE",
                   "tickvals": [0, 60, 80, 95, 100]},
            paper_bgcolor="white", plot_bgcolor="white",
            legend={"orientation": "h", "y": 1.12, "font": {"size": 10}},
        )
        st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})

    # ── Strategy breakdown — last 60 days ────────────────────────────────────
    eq60_raw = sig.get("equity_60d", [])
    attr_raw = sig.get("period_attribution", [])

    if eq60_raw:
        eq60 = pd.DataFrame(eq60_raw)
        eq60["date"] = pd.to_datetime(eq60["date"])

        end_val  = eq60["portfolio_value"].iloc[-1]
        gain_pct = (end_val / 100_000 - 1) * 100
        c60      = "#2E7D32" if gain_pct >= 0 else "#B71C1C"

        st.markdown(
            f'**Strategy Breakdown — Last 60 Days** '
            f'<span style="font-size:.88rem;color:{c60};font-weight:600">'
            f'Portfolio: {gain_pct:+.1f}%  ·  $100K → ${end_val/1e3:.1f}K'
            f'</span>', unsafe_allow_html=True)

        # ── Subplot: sleeve lines (top) + barometer (bottom) ─────────────────
        fig_s = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.04,
        )

        # Regime background bands on barometer subplot
        for lo, hi, lbl, col in [
            (0,  60, "Calm",        "rgba(46,125,50,0.07)"),
            (60, 80, "Defensive",   "rgba(230,81,0,0.09)"),
            (80, 95, "High-Stress", "rgba(191,54,12,0.09)"),
            (95,100, "Extreme",     "rgba(183,28,28,0.12)"),
        ]:
            fig_s.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0,
                            row=2, col=1)
        for thresh, col in [(60,"#F9A825"),(80,"#E65C00"),(95,"#B71C1C")]:
            fig_s.add_hline(y=thresh, line_dash="dot", line_color=col,
                            line_width=1, row=2, col=1)

        # Barometer line (coloured by regime, same logic as barometer chart)
        if not baro_60.empty:
            _seg_b = []; cr = baro_60["regime"].iloc[0]; cs = 0
            for i in range(1, len(baro_60)):
                if baro_60["regime"].iloc[i] != cr:
                    _seg_b.append((cs, i, cr)); cs = i; cr = baro_60["regime"].iloc[i]
            _seg_b.append((cs, len(baro_60)-1, cr))
            shown_b = set()
            for s, e, sr in _seg_b:
                ei = min(e+1, len(baro_60)-1)
                seg = baro_60.iloc[s:ei+1]
                col = _seg_color.get(sr, "#555")
                fig_s.add_trace(go.Scatter(
                    x=seg["date"], y=seg["barometer"],
                    mode="lines", line={"color": col, "width": 1.8},
                    name=sr, legendgroup=sr,
                    showlegend=(sr not in shown_b),
                ), row=2, col=1)
                shown_b.add(sr)

        # Vertical regime-change markers on both subplots
        if not baro_60.empty:
            for i in range(1, len(baro_60)):
                if baro_60["regime"].iloc[i] != baro_60["regime"].iloc[i-1]:
                    chg_d = baro_60["date"].iloc[i]
                    col   = _seg_color.get(baro_60["regime"].iloc[i], "#555")
                    for row in [1, 2]:
                        fig_s.add_vline(x=chg_d, line_dash="dash",
                                        line_color=col, line_width=1.2, row=row, col=1)
                    fig_s.add_annotation(
                        x=chg_d, y=103, xref="x2", yref="y2",
                        text=f"→{baro_60['regime'].iloc[i][:3]}",
                        showarrow=False,
                        font={"size": 7, "color": col, "family": "monospace"},
                        xanchor="left",
                    )

        # Sleeve lines — TV, DR, STR each indexed to 100
        SLEEVE_STYLE = {
            "TV":  {"color": "#1A3D6E", "dash": "solid",  "width": 2.2, "label": "TV (Tactical Vol)"},
            "DR":  {"color": "#E65C00", "dash": "solid",  "width": 2.2, "label": "DR (Defensive Rot.)"},
            "STR": {"color": "#6A1B9A", "dash": "solid",  "width": 2.2, "label": "STR (Strategic Ret.)"},
        }
        for sleeve, style in SLEEVE_STYLE.items():
            if sleeve not in eq60.columns: continue
            last_v = eq60[sleeve].iloc[-1]
            last_g = last_v - 100
            fig_s.add_trace(go.Scatter(
                x=eq60["date"], y=eq60[sleeve],
                mode="lines",
                line={"color": style["color"], "width": style["width"],
                      "dash": style["dash"]},
                name=style["label"],
                hovertemplate=f"<b>{sleeve}</b>: %{{y:.1f}}<extra>%{{x|%d %b}}</extra>",
            ), row=1, col=1)
            # End label
            fig_s.add_annotation(
                x=eq60["date"].iloc[-1], y=last_v, xref="x", yref="y",
                text=f" {last_g:+.1f}",
                showarrow=False, xanchor="left",
                font={"color": style["color"], "size": 10, "family": "monospace"},
            )

        # 100-baseline
        fig_s.add_hline(y=100, line_dash="dot", line_color="#CCC",
                        line_width=1, row=1, col=1)

        fig_s.update_layout(
            height=360,
            margin=dict(l=0, r=75, t=10, b=5),
            paper_bgcolor="white", plot_bgcolor="white",
            legend={"orientation": "h", "y": 1.07, "font": {"size": 10}},
        )
        fig_s.update_xaxes(showgrid=False, tickformat="%b %d", row=1, col=1)
        fig_s.update_xaxes(showgrid=False, tickformat="%b %d", row=2, col=1)
        fig_s.update_yaxes(showgrid=True, gridcolor="#EEE",
                           ticksuffix="", tickprefix="",
                           title_text="Indexed (100 = start)", title_font_size=9,
                           row=1, col=1)
        fig_s.update_yaxes(range=[0, 107], showgrid=False,
                           tickvals=[0, 60, 80, 95, 100],
                           title_text="Barometer", title_font_size=9,
                           row=2, col=1)

        st.plotly_chart(fig_s, use_container_width=True,
                        config={"displayModeBar": False})

        # ── Period attribution table ──────────────────────────────────────────
        if attr_raw:
            attr = pd.DataFrame(attr_raw)
            # Emoji regime icons
            _ri = {"Calm":"🟢","Defensive":"🟡","High-Stress":"🟠","Extreme":"🔴"}
            attr["Regime"] = attr["regime"].apply(
                lambda r: f'{_ri.get(r,"⚪")} {r}')
            attr["Period"] = attr.apply(
                lambda row: f'{pd.to_datetime(row["date_from"]).strftime("%d %b")} – '
                            f'{pd.to_datetime(row["date_to"]).strftime("%d %b")}', axis=1)
            attr["Days"]  = attr["days"]
            for col, key in [("TV %","TV_pct"),("DR %","DR_pct"),
                             ("STR %","STR_pct"),("Port %","port_pct")]:
                attr[col] = attr[key].apply(lambda x: f"{x:+.1f}%")

            st.dataframe(
                attr[["Regime","Period","Days","TV %","DR %","STR %","Port %"]],
                hide_index=True, use_container_width=True,
                column_config={
                    "Regime": st.column_config.TextColumn(width="medium"),
                    "Period": st.column_config.TextColumn(width="medium"),
                    "Days":   st.column_config.NumberColumn(width="small"),
                    "TV %":   st.column_config.TextColumn(width="small"),
                    "DR %":   st.column_config.TextColumn(width="small"),
                    "STR %":  st.column_config.TextColumn(width="small"),
                    "Port %": st.column_config.TextColumn(width="small"),
                }
            )

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.markdown("**Portfolio Value — Full Period** *(starting $100K)*")
    if equity is not None:
        final = equity.iloc[-1]
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines", name="VTS V20",
            line={"color":"#1A3D6E","width":2},
            fill="tozeroy", fillcolor="rgba(26,61,110,0.06)",
            hovertemplate="<b>$%{y:,.0f}</b><extra>%{x|%b %Y}</extra>",
        ))
        fig_e.add_hline(y=100_000, line_dash="dot", line_color="#BBB", line_width=1)
        fig_e.add_annotation(
            x=equity.index[-1], y=final,
            text=f" ${final/1e6:.2f}M",
            showarrow=False, xanchor="left",
            font={"color":"#1A3D6E","size":11,"family":"monospace"},
        )
        fig_e.update_layout(
            height=240, margin=dict(l=0,r=80,t=5,b=25),
            xaxis={"showgrid":False},
            yaxis={"showgrid":True,"gridcolor":"#EEE","tickformat":"$,.0f"},
            paper_bgcolor="white", plot_bgcolor="white",
            legend={"orientation":"h","y":1.08},
        )
        st.plotly_chart(fig_e, use_container_width=True, config={"displayModeBar":False})

    # ── Signal history — last 3 months ───────────────────────────────────────
    st.markdown("**Signal History — Last 3 Months**")
    hist3m_raw = sig.get("signal_history_3m", [])
    if hist3m_raw:
        hist3m = pd.DataFrame(hist3m_raw)
        hist3m["date"] = pd.to_datetime(hist3m["date"])
        hist3m = hist3m.sort_values("date", ascending=False).reset_index(drop=True)

        # Styled columns
        hist3m["Date"]    = hist3m["date"].dt.strftime("%d %b %Y")
        hist3m["Baro"]    = hist3m["barometer"].round(1)
        hist3m["VIX"]     = hist3m["vix"].round(1)

        def _regime_icon(r):
            icons = {"Calm":"🟢","Defensive":"🟡","High-Stress":"🟠","Extreme":"🔴"}
            return f'{icons.get(r,"⚪")} {r}'
        hist3m["Regime"]  = hist3m["regime"].apply(_regime_icon)
        hist3m["Chg"]     = hist3m["regime_changed"].apply(lambda x: "⚡" if x else "")
        hist3m["SVXY $"]  = hist3m["svxy_price"].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) and x else "—")
        hist3m["DD%"]     = hist3m["svxy_dd_pct"].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) and x is not None else "—")

        st.dataframe(
            hist3m[["Date","Baro","VIX","Regime","Chg","SVXY $","DD%"]],
            hide_index=True,
            use_container_width=True,
            height=420,          # scrollable — ~15 rows visible, rest scroll
            column_config={
                "Date":   st.column_config.TextColumn("Date",   width="small"),
                "Baro":   st.column_config.NumberColumn("Baro", format="%.1f", width="small"),
                "VIX":    st.column_config.NumberColumn("VIX",  format="%.1f", width="small"),
                "Regime": st.column_config.TextColumn("Regime", width="medium"),
                "Chg":    st.column_config.TextColumn("Chg",    width="small"),
                "SVXY $": st.column_config.TextColumn("SVXY",   width="small"),
                "DD%":    st.column_config.TextColumn("DD%",    width="small"),
            }
        )
    elif not history.empty:
        # Fallback: live CSV data (fills in over time)
        show = history.sort_values("date", ascending=False).copy()
        show["Date"]    = show["date"].dt.strftime("%d %b %Y")
        show["Baro"]    = show["barometer"].round(1)
        show["Regime"]  = show["regime"]
        show["Chg"]     = show["regime_changed"].apply(lambda x: "⚡" if x else "")
        show["SVXY DD"] = show["svxy_dd_pct"].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
        st.dataframe(
            show[["Date","Baro","Regime","Chg","SVXY DD"]],
            hide_index=True, use_container_width=True, height=420)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"VTS V20 · Signal from US closing prices (VIX + HYG) · "
    f"1-day execution lag · SVXY trailing stop −12% · "
    f"Auto-refresh every 5 min · {datetime.now().strftime('%H:%M:%S')}"
)
