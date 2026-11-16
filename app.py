"""ERMM-Chart designer: interactive design, monitoring and diagnosis of the
EWMA robust Max-M control chart.

Run locally with   streamlit run app.py
"""
from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ermm import __version__, config
from ermm.charts import HALF_NORMAL_MEAN, L_from_ucl, ermm_path, mewma_path, ucl_from_L
from ermm.design import Shift, calibrate, evaluate_arl, spec_driven_shifts, uniform_shift
from ermm.diagnosis import (characteristic_deviations, component_summary, first_signal,
                            property_comparison, rank_actions, signal_table)
from ermm.estimators import classical_estimate, reweighted_fast_mcd

HERE = Path(__file__).parent
DATA = HERE / "data"
BLUE, RED, GREY, ORANGE, GREEN = "#1f4e79", "#c0392b", "#7f8c8d", "#d68910", "#1e8449"

st.set_page_config(page_title="ERMM-Chart designer", page_icon="🏭", layout="wide")


def md_table(d: pd.DataFrame) -> str:
    """Markdown table without the optional 'tabulate' dependency."""
    cols = [str(c) for c in d.columns]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# cached computations
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    return pd.read_csv(DATA / "opc_synthetic.csv")


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data(show_spinner=False)
def phase1_estimates(X: np.ndarray, alpha: float, n_starts: int, q: float, seed: int):
    t0 = time.perf_counter()
    rob = reweighted_fast_mcd(X, alpha, n_starts, q, np.random.default_rng(seed))
    t_rob = time.perf_counter() - t0
    return classical_estimate(X), rob, t_rob


def lookup_limit(chart, method, p, N, lam, arl0):
    """Limit from the session (calibrated here) or from the shipped table."""
    key = (chart, method, p, N, round(lam, 4), arl0)
    if key in st.session_state.limits:
        return st.session_state.limits[key]
    tab = load_table("calibration_constants.csv")
    m = tab[(tab.chart == chart) & (tab.method == method) & (tab.p == p) & (tab.N == N)
            & (np.isclose(tab.lam, lam)) & (np.isclose(tab.arl0, arl0))]
    if len(m):
        r = m.iloc[0]
        return {"limit": float(r.limit), "L": None if pd.isna(r.L) else float(r.L),
                "n_rep": int(r.n_rep), "source": r.source}
    return None


def run_calibration(chart, method, p, N, lam, arl0, n_rep, seed, label):
    bar = st.progress(0.0, text=f"Calibrating {label}: {n_rep:,} replications")
    t0 = time.perf_counter()
    cal = calibrate(chart, method, p, N, lam, arl0, n_rep, config.HORIZON, seed,
                    st.session_state.alpha, st.session_state.n_starts,
                    st.session_state.q, progress=lambda f: bar.progress(min(f, 1.0)))
    bar.empty()
    res = {"limit": cal.limit, "L": cal.L, "n_rep": n_rep,
           "source": f"calibrated in this session ({n_rep:,} replications, seed {seed}, "
                     f"{time.perf_counter() - t0:.0f} s)"}
    st.session_state.limits[(chart, method, p, N, round(lam, 4), arl0)] = res
    return res


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------
for k, v in {"limits": {}, "arl_results": None, "alpha": config.MCD_ALPHA,
             "n_starts": config.MCD_N_STARTS, "q": config.REWEIGHT_QUANTILE}.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------------------
# sidebar: data and design settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏭 ERMM-Chart designer")
    st.caption(f"EWMA robust Max-M chart · v{__version__}")
    source = st.radio("Data", ["Demo: synthetic OPC finish mill", "Upload CSV"], index=0)
    if source.startswith("Demo"):
        df = load_demo()
        st.caption("Synthetic data generated from the published summary statistics "
                   "of the case study; not plant records.")
    else:
        up = st.file_uploader("CSV with one row per production day", type="csv")
        if up is None:
            st.info("Upload a CSV file or use the demo data.")
            st.stop()
        df = pd.read_csv(up)
    cols = list(df.columns)
    date_col = st.selectbox("Date column", cols,
                            index=cols.index("Date") if "Date" in cols else 0)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    demo_chars = [c for c in ["Blaine", "Residue45", "FreeLime", "LOI", "SO3"] if c in numeric]
    chars = st.multiselect("Quality characteristics (p)", numeric,
                           default=demo_chars or numeric[: min(5, len(numeric))])
    props = st.multiselect("Product properties (optional)",
                           [c for c in numeric if c not in chars],
                           default=[c for c in numeric if c not in chars
                                    and ("Strength" in c or "Setting" in c)])
    if len(chars) < 2:
        st.warning("Select at least two characteristics.")
        st.stop()
    if "Phase" in cols and set(df["Phase"].astype(str)) >= {"I", "II"}:
        n_default = int((df["Phase"].astype(str) == "I").sum())
    else:
        n_default = int(0.6 * len(df))
    N = st.number_input("Reference (Phase I) days N", min_value=len(chars) + 10,
                        max_value=len(df) - 2, value=n_default, step=1)
    st.markdown("---")
    lam = st.select_slider("Weighting constant λ", options=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                                            0.6, 0.7, 0.8, 0.9, 1.0],
                           value=config.LAMBDA)
    arl0 = st.number_input("Target in-control ARL", min_value=50.0, max_value=2000.0,
                           value=config.ARL0, step=10.0)
    seed = st.number_input("Random seed", value=config.SEED, step=1)
    with st.expander("Robust estimator"):
        st.session_state.alpha = st.slider("MCD α (h ≈ αN)", 0.55, 1.0, config.MCD_ALPHA, 0.05)
        st.session_state.n_starts = st.number_input("Fast-MCD random starts", 50, 3000,
                                                    config.MCD_N_STARTS, 50)
        st.session_state.q = st.slider("Reweighting quantile", 0.90, 0.995,
                                       config.REWEIGHT_QUANTILE, 0.005)
    gap_days = st.number_input("Flag days after a stop longer than (days)", 1, 60,
                               config.GAP_DAYS)

# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------
data = df[[date_col] + chars + props].copy()
data[date_col] = pd.to_datetime(data[date_col])
data = data.sort_values(date_col).reset_index(drop=True)
if data[chars].isna().any().any():
    st.error("Missing values in the selected characteristics. Remove or impute these "
             "days before monitoring.")
    st.stop()
X = data[chars].to_numpy(float)
dates = data[date_col]
p, N = len(chars), int(N)
X1, X2 = X[:N], X[N:]
clas, rob, t_rob = phase1_estimates(X1, st.session_state.alpha, int(st.session_state.n_starts),
                                    st.session_state.q, int(seed))

if source.startswith("Demo"):
    spec0 = load_table("specifications_demo.csv")
    ocap0 = load_table("ocap_finish_mill.csv")
else:
    spec0 = pd.DataFrame({"characteristic": chars, "unit": "", "LSL": np.nan,
                          "target": np.nan, "USL": np.nan, "source": ""})
    ocap0 = pd.DataFrame({"characteristic": chars, "process_unit": "", "first_check": ""})
spec0 = spec0[spec0.characteristic.isin(chars)]
ocap0 = ocap0[ocap0.characteristic.isin(chars)]

st.title("ERMM-Chart designer")
st.markdown("Design, run and interpret an **EWMA robust Max-M chart** that monitors the "
            "mean vector and the covariance matrix of daily quality characteristics in "
            "one statistic. Work from left to right through the tabs.")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Characteristics p", p)
c2.metric("Reference days N", N)
c3.metric("Monitored days", len(X2))
c4.metric("Days excluded by Fast-MCD", int((rob.weights == 0).sum()))
lim_now = lookup_limit("maxm", "fastmcd", p, N, lam, arl0)
c5.metric("ERMM limit (UCL)", f"{lim_now['limit']:.4f}" if lim_now else "not calibrated")

tabs = st.tabs(["① Product & process", "② Reference period", "③ Chart design",
                "④ Monitoring", "⑤ Diagnosis & action"])

# ---------------------------------------------------------------------------
# ① product specifications and process map
# ---------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([1, 1])
    with right:
        st.subheader("Design specifications")
        st.caption("Limits that the product design imposes on each characteristic. "
                   "They define the shifts the chart must detect (tab ③).")
        spec = st.data_editor(
            spec0, hide_index=True, width="stretch", key="spec_editor", num_rows="fixed",
            column_config={"LSL": st.column_config.NumberColumn("LSL", format="%.4g"),
                           "target": st.column_config.NumberColumn("target", format="%.4g"),
                           "USL": st.column_config.NumberColumn("USL", format="%.4g"),
                           "source": st.column_config.TextColumn("source", width="large")})
        st.subheader("Out-of-control action plan")
        st.caption("Process unit and first check for each characteristic (tab ⑤).")
        ocap = st.data_editor(ocap0, hide_index=True, width="stretch",
                              key="ocap_editor", num_rows="fixed")
    with left:
        st.subheader("Daily quality characteristics")
        fig = make_subplots(rows=p, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                            subplot_titles=chars)
        for j, c in enumerate(chars):
            fig.add_trace(go.Scatter(x=dates, y=X[:, j], mode="lines+markers",
                                     marker=dict(size=3), line=dict(width=1, color=BLUE),
                                     name=c, showlegend=False), row=j + 1, col=1)
            s = spec[spec.characteristic == c]
            for lim_name in ["LSL", "USL"]:
                if len(s) and pd.notna(s[lim_name].iloc[0]):
                    fig.add_hline(y=float(s[lim_name].iloc[0]), line=dict(color=RED, dash="dot"),
                                  row=j + 1, col=1)
            fig.add_vline(x=dates.iloc[N - 1], line=dict(color=GREY, dash="dash"),
                          row=j + 1, col=1)
        fig.update_layout(height=150 * p + 60, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Dashed line: end of the reference period. Dotted red lines: "
                   "specification limits.")

# ---------------------------------------------------------------------------
# ② reference period
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Phase I estimates: classical and reweighted Fast-MCD")
    sd_c, sd_r = np.sqrt(np.diag(clas.cov)), np.sqrt(np.diag(rob.cov))
    est_tab = pd.DataFrame({"characteristic": chars, "mean (classical)": clas.center,
                            "mean (Fast-MCD)": rob.center, "SD (classical)": sd_c,
                            "SD (Fast-MCD)": sd_r,
                            "SD change (%)": 100 * (sd_r / sd_c - 1)})
    a, b = st.columns([3, 2])
    a.dataframe(est_tab.style.format(precision=3), hide_index=True, width="stretch")
    det_ratio = np.linalg.det(rob.cov) / np.linalg.det(clas.cov)
    b.markdown(f"- Subset size h = **{rob.h}** of N = {N} (α = {st.session_state.alpha})\n"
               f"- Days with weight 0: **{int((rob.weights == 0).sum())}**\n"
               f"- det(Σ̂ robust) / det(S) = **{det_ratio:.2f}**\n"
               f"- Fast-MCD computing time: **{1000 * t_rob:.0f} ms**")
    z = (X1 - rob.center) / sd_r
    out_idx = np.where(rob.weights == 0)[0]
    if out_idx.size:
        worst = np.abs(z[out_idx]).argmax(axis=1)
        excl = pd.DataFrame({"date": dates.iloc[out_idx].dt.date.values,
                             "most deviating": [chars[k] for k in worst],
                             "value": X1[out_idx, worst],
                             "robust z": z[out_idx, worst],
                             "robust distance²": rob.d2[out_idx]})
        st.markdown("**Days excluded from the reference estimates**")
        st.dataframe(excl.style.format({"value": "{:.3f}", "robust z": "{:+.1f}",
                                        "robust distance²": "{:.1f}"}),
                     hide_index=True, width="stretch")
    from scipy import stats as _st
    cut = _st.chi2.ppf(st.session_state.q, p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates.iloc[:N], y=rob.d2, mode="markers",
                             marker=dict(color=np.where(rob.weights == 0, RED, BLUE), size=7),
                             name="robust distance²"))
    fig.add_hline(y=cut, line=dict(color=RED, dash="dash"),
                  annotation_text=f"χ²({p}; {st.session_state.q})")
    fig.update_layout(height=320, yaxis_title="squared robust distance",
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# ③ chart design
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Control limit for the available reference size")
    st.caption("The constant L depends on N and p, so it is calibrated by simulation "
               "(Algorithm 1) for the reference period at hand.")
    col_a, col_b = st.columns([2, 3])
    with col_a:
        info = lookup_limit("maxm", "fastmcd", p, N, lam, arl0)
        if info:
            st.success(f"UCL = **{info['limit']:.4f}**, L = **{L_from_ucl(info['limit'], lam):.3f}**  \n"
                       f"Source: {info['source']}")
        else:
            st.warning(f"No limit for N = {N}, p = {p}, λ = {lam}, ARL₀ = {arl0:g} yet.")
        n_cal = st.select_slider("Replications", [500, 1000, 2000, 5000, 10000],
                                 value=config.CAL_N_REP_QUICK)
        if st.button("Calibrate ERMM limit", type="primary"):
            info = run_calibration("maxm", "fastmcd", p, N, lam, arl0, n_cal, int(seed), "ERMM")
            st.rerun()
        with st.expander("Calibrate comparison charts (EMM, MEWMA)"):
            for chart, method, label in [("maxm", "classical", "EMM"),
                                         ("mewma", "classical", "MEWMA classical"),
                                         ("mewma", "fastmcd", "MEWMA Fast-MCD")]:
                cur = lookup_limit(chart, method, p, N, lam, arl0)
                txt = f"{label}: " + (f"{cur['limit']:.4f}" if cur else "not calibrated")
                if st.button(f"Calibrate {label}", key=f"cal_{label}"):
                    run_calibration(chart, method, p, N, lam, arl0, n_cal, int(seed), label)
                    st.rerun()
                st.caption(txt)
    with col_b:
        tab = load_table("calibration_constants.csv")
        show = tab[(tab.p == p) & (tab.N == N)] if len(tab[(tab.p == p) & (tab.N == N)]) else tab
        st.markdown("**Shipped constants**")
        st.dataframe(show[["chart", "method", "p", "N", "lam", "arl0", "limit", "L",
                           "n_rep", "source"]].style.format({"limit": "{:.4f}", "L": "{:.3f}", "lam": "{:g}",
                                                    "arl0": "{:g}"}),
                     hide_index=True, width="stretch", height=240)

    st.markdown("---")
    st.subheader("From product specifications to the shifts that matter")
    tnc = st.select_slider("Nonconforming fraction that defines a critical shift",
                           [0.001, 0.0027, 0.005, 0.01, 0.02, 0.05],
                           value=config.TARGET_NONCONFORMING)
    sp = spec.set_index("characteristic").reindex(chars)
    crit = pd.DataFrame(spec_driven_shifts(chars, rob.center, sd_r,
                                           sp["LSL"].to_numpy(float), sp["USL"].to_numpy(float), tnc))
    if crit.empty:
        st.info("Enter at least one specification limit in tab ① to derive critical shifts.")
    else:
        crit["status"] = np.where(crit["critical_mean_shift_sd"] > 0, "monitor with the chart",
                                  "act on process or spec")
        st.dataframe(crit.drop(columns=["direction"]).rename(columns={
            "distance_sd": "margin to limit (robust SD)",
            "nonconforming_now": "nonconforming now",
            "critical_mean_shift_sd": "critical mean shift (SD)",
            "critical_var_factor": "critical variance factor"}).style.format(
            {"margin to limit (robust SD)": "{:.2f}", "nonconforming now": "{:.2%}",
             "critical mean shift (SD)": "{:.2f}", "critical variance factor": "{:.2f}"}),
            hide_index=True, width="stretch")

    st.subheader("Compare weighting constants on these shifts")
    lam_cands = st.multiselect("Candidate λ", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                               default=list(config.LAMBDA_CANDIDATES))
    std_shifts = st.multiselect("Additional shifts of all characteristics (a SD, variance × b)",
                                ["a=0.5, b=1", "a=1, b=1", "a=0, b=1.4", "a=0, b=2", "a=0.6, b=1.6"],
                                default=["a=1, b=1", "a=0, b=1.4"])
    n_arl = st.select_slider("Replications per shift", [200, 500, 1000, 2000], value=config.ARL_N_REP)
    missing = [l for l in lam_cands if lookup_limit("maxm", "fastmcd", p, N, l, arl0) is None]
    if missing:
        st.caption(f"Limits missing for λ = {missing}; they are calibrated first "
                   f"({config.CAL_N_REP_QUICK:,} replications each).")
    if st.button("Evaluate ARL", type="primary", disabled=not lam_cands):
        for l in missing:
            run_calibration("maxm", "fastmcd", p, N, l, arl0, config.CAL_N_REP_QUICK, int(seed),
                            f"ERMM λ={l}")
        shifts = []
        for _, r in crit.iterrows():
            j = chars.index(r["characteristic"])
            if r["critical_mean_shift_sd"] > 0:
                d = np.zeros(p); d[j] = r["direction"] * r["critical_mean_shift_sd"]
                shifts.append(Shift(f"{r['characteristic']} mean → {tnc:.1%} NC", d, np.ones(p)))
            if r["critical_var_factor"] > 1:
                v = np.ones(p); v[j] = r["critical_var_factor"]
                shifts.append(Shift(f"{r['characteristic']} spread → {tnc:.1%} NC", np.zeros(p), v))
        for s in std_shifts:
            a_, b_ = [float(x.split("=")[1]) for x in s.split(", ")]
            shifts.append(uniform_shift(p, a_, b_, s))
        charts = [dict(name=f"λ = {l}", chart="maxm", method="fastmcd", lam=l,
                       limit=lookup_limit("maxm", "fastmcd", p, N, l, arl0)["limit"])
                  for l in lam_cands]
        R = rob.cov / np.outer(sd_r, sd_r)
        bar = st.progress(0.0, text="Evaluating run lengths")
        out = evaluate_arl(charts, shifts, R, N, n_arl, config.HORIZON, int(seed) + 1,
                           st.session_state.alpha, int(st.session_state.n_starts),
                           st.session_state.q, progress=lambda f: bar.progress(min(f, 1.0)))
        bar.empty()
        st.session_state.arl_results = pd.DataFrame(out)
    if st.session_state.arl_results is not None:
        res = st.session_state.arl_results
        fig = go.Figure()
        for name, g in res.groupby("chart", sort=False):
            fig.add_trace(go.Bar(x=g["shift"], y=g["ARL"], name=name,
                                 error_y=dict(type="data", array=1.96 * g["SE"])))
        fig.update_layout(barmode="group", yaxis=dict(type="log", dtick=1, title="ARL (log scale)"),
                          height=380, margin=dict(l=10, r=10, t=20, b=10),
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")
        best = res.loc[res.groupby("shift", sort=False)["ARL"].idxmin(), ["shift", "chart", "ARL"]]
        st.markdown("**Fastest weighting constant per shift**")
        st.dataframe(best.rename(columns={"chart": "best λ"}).style.format({"ARL": "{:.1f}"}),
                     hide_index=True, width="stretch")
        st.download_button("Download ARL table (CSV)", res.to_csv(index=False),
                           "arl_design_study.csv", "text/csv")

# ---------------------------------------------------------------------------
# ④ monitoring
# ---------------------------------------------------------------------------
lim = lookup_limit("maxm", "fastmcd", p, N, lam, arl0)
path = ermm_path(X2, X1[-1], rob.center, rob.cov, lam) if len(X2) else None
with tabs[3]:
    if lim is None:
        st.warning("Calibrate the ERMM limit in tab ③ first.")
    elif path is None:
        st.info("No monitored days after the reference period.")
    else:
        tbl = signal_table(dates.iloc[N:], path["stat"], lim["limit"], gap_days)
        k1 = first_signal(tbl)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signals", f"{int(tbl.signal.sum())} of {len(tbl)}")
        m2.metric("First signal", tbl.date.iloc[k1].strftime("%d %b %Y") if k1 is not None else "none")
        m3.metric("Signals after a stop", int((tbl.signal & tbl.after_stop).sum()))
        m4.metric("UCL", f"{lim['limit']:.4f}")
        overlay = st.multiselect("Overlay comparison charts (if calibrated)",
                                 ["EMM", "MEWMA classical", "MEWMA Fast-MCD"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tbl.date, y=tbl.statistic, mode="lines+markers",
                                 name="ERMM", line=dict(color=BLUE), marker=dict(size=5)))
        sig = tbl[tbl.signal]
        fig.add_trace(go.Scatter(x=sig.date, y=sig.statistic, mode="markers", name="signal",
                                 marker=dict(color=RED, size=9, symbol="circle-open", line=dict(width=2))))
        fig.add_hline(y=lim["limit"], line=dict(color=RED, dash="dash"),
                      annotation_text=f"UCL = {lim['limit']:.4f}")
        for d in tbl.date[tbl.after_stop]:
            fig.add_vline(x=d, line=dict(color=GREY, dash="dot"))
        fig.update_layout(height=380, yaxis_title=f"ERMM statistic (λ = {lam})",
                          margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.caption("Dotted grey lines: first production day after a stop longer than "
                   f"{gap_days} days.")
        for name in overlay:
            chart, method = {"EMM": ("maxm", "classical"),
                             "MEWMA classical": ("mewma", "classical"),
                             "MEWMA Fast-MCD": ("mewma", "fastmcd")}[name]
            li = lookup_limit(chart, method, p, N, lam, arl0)
            if li is None:
                st.warning(f"{name}: calibrate its limit in tab ③.")
                continue
            est = clas if method == "classical" else rob
            stat = (ermm_path(X2, X1[-1], est.center, est.cov, lam)["stat"] if chart == "maxm"
                    else mewma_path(X2, est.center, est.cov, lam))
            t2 = signal_table(dates.iloc[N:], stat, li["limit"], gap_days)
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=t2.date, y=t2.statistic, mode="lines+markers",
                                    name=name, line=dict(color=ORANGE), marker=dict(size=4)))
            f2.add_hline(y=li["limit"], line=dict(color=RED, dash="dash"))
            k2 = first_signal(t2)
            f2.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                             title=f"{name}: {int(t2.signal.sum())} signals, first "
                                   f"{t2.date.iloc[k2].date() if k2 is not None else 'none'}")
            st.plotly_chart(f2, width="stretch")
        st.dataframe(tbl.assign(date=tbl.date.dt.date).style.format({"statistic": "{:.4f}",
                                                                    "limit": "{:.4f}"}),
                     hide_index=True, width="stretch", height=220)
        st.download_button("Download monitoring table (CSV)", tbl.to_csv(index=False),
                           "ermm_monitoring.csv", "text/csv")

# ---------------------------------------------------------------------------
# ⑤ diagnosis and action
# ---------------------------------------------------------------------------
with tabs[4]:
    if lim is None or path is None:
        st.warning("Run the monitoring in tab ④ first.")
    else:
        tbl = signal_table(dates.iloc[N:], path["stat"], lim["limit"], gap_days)
        k1 = first_signal(tbl)
        options = list(tbl.date.dt.date)
        start_date = st.select_slider("Start of the period to diagnose", options=options,
                                      value=options[k1] if k1 is not None else options[0])
        k = options.index(start_date)
        if len(X2) - k < 2:
            st.warning("Choose a start date with at least two monitored days after it.")
            st.stop()
        comp = component_summary(path["U"], path["V"], k)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tbl.date, y=path["U_ewma"], name="EWMA of |U| (location)",
                                 line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=tbl.date, y=path["V_ewma"], name="EWMA of |V| (successive difference)",
                                 line=dict(color=ORANGE)))
        fig.add_hline(y=HALF_NORMAL_MEAN, line=dict(color=GREY, dash="dot"),
                      annotation_text="in-control mean √(2/π)")
        fig.add_vline(x=tbl.date.iloc[k], line=dict(color=RED, dash="dash"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(f"Mean |U| after start (in control {comp['in_control_mean_abs']:.2f})", f"{comp['mean_abs_U']:.2f}")
        d2.metric(f"Mean |V| after start (in control {comp['in_control_mean_abs']:.2f})", f"{comp['mean_abs_V']:.2f}")
        d3.metric("Days with V < 0 (50% in control)", f"{comp['share_V_negative']:.0%}")
        d4.metric("Location-dominated days", f"{comp['share_location_dominant']:.0%}")
        if comp["share_V_negative"] > 0.6:
            st.info("Most successive-difference components are negative: the day-to-day "
                    "variation has become **smaller** than in the reference period.")
        dev = characteristic_deviations(X2, chars, rob.center, rob.cov, k)
        a, b = st.columns([3, 2])
        with a:
            st.markdown("**Characteristics after the start date (relative to the robust reference)**")
            st.dataframe(dev.style.format({"reference_centre": "{:.3f}", "mean_after": "{:.3f}",
                                           "shift_robust_sd": "{:+.2f}", "reference_sd": "{:.3f}",
                                           "sd_after": "{:.3f}", "sd_ratio": "{:.2f}",
                                           "sd_change_pct": "{:+.0f}"}),
                         hide_index=True, width="stretch")
        with b:
            figd = go.Figure(go.Bar(x=dev["shift_robust_sd"], y=dev["characteristic"],
                                    orientation="h", marker_color=[RED if v < 0 else GREEN
                                                                   for v in dev["shift_robust_sd"]]))
            figd.update_layout(height=260, xaxis_title="mean shift (robust SD)",
                               margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(figd, width="stretch")
        th1, th2 = st.columns(2)
        t_mean = th1.slider("Report a level change from (robust SD)", 0.2, 2.0, 0.5, 0.1)
        t_sd = th2.slider("Report a spread change from (%)", 5, 60, 20, 5)
        act = rank_actions(dev, ocap, t_mean, t_sd)
        st.markdown("**Where to look first**")
        if act.empty:
            st.write("No characteristic exceeds the reporting thresholds.")
        else:
            st.dataframe(act.style.format({"shift_robust_sd": "{:+.2f}", "sd_change_pct": "{:+.0f}"}),
                         hide_index=True, width="stretch")
        prop_tab = pd.DataFrame()
        if props:
            ref_mask = np.r_[np.ones(N, bool), np.zeros(len(X2), bool)]
            aft_mask = np.r_[np.zeros(N + k, bool), np.ones(len(X2) - k, bool)]
            prop_tab = property_comparison(data[props], ref_mask, aft_mask)
            st.markdown("**Product properties: reference period vs. period after the start date**")
            st.dataframe(prop_tab.style.format({"mean_ref": "{:.1f}", "sd_ref": "{:.1f}",
                                                "mean_after": "{:.1f}", "sd_after": "{:.1f}",
                                                "welch_p": "{:.3g}"}),
                         hide_index=True, width="stretch")
        rep = io.StringIO()
        rep.write(f"# ERMM-Chart report\n\nGenerated {datetime.now():%Y-%m-%d %H:%M} with "
                  f"ERMM-Chart designer v{__version__}\n\n## Design\n\n"
                  f"- characteristics (p = {p}): {', '.join(chars)}\n- reference days N = {N}, "
                  f"Fast-MCD h = {rob.h}, excluded days = {int((rob.weights == 0).sum())}\n"
                  f"- lambda = {lam}, target ARL0 = {arl0:g}, UCL = {lim['limit']:.4f} "
                  f"(L = {L_from_ucl(lim['limit'], lam):.3f}); {lim['source']}\n\n## Monitoring\n\n"
                  f"- signals: {int(tbl.signal.sum())} of {len(tbl)} days; first signal: "
                  f"{tbl.date.iloc[k1].date() if k1 is not None else 'none'}\n\n## Diagnosis from "
                  f"{start_date}\n\n{md_table(dev)}\n\n")
        if not act.empty:
            rep.write("## Suggested checks\n\n" + md_table(act) + "\n\n")
        if len(prop_tab):
            rep.write("## Product properties\n\n" + md_table(prop_tab) + "\n")
        st.download_button("Download diagnosis report (Markdown)", rep.getvalue(),
                           "ermm_report.md", "text/markdown", type="primary")

st.markdown("---")
st.caption("ERMM-Chart designer · reference implementation of the EWMA robust Max-M chart · "
           "Fast-MCD estimator identical to robustbase::covMcd · MIT licence · "
           "https://github.com/ahsan4th/ERMM-Chart")
