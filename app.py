from __future__ import annotations

import pandas as pd
import streamlit as st
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    PLOTLY_AVAILABLE = False

from msme_platform.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    create_scheme_dataset,
    generate_msme_dataset,
)
from msme_platform.model import (
    generate_local_reasons,
    predict_growth,
    score_msme_dataset,
    train_growth_model,
)
from msme_platform.optimization import (
    build_decision_views,
    build_policy_options,
    optimize_budget_allocation,
)
from msme_platform.simulation import rank_impacts_for_advisory

ACCENT = "#14b8a6"
DANGER = "#ef4444"
WARN = "#f59e0b"
SUCCESS = "#22c55e"
PANEL = "#101827"
MUTED = "#93a4bd"


@st.cache_data(show_spinner=False)
def _get_synthetic_data(n_entries: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    msme_df = generate_msme_dataset(n_entries=n_entries, seed=seed)
    schemes_df = create_scheme_dataset()
    return msme_df, schemes_df


@st.cache_resource(show_spinner=False)
def _get_model_bundle(msme_df: pd.DataFrame, seed: int):
    model_artifacts = train_growth_model(msme_df, random_state=seed)
    scored_msme_df = score_msme_dataset(model_artifacts, msme_df)
    return model_artifacts, scored_msme_df


@st.cache_data(show_spinner=False)
def _run_policy_engine(
    msme_df: pd.DataFrame,
    schemes_df: pd.DataFrame,
    scored_msme_df: pd.DataFrame,
    revenue_weight: float,
    employment_weight: float,
    total_budget: float,
):
    growth_scores = scored_msme_df.set_index("MSME_ID")["Growth_Score"]
    policy_options_df, no_eligibility_ids = build_policy_options(
        msme_df=msme_df,
        schemes_df=schemes_df,
        growth_scores=growth_scores,
        revenue_weight=revenue_weight,
        employment_weight=employment_weight,
    )
    selected_df, optimization_summary = optimize_budget_allocation(
        options_df=policy_options_df,
        total_budget=float(total_budget),
    )
    selected_view, non_selected_df, sector_summary_df, totals = build_decision_views(
        msme_df=msme_df,
        options_df=policy_options_df,
        selected_df=selected_df,
        no_eligibility_ids=no_eligibility_ids,
        total_budget=float(total_budget),
    )
    return (
        policy_options_df,
        no_eligibility_ids,
        selected_df,
        optimization_summary,
        selected_view,
        non_selected_df,
        sector_summary_df,
        totals,
    )


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
            html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; color:#e2e8f0; }
            .stApp { background: radial-gradient(900px 300px at -10% -10%, rgba(20,184,166,.18), transparent 58%), radial-gradient(700px 250px at 110% -10%, rgba(99,102,241,.15), transparent 56%), linear-gradient(180deg,#0b1020 0%,#0b1124 100%); }
            [data-testid="stSidebar"] { background: linear-gradient(180deg,#0e1629 0%,#0c1424 100%); border-right:1px solid #23304a; }
            [data-testid="stSidebar"] * { color:#e2e8f0 !important; }
            .topbar { border:1px solid #23304a; border-radius:14px; padding:.8rem 1rem; margin-bottom:1rem; background:rgba(16,24,39,.88); display:flex; justify-content:space-between; align-items:center; }
            .topbar-title { font-weight:800; font-size:1rem; }
            .topbar-tag { color:#93a4bd; font-size:.82rem; font-weight:600; }
            .section-title { font-size:1.04rem; font-weight:800; margin-bottom:.2rem; }
            .section-subtitle { color:#93a4bd; font-size:.84rem; margin-bottom:.8rem; }
            .divider { height:1px; width:100%; background:#23304a; margin:.55rem 0 .9rem 0; }
            .card { border:1px solid #23304a; border-radius:14px; padding:.85rem .9rem; background:linear-gradient(180deg,#111a2d 0%,#0f1727 100%); transition:all .22s ease; }
            .card:hover { border-color:#31405f; transform:translateY(-1px); box-shadow:0 8px 26px rgba(2,10,28,.45); }
            .metric-label { color:#93a4bd; font-size:.75rem; font-weight:700; margin-bottom:.28rem; }
            .metric-value { color:#e2e8f0; font-size:1.28rem; font-weight:800; line-height:1.1; }
            .metric-foot { color:#14b8a6; margin-top:.2rem; font-size:.72rem; font-weight:700; }
            .recommend-card { border:1px solid #1e3a5f; border-left:4px solid #14b8a6; border-radius:14px; padding:.9rem; background:linear-gradient(135deg,#102438 0%,#0f1b2d 100%); margin-bottom:.8rem; }
            .recommend-title { font-size:.96rem; font-weight:800; margin-bottom:.2rem; }
            .recommend-sub { color:#93a4bd; font-size:.83rem; }
            .status-pill { display:inline-block; border-radius:999px; padding:.18rem .55rem; font-size:.73rem; font-weight:800; border:1px solid; margin-left:.4rem; }
            .pill-high { color:#86efac; border-color:#166534; background:rgba(34,197,94,.12); }
            .pill-moderate { color:#fcd34d; border-color:#92400e; background:rgba(245,158,11,.12); }
            .pill-low { color:#fca5a5; border-color:#7f1d1d; background:rgba(239,68,68,.12); }
            .status-banner { border:1px solid #204f46; background:rgba(20,184,166,.1); color:#99f6e4; border-radius:12px; padding:.7rem .8rem; margin-bottom:.7rem; font-size:.86rem; font-weight:700; }
            .status-banner.warn { border-color:#92400e; background:rgba(245,158,11,.1); color:#fcd34d; }
            .stTabs [data-baseweb="tab-list"] { gap:.45rem; }
            .stTabs [data-baseweb="tab"] { border-radius:10px; border:1px solid #23304a; background:#0f1a2d; color:#93a4bd; font-weight:700; min-height:2.4rem; padding:.28rem .9rem; }
            .stTabs [aria-selected="true"] { background:#12233a; color:#e2e8f0; border-color:#2f4e74; }
            div[data-testid="stMetric"] { border:1px solid #23304a; background:#101827; border-radius:12px; padding:.55rem .7rem; }
            div[data-testid="stDataFrame"] { border:1px solid #23304a; border-radius:12px; overflow:hidden; }
            button[kind="primary"], button[kind="secondary"] { border-radius:10px !important; transition:all .18s ease !important; }
            button[kind="primary"]:hover, button[kind="secondary"]:hover { transform:translateY(-1px); filter:brightness(1.05); }
            .module-box { border:1px solid #23304a; border-radius:14px; padding:.9rem; background:#101827; margin-bottom:1rem; }
            .reason-line { border-left:3px solid #14b8a6; padding:.35rem .55rem; margin-bottom:.38rem; border-radius:0 8px 8px 0; background:rgba(20,184,166,.09); color:#c5f4ee; font-size:.82rem; }
            .stProgress > div > div > div > div { background-color:#14b8a6; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_currency(value: float) -> str:
    return f"{value:,.0f}"


def _normalize_weights(revenue_weight: float, employment_weight: float) -> tuple[float, float]:
    total = revenue_weight + employment_weight
    if total <= 0:
        return 0.5, 0.5
    return revenue_weight / total, employment_weight / total


def _render_topbar() -> None:
    st.markdown(
        """
        <div class="topbar">
            <div class="topbar-title">Zyra MSME Intelligence Platform</div>
            <div class="topbar-tag">Predictive Growth Advisory + Budget-Aware Policy Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def _growth_badge(label: str) -> str:
    name = label.strip().lower()
    if name == "high":
        return '<span class="status-pill pill-high">HIGH</span>'
    if name == "moderate":
        return '<span class="status-pill pill-moderate">MODERATE</span>'
    return '<span class="status-pill pill-low">LOW</span>'


def _kpi_card(label: str, value: str, foot: str = "") -> None:
    foot_html = f'<div class="metric-foot">{foot}</div>' if foot else ""
    st.markdown(f"""<div class="card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div>{foot_html}</div>""", unsafe_allow_html=True)


def _advisory_card(scheme_name: str, scheme_id: str, reason: str, subsidy: float) -> None:
    st.markdown(
        f"""
        <div class="recommend-card">
            <div class="recommend-title">Recommended Scheme: {scheme_name} ({scheme_id})</div>
            <div class="recommend-sub">{reason}</div>
            <div class="recommend-sub" style="margin-top:6px; color:#8be8dd; font-weight:700;">Estimated Subsidy: INR {_format_currency(subsidy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _optimization_summary_card(totals: dict[str, float], optimization_summary: dict[str, float]) -> None:
    status_class = "status-banner" if totals["budget_respected"] else "status-banner warn"
    status_text = "Budget Status: Constraint satisfied and allocation is policy-aligned." if totals["budget_respected"] else "Budget Status: Constraint violation detected."
    st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
    pct = float(optimization_summary.get("budget_utilization_pct", 0.0))
    if PLOTLY_AVAILABLE:
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=pct, number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 26}}, gauge={"axis": {"range": [0, 100], "tickcolor": "#64748b"}, "bar": {"color": ACCENT}, "bgcolor": "#101827", "borderwidth": 0, "steps": [{"range": [0, 55], "color": "#1f2937"}, {"range": [55, 85], "color": "#1f3650"}, {"range": [85, 100], "color": "#123a35"}]}, title={"text": "Budget Utilization", "font": {"color": MUTED, "size": 13}}))
        gauge.update_layout(height=220, margin=dict(l=8, r=8, t=22, b=8), paper_bgcolor=PANEL, plot_bgcolor=PANEL)
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
    else:
        st.metric("Budget Utilization", f"{pct:.1f}%")
        st.progress(min(max(pct / 100.0, 0.0), 1.0))


def _styled_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _stripe(row: pd.Series) -> list[str]:
        base = "background-color: #101827; color: #e2e8f0;" if row.name % 2 == 0 else "background-color: #0f1727; color: #e2e8f0;"
        return [base] * len(row)

    return df.style.apply(_stripe, axis=1).set_table_styles([
        {"selector": "thead th", "props": "background-color:#121f34; color:#9fb0c8; font-weight:700; border-bottom:1px solid #26344f;"},
        {"selector": "tbody td", "props": "border-bottom:1px solid #1f2b44;"},
    ]).format(na_rep="-")


def _build_manual_profile(default_row: pd.Series, sectors: list[str]) -> dict[str, object]:
    col1, col2, col3 = st.columns(3)
    with col1:
        msme_id = st.text_input("MSME_ID", value="CUSTOM_0001")
        sector = st.selectbox("Sector", options=sectors, index=0)
        years = st.number_input("Years_of_Operation", min_value=1, max_value=40, value=int(default_row["Years_of_Operation"]))
        ownership = st.selectbox("Ownership_Type", options=["Proprietorship", "Partnership", "Private Limited", "Cooperative"], index=0)
        category = st.selectbox("Category", options=["Micro", "Small", "Medium"], index=0)
        location = st.selectbox("Location_Type", options=["Urban", "Semi-Urban", "Rural"], index=0)
    with col2:
        annual_revenue = st.number_input("Annual_Revenue", min_value=100_000.0, value=float(default_row["Annual_Revenue"]), step=100_000.0)
        revenue_growth = st.number_input("Revenue_Growth_Rate", min_value=-20.0, max_value=40.0, value=float(default_row["Revenue_Growth_Rate"]))
        profit_margin = st.number_input("Profit_Margin", min_value=0.0, max_value=40.0, value=float(default_row["Profit_Margin"]))
        debt_outstanding = st.number_input("Debt_Outstanding", min_value=0.0, value=float(default_row["Debt_Outstanding"]), step=100_000.0)
        loan_to_revenue = st.number_input("Loan_to_Revenue_Ratio", min_value=0.0, max_value=2.5, value=float(default_row["Loan_to_Revenue_Ratio"]), step=0.01)
    with col3:
        employees = st.number_input("Number_of_Employees", min_value=1, max_value=1000, value=int(default_row["Number_of_Employees"]))
        capacity = st.number_input("Capacity_Utilization", min_value=0.0, max_value=100.0, value=float(default_row["Capacity_Utilization"]))
        export_pct = st.number_input("Export_Percentage", min_value=0.0, max_value=100.0, value=float(default_row["Export_Percentage"]))
        tech_level = st.number_input("Technology_Level", min_value=1, max_value=5, value=int(default_row["Technology_Level"]))
        gst_score = st.number_input("GST_Compliance_Score", min_value=0.0, max_value=100.0, value=float(default_row["GST_Compliance_Score"]))
        inspection_score = st.number_input("Inspection_Score", min_value=0.0, max_value=100.0, value=float(default_row["Inspection_Score"]))
        docs_score = st.number_input("Documentation_Readiness_Score", min_value=0.0, max_value=100.0, value=float(default_row["Documentation_Readiness_Score"]))

    return {
        "MSME_ID": msme_id,
        "Sector": sector,
        "Years_of_Operation": int(years),
        "Ownership_Type": ownership,
        "Category": category,
        "Location_Type": location,
        "Annual_Revenue": float(annual_revenue),
        "Revenue_Growth_Rate": float(revenue_growth),
        "Profit_Margin": float(profit_margin),
        "Debt_Outstanding": float(debt_outstanding),
        "Loan_to_Revenue_Ratio": float(loan_to_revenue),
        "Number_of_Employees": int(employees),
        "Capacity_Utilization": float(capacity),
        "Export_Percentage": float(export_pct),
        "Technology_Level": int(tech_level),
        "GST_Compliance_Score": float(gst_score),
        "Inspection_Score": float(inspection_score),
        "Documentation_Readiness_Score": float(docs_score),
    }

def _select_profile(msme_df: pd.DataFrame) -> pd.DataFrame | None:
    st.markdown('<div class="module-box">', unsafe_allow_html=True)
    _render_section("Profile Intake", "Use existing records or enter a manual MSME profile.")
    mode = st.radio("Input Mode", options=["Use Existing MSME", "Manual Input"], horizontal=True)
    profile_df: pd.DataFrame | None = None

    if mode == "Use Existing MSME":
        selected_id = st.selectbox("Select MSME_ID", options=msme_df["MSME_ID"].tolist())
        if st.button("Run Advisory", use_container_width=True):
            selected_row = msme_df[msme_df["MSME_ID"] == selected_id].iloc[0]
            profile_df = pd.DataFrame([selected_row.drop(labels=["Growth_Category"])])
    else:
        default_row = msme_df.iloc[0]
        with st.form("manual_profile_form"):
            manual_profile = _build_manual_profile(default_row, sorted(msme_df["Sector"].unique()))
            submitted = st.form_submit_button("Run Advisory", use_container_width=True)
        if submitted:
            profile_df = pd.DataFrame([manual_profile])

    st.markdown("</div>", unsafe_allow_html=True)
    return profile_df


def _plot_probability(prob_map: dict[str, float]) -> None:
    df = pd.DataFrame({"Growth_Category": list(prob_map.keys()), "Probability": [float(v) * 100 for v in prob_map.values()]})
    if PLOTLY_AVAILABLE:
        fig = px.bar(df, x="Growth_Category", y="Probability", color="Growth_Category", color_discrete_map={"High": SUCCESS, "Moderate": WARN, "Low": DANGER}, template="plotly_dark", title="Class Probability Distribution")
        fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color="#dbeafe"), margin=dict(l=10, r=10, t=48, b=10), showlegend=False)
        fig.update_traces(hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.bar_chart(df.set_index("Growth_Category")["Probability"])


def _plot_sector_allocation(sector_summary_df: pd.DataFrame) -> None:
    if PLOTLY_AVAILABLE:
        fig = px.bar(sector_summary_df, x="Sector", y="Allocated_Subsidy", color="Sector", template="plotly_dark", color_discrete_sequence=["#14b8a6", "#22d3ee", "#38bdf8", "#818cf8", "#a78bfa", "#f59e0b"], title="Sector-wise Subsidy Allocation")
        fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color="#dbeafe"), margin=dict(l=10, r=10, t=48, b=10), xaxis_title=None, yaxis_title="Allocated Subsidy", showlegend=False)
        fig.update_traces(hovertemplate="%{x}<br>INR %{y:,.0f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.bar_chart(sector_summary_df.set_index("Sector")["Allocated_Subsidy"])


def _plot_feature_importance(feature_importance_df: pd.DataFrame) -> None:
    top_df = feature_importance_df.head(12).sort_values("importance_pct", ascending=True)
    if PLOTLY_AVAILABLE:
        fig = go.Figure(go.Bar(x=top_df["importance_pct"], y=top_df["feature"], orientation="h", marker=dict(color="#22d3ee", line=dict(color="#164e63", width=0.6)), hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
        fig.update_layout(template="plotly_dark", title="Feature Importance Signal", paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color="#dbeafe"), margin=dict(l=8, r=8, t=48, b=8), xaxis_title="Importance (%)", yaxis_title=None, height=420)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.bar_chart(top_df.set_index("feature")["importance_pct"])


def _plot_donut(
    labels: list[str],
    values: list[float],
    title: str,
    colors: list[str] | None = None,
) -> None:
    if not labels or not values or sum(values) == 0:
        st.info("No data available for this visualization.")
        return

    if PLOTLY_AVAILABLE:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.48,
                    sort=False,
                    marker=dict(
                        colors=colors if colors else None,
                        line=dict(color="#0b1020", width=2),
                    ),
                    pull=[0.03] + [0.0] * (len(labels) - 1),
                    textinfo="label+percent",
                    insidetextorientation="horizontal",
                    hovertemplate="%{label}<br>%{value:,.0f} (%{percent})<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor=PANEL,
            plot_bgcolor=PANEL,
            font=dict(color="#dbeafe"),
            margin=dict(l=8, r=8, t=48, b=8),
            showlegend=False,
            height=330,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        fallback_df = pd.DataFrame({"label": labels, "value": values}).set_index("label")
        st.bar_chart(fallback_df["value"])


def main() -> None:
    st.set_page_config(page_title="MSME Dual-Layer Advisory", layout="wide")
    _inject_theme()
    _render_topbar()
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not installed. Running with fallback charts. Install with: `pip install plotly`")

    with st.sidebar:
        st.markdown("### Controls")
        st.markdown("#### Data")
        n_entries = st.slider("Synthetic MSME entries", min_value=300, max_value=1200, value=360, step=20)
        random_seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)
        st.markdown("#### Budget")
        total_budget = st.number_input("Total Subsidy Budget", min_value=1_000_000.0, value=100_000_000.0, step=1_000_000.0)
        st.markdown("#### Policy Weights")
        raw_revenue_weight = st.slider("Revenue Priority Weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        raw_employment_weight = st.slider("Employment Priority Weight", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        revenue_weight, employment_weight = _normalize_weights(raw_revenue_weight, raw_employment_weight)
        st.caption(f"Effective policy mix: Revenue {revenue_weight:.2f} | Employment {employment_weight:.2f}")
        if st.button("Clear cached computations", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    with st.spinner("Preparing model signals and optimization baseline..."):
        msme_df, schemes_df = _get_synthetic_data(n_entries=n_entries, seed=int(random_seed))
        model_artifacts, scored_msme_df = _get_model_bundle(msme_df, int(random_seed))
        (
            policy_options_df,
            no_eligibility_ids,
            selected_df,
            optimization_summary,
            selected_view,
            non_selected_df,
            sector_summary_df,
            totals,
        ) = _run_policy_engine(
            msme_df=msme_df,
            schemes_df=schemes_df,
            scored_msme_df=scored_msme_df,
            revenue_weight=revenue_weight,
            employment_weight=employment_weight,
            total_budget=float(total_budget),
        )

    _render_section("Platform Snapshot", "Live analytics generated from current scenario controls.")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _kpi_card("MSMEs", f"{len(msme_df):,}", "Active simulation records")
    with m2:
        _kpi_card("Schemes", f"{len(schemes_df)}", "Policy options")
    with m3:
        _kpi_card("Model Accuracy", f"{model_artifacts.metrics['accuracy']:.3f}", "Classification validation")
    with m4:
        _kpi_card("Macro F1", f"{model_artifacts.metrics['macro_f1']:.3f}", "Balanced model quality")

    tab1, tab2, tab3 = st.tabs(["Advisory", "Policy Engine", "Transparency"])

    with tab1:
        _render_section("MSME Advisory Module", "Personalized growth prediction, scheme matching, and impact simulation.")
        left, right = st.columns([1.05, 1.45], gap="large")
        with left:
            profile_df = _select_profile(msme_df)
        with right:
            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Recommendation Output", "Predicted growth category, projected impact, and rationale.")
            if profile_df is None:
                st.info("Submit a profile from the left panel to run advisory.")
            else:
                with st.spinner("Running advisory prediction and scheme simulation..."):
                    prediction = predict_growth(model_artifacts, profile_df)
                    profile_row = profile_df.iloc[0]
                    options_df, _ = build_policy_options(msme_df=profile_df.assign(Growth_Category="Unknown"), schemes_df=schemes_df, growth_scores=pd.Series([prediction["growth_score"]], index=[profile_row["MSME_ID"]]), revenue_weight=revenue_weight, employment_weight=employment_weight)

                c1, c2, c3 = st.columns(3)
                with c1:
                    _kpi_card("Predicted Category", prediction["predicted_category"], "Model signal")
                with c2:
                    _kpi_card("Growth Score", f"{prediction['growth_score']:.1f}", "Scale: 0-100")
                with c3:
                    badge = _growth_badge(prediction["predicted_category"])
                    st.markdown(f'<div class="card"><div class="metric-label">Risk Band</div><div class="metric-value">{badge}</div></div>', unsafe_allow_html=True)

                if options_df.empty:
                    st.warning("No schemes are eligible for this MSME profile.")
                else:
                    ranked_options = rank_impacts_for_advisory(impacts_df=options_df, revenue_weight=revenue_weight, employment_weight=employment_weight)
                    best = ranked_options.iloc[0]
                    _advisory_card(
                        scheme_name=str(best["Scheme_Name"]),
                        scheme_id=str(best["Scheme_ID"]),
                        reason=f"Selected for highest weighted one-year impact under current policy mix (Revenue {revenue_weight:.2f}, Employment {employment_weight:.2f}).",
                        subsidy=float(best["Subsidy_Amount"]),
                    )

                    p1, p2, p3, p4 = st.columns(4)
                    with p1:
                        _kpi_card("Current Revenue", _format_currency(float(best["Base_Revenue"])))
                    with p2:
                        _kpi_card("Projected Revenue", _format_currency(float(best["Projected_Revenue_After"])), f"+{_format_currency(float(best['Projected_Revenue_Increase']))}")
                    with p3:
                        _kpi_card("Current Employment", str(int(best["Base_Employment"])))
                    with p4:
                        _kpi_card("Projected Employment", str(int(best["Projected_Employment_After"])), f"+{int(best['Projected_Employment_Increase'])}")

                    st.markdown("#### Eligible Scheme Comparison")
                    display_cols = ["Scheme_ID", "Scheme_Name", "Subsidy_Amount", "Base_Revenue", "Projected_Revenue_Increase", "Projected_Revenue_After", "Base_Employment", "Projected_Employment_Increase", "Projected_Employment_After", "Weighted_Impact_Score"]
                    table_df = ranked_options[display_cols].sort_values("Weighted_Impact_Score", ascending=False).copy()
                    table_df["Subsidy_Amount"] = table_df["Subsidy_Amount"].map(lambda x: _format_currency(float(x)))
                    table_df["Base_Revenue"] = table_df["Base_Revenue"].map(lambda x: _format_currency(float(x)))
                    table_df["Projected_Revenue_Increase"] = table_df["Projected_Revenue_Increase"].map(lambda x: _format_currency(float(x)))
                    table_df["Projected_Revenue_After"] = table_df["Projected_Revenue_After"].map(lambda x: _format_currency(float(x)))
                    st.dataframe(_styled_table(table_df), use_container_width=True, hide_index=True)

                st.markdown("#### Explainability")
                reasons = generate_local_reasons(profile_row, model_artifacts.reference_medians)
                for reason in reasons:
                    st.markdown(f'<div class="reason-line">{reason}</div>', unsafe_allow_html=True)
                _plot_probability(prediction["probabilities"])
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        _render_section("Policy Engine Module", "Budget-constrained optimization with weighted economic outcomes.")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            _kpi_card("Selected MSMEs", str(int(totals["selected_count"])))
        with k2:
            _kpi_card("Subsidy Allocated", _format_currency(float(totals["total_subsidy_allocated"])))
        with k3:
            _kpi_card("Revenue Gain", _format_currency(float(totals["total_projected_revenue_gain"])))
        with k4:
            _kpi_card("Employment Generation", str(int(totals["total_projected_employment_generation"])))
        with k5:
            _kpi_card("Budget Utilization", f"{optimization_summary['budget_utilization_pct']:.1f}%")

        left, right = st.columns([1.1, 1.2], gap="large")
        with left:
            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Executive Summary", "Decision-grade overview for policy teams.")
            _optimization_summary_card(totals, optimization_summary)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Sector Allocation", "Subsidy distribution by sector.")
            if sector_summary_df.empty:
                st.info("No sector allocation available under current constraints.")
            else:
                _plot_sector_allocation(sector_summary_df)
                sector_table = sector_summary_df.copy()
                sector_table["Allocated_Subsidy"] = sector_table["Allocated_Subsidy"].map(lambda x: _format_currency(float(x)))
                sector_table["Projected_Revenue_Gain"] = sector_table["Projected_Revenue_Gain"].map(lambda x: _format_currency(float(x)))
                st.dataframe(_styled_table(sector_table), use_container_width=True, hide_index=True)

                d1, d2 = st.columns(2)
                with d1:
                    _plot_donut(
                        labels=sector_summary_df["Sector"].astype(str).tolist(),
                        values=sector_summary_df["Allocated_Subsidy"].astype(float).tolist(),
                        title="Allocation Share by Sector (3D-style Donut)",
                        colors=["#14b8a6", "#22d3ee", "#38bdf8", "#818cf8", "#a78bfa", "#f59e0b"],
                    )
                with d2:
                    scheme_mix = (
                        selected_view.groupby("Scheme_Name", as_index=False)["Subsidy_Amount"]
                        .sum()
                        .sort_values("Subsidy_Amount", ascending=False)
                    )
                    _plot_donut(
                        labels=scheme_mix["Scheme_Name"].astype(str).tolist(),
                        values=scheme_mix["Subsidy_Amount"].astype(float).tolist(),
                        title="Allocation Share by Scheme (3D-style Donut)",
                        colors=["#14b8a6", "#06b6d4", "#3b82f6", "#8b5cf6", "#f59e0b"],
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Selected MSMEs", "Ranked outputs under current budget and policy mix.")
            selected_table = selected_view.copy()
            if selected_table.empty:
                st.info("No MSMEs selected in current run.")
            else:
                selected_table["Subsidy_Amount"] = selected_table["Subsidy_Amount"].map(lambda x: _format_currency(float(x)))
                selected_table["Projected_Revenue_Increase"] = selected_table["Projected_Revenue_Increase"].map(lambda x: _format_currency(float(x)))
                st.dataframe(_styled_table(selected_table), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Selection Mix", "Quick split of selected vs non-selected MSMEs.")
            _plot_donut(
                labels=["Selected", "Non-selected"],
                values=[float(totals["selected_count"]), float(totals["non_selected_count"])],
                title="Selection Status (3D-style Donut)",
                colors=["#22c55e", "#ef4444"],
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Non-selected MSMEs", "Transparent reasons for non-allocation.")
            non_sel_table = non_selected_df.copy()
            if non_sel_table.empty:
                st.info("No non-selected MSMEs to display.")
            else:
                non_sel_table["Required_Subsidy"] = non_sel_table["Required_Subsidy"].map(lambda x: _format_currency(float(x)))
                non_sel_table["Potential_Revenue_Gain"] = non_sel_table["Potential_Revenue_Gain"].map(lambda x: _format_currency(float(x)))
                st.dataframe(_styled_table(non_sel_table.head(250)), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        _render_section("Transparency Module", "Audit-ready evidence, model behavior, and data visibility.")
        c1, c2 = st.columns([1.2, 1.0], gap="large")
        with c1:
            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Model Evaluation", "Performance diagnostics for growth category classifier.")
            report_df = model_artifacts.classification_report[["label", "precision", "recall", "f1-score", "support"]]
            st.dataframe(_styled_table(report_df), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Feature Set", "Columns used for model training and policy simulation.")
            st.code("Feature columns used by model:\n" + str(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS), language="text")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Feature Importance", "Top global signals influencing growth prediction.")
            _plot_feature_importance(model_artifacts.feature_importance)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Growth Distribution", "Class balance in synthetic MSME dataset.")
            growth_counts = msme_df["Growth_Category"].value_counts().sort_index().reset_index()
            growth_counts.columns = ["Growth_Category", "Count"]
            if PLOTLY_AVAILABLE:
                dist_fig = px.bar(growth_counts, x="Growth_Category", y="Count", color="Growth_Category", color_discrete_map={"High": SUCCESS, "Moderate": WARN, "Low": DANGER}, template="plotly_dark", title="Growth Category Distribution")
                dist_fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color="#dbeafe"), margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
                st.plotly_chart(dist_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.bar_chart(growth_counts.set_index("Growth_Category")["Count"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Population Mix", "Sector and category composition of synthetic MSMEs.")
            t1, t2 = st.columns(2)
            with t1:
                sector_mix = msme_df["Sector"].value_counts().reset_index()
                sector_mix.columns = ["Sector", "Count"]
                _plot_donut(
                    labels=sector_mix["Sector"].astype(str).tolist(),
                    values=sector_mix["Count"].astype(float).tolist(),
                    title="MSME Sector Mix (3D-style Donut)",
                    colors=["#14b8a6", "#22d3ee", "#38bdf8", "#818cf8", "#a78bfa", "#f59e0b"],
                )
            with t2:
                category_mix = msme_df["Category"].value_counts().reset_index()
                category_mix.columns = ["Category", "Count"]
                _plot_donut(
                    labels=category_mix["Category"].astype(str).tolist(),
                    values=category_mix["Count"].astype(float).tolist(),
                    title="MSME Category Mix (3D-style Donut)",
                    colors=["#22c55e", "#f59e0b", "#ef4444"],
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="module-box">', unsafe_allow_html=True)
            _render_section("Dataset Snapshot", "Quick view for data quality verification.")
            st.dataframe(_styled_table(msme_df.head(20)), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
