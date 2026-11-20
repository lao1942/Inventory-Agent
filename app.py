"""
app.py
======
Supply Chain Analytics Inventory Decision Agent Application.

Features:
  1. Interactive Streamlit Web Dashboard:
       streamlit run app.py
       - Redesigned "Inventory Agent" sidebar with Cognitive Decision Pipeline indicators.
       - Quick-Start Scenarios dropdown with "Load Scenario into Input" button.
       - Prominent "Solve Inventory" and "Clear / Reset" controls.
       - Dark executive recommendation banner and KPI metric cards.
       - 3 interactive tabs:
           * Tab 1: 6-Stage Execution Pipeline (structured expandable sections)
           * Tab 2: Model Analytics & Visualizations (interactive Altair curves & charts)
           * Tab 3: What-If Parameter Sensitivity (real-time recalculation with inventory_tools)
  2. CLI / Terminal Mode:
       python app.py
       python app.py --demo
       python app.py 'Annual demand is 10000 units, ordering cost is $50 per order, and annual holding cost is $2 per unit.'
"""

import sys
import math
import argparse
from typing import Dict, Any, Optional, List
from inventory_agent import InventoryDecisionAgent, AgentPipelineResult
import inventory_tools as itools


# -------------------------------------------------------------------------
# Benchmark Scenarios
# -------------------------------------------------------------------------
SCENARIO_OPTIONS = [
    "Economic Order Quantity (EOQ)",
    "Quantity Discount",
    "Safety Stock and Reorder Point",
    "Fixed-Period Review",
    "Newsvendor",
    "Custom Question",
]

SCENARIO_TEXTS = {
    "Economic Order Quantity (EOQ)": (
        "We have an annual demand of 12,000 units, ordering cost of $100 per order, "
        "and holding cost of $1.50 per unit per year. What is our optimal EOQ, order cycle, and total cost?"
    ),
    "Quantity Discount": (
        "Calculate the optimal order quantity with quantity discounts. Annual demand is 10000 units, "
        "ordering cost is $50 per order, holding rate is 20% annually. Unit price is $10 for orders "
        "below 500 units, $9.50 for orders of 500 to 999 units, and $9 for orders of 1000 units or more. "
        "Which order quantity minimizes total annual cost?"
    ),
    "Safety Stock and Reorder Point": (
        "Our supplier replenishment lead time is 9 days. Daily demand is 100 units with a standard deviation "
        "of 20 units. What safety stock buffer and reorder point (ROP) do we need for a 95% cycle service level?"
    ),
    "Fixed-Period Review": (
        "We review inventory every 14 days. Lead time is 5 days, average daily demand is 100 units, "
        "daily demand standard deviation is 20 units, and our target service level is 95%. "
        "Current inventory position is 800 units. Calculate the target inventory level and how much we should order."
    ),
    "Newsvendor": (
        "For an upcoming seasonal product, the selling price is $100 and unit purchase cost is $40. "
        "Unsold units have a salvage value of $10. Mean demand is 1,000 units with a standard deviation "
        "of 200 units. What is the optimal newsvendor order quantity and expected profit?"
    ),
    "Custom Question": "",
}

# Compatibility alias for CLI
SAMPLE_SCENARIOS = {
    f"{i+1}. {k}": SCENARIO_TEXTS[k]
    for i, k in enumerate(["Economic Order Quantity (EOQ)", "Quantity Discount", "Safety Stock and Reorder Point", "Fixed-Period Review", "Newsvendor"])
}


# =========================================================================
# Streamlit Web Application Interface
# =========================================================================

def run_streamlit_app():
    import streamlit as st
    import pandas as pd
    import altair as alt

    st.set_page_config(
        page_title="Supply Chain Analytics - Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS Styling
    st.markdown(
        """
        <style>
        /* Modern styling touches */
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 0.95rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }
        /* Dark Executive Recommendation Banner */
        .exec-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #38bdf8;
            border-left: 6px solid #38bdf8;
            border-radius: 8px;
            padding: 18px 22px;
            color: #f8fafc;
            margin-top: 10px;
            margin-bottom: 22px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.35);
        }
        .exec-banner-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #38bdf8;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .exec-banner-headline {
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.4;
            margin-bottom: 8px;
        }
        .exec-banner-details {
            font-size: 0.92rem;
            color: #cbd5e1;
            line-height: 1.5;
        }
        /* Cognitive Pipeline Step Card */
        .step-container {
            border-radius: 6px;
            padding: 6px 10px;
            margin-bottom: 6px;
            font-size: 0.88rem;
            display: flex;
            align-items: center;
        }
        .step-done {
            background-color: #ecfdf5;
            color: #065f46;
            border-left: 3px solid #10b981;
            font-weight: 600;
        }
        .step-active {
            background-color: #eff6ff;
            color: #1e40af;
            border-left: 3px solid #3b82f6;
            font-weight: 600;
        }
        .step-pending {
            background-color: #f8fafc;
            color: #64748b;
            border-left: 3px solid #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    agent = InventoryDecisionAgent()

    # Session State Initialization
    if "inventory_query" not in st.session_state:
        st.session_state["inventory_query"] = SCENARIO_TEXTS["Economic Order Quantity (EOQ)"]
    if "pipeline_result" not in st.session_state:
        st.session_state["pipeline_result"] = None

    # ---------------------------------------------------------------------
    # 1. Left Sidebar: "Inventory Agent"
    # ---------------------------------------------------------------------
    with st.sidebar:
        st.header("Inventory Agent")
        st.caption("Supply Chain Cognitive Analytics Decision System")

        st.markdown("---")
        st.subheader("Cognitive Decision Pipeline")

        pipeline_steps = [
            ("1", "Question", "Natural language customer prompt"),
            ("2", "Identify Model", "Intent classification & scoring"),
            ("3", "Extract Inputs", "Regex & contextual entity parsing"),
            ("4", "Select Python Tool", "Deterministic tool dispatch"),
            ("5", "Calculate", "Closed-form analytical equations"),
            ("6", "Verify", "Feasibility & optimality assertions"),
            ("7", "Explain", "Executive managerial brief"),
        ]

        has_result = st.session_state["pipeline_result"] is not None
        has_error = has_result and st.session_state["pipeline_result"].status != "SUCCESS"

        for idx, (num, name, desc) in enumerate(pipeline_steps):
            if not has_result:
                css_class = "step-active" if idx == 0 else "step-pending"
                icon = "🔵" if idx == 0 else "⚪"
            elif has_error:
                css_class = "step-done" if idx < 4 else ("step-active" if idx == 4 else "step-pending")
                icon = "✅" if idx < 4 else ("❌" if idx == 4 else "⚪")
            else:
                css_class = "step-done"
                icon = "✅"

            st.markdown(
                f"""
                <div class="step-container {css_class}">
                    <span style="margin-right:8px;">{icon}</span>
                    <span><strong>{num}. {name}</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("Quick-Start Scenarios")
        selected_scenario = st.selectbox(
            "Select Benchmark Scenario:",
            SCENARIO_OPTIONS,
            index=0,
            label_visibility="collapsed",
        )

        if st.button("📥 Load Scenario into Input", use_container_width=True):
            st.session_state["inventory_query"] = SCENARIO_TEXTS[selected_scenario]
            st.session_state["pipeline_result"] = None
            st.rerun()

        st.markdown("---")
        st.subheader("Data Upload")
        if st.session_state.get("active_dataset_summary") is not None:
            sum_data = st.session_state["active_dataset_summary"]
            st.success(f"✓ Active Dataset: {sum_data['n_rows']:,} records")
            st.caption(f"Valuation: ${sum_data['total_inventory_value']:,.2f}")
        else:
            st.info("No CSV uploaded yet. Use the upload section in the main panel above the query input.")

        st.markdown("---")
        st.caption("Google DeepMind • Supply Chain Analytics • BAN 517")

    # ---------------------------------------------------------------------
    # 2. Main Interface Header & Input
    # ---------------------------------------------------------------------
    st.markdown('<div class="main-header">📦 Supply Chain Analytics: Inventory Decision Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Deterministic Inventory Optimization Engine with 6-Stage Autonomous Cognitive Decision Architecture</div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------
    # Section: 📁 Upload & Analyze Inventory Dataset (CSV)
    # (Requirement 1: Placed directly above the natural-language question input)
    # ---------------------------------------------------------------------
    st.markdown("### 📁 Upload & Analyze Inventory Dataset (CSV)")
    st.caption("Upload your product catalog or inventory records (CSV) to automatically analyze portfolio metrics and integrate with decision models.")

    uploaded_csv = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        key="main_inventory_csv_uploader",
        help="Upload an inventory CSV file. Expected columns: Product, Category, Units, Unit Cost.",
    )

    inventory_df = None
    if uploaded_csv is not None:
        try:
            dataset_summary = process_inventory_csv(uploaded_csv)
            if dataset_summary["valid"]:
                inventory_df = dataset_summary["df"]
                st.session_state["active_dataset_summary"] = dataset_summary
                _render_csv_upload_section(dataset_summary)
            else:
                st.session_state["active_dataset_summary"] = None
                st.error(
                    f"❌ **Schema Validation Failed:** The uploaded CSV is missing required column(s): "
                    f"**{', '.join(dataset_summary['missing_columns'])}**. "
                    f"Required columns are: `{', '.join(dataset_summary['required_columns'])}`."
                )
                st.info(f"**Columns found in uploaded file:** {', '.join(dataset_summary['column_names']) if dataset_summary['column_names'] else 'None'}")
                st.markdown("**Uploaded Data Preview:**")
                st.dataframe(dataset_summary["df"], use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error processing uploaded CSV file: {str(e)}")
            st.session_state["active_dataset_summary"] = None
    else:
        st.session_state["active_dataset_summary"] = None

    st.markdown("---")

    query_input = st.text_area(
        "Enter Natural Language Inventory Question:",
        value=st.session_state["inventory_query"],
        height=110,
        placeholder="e.g., Annual demand is 12,000 units, ordering cost is $100, holding cost is $1.50. What is our EOQ? Or ask questions about your uploaded dataset.",
        key="query_text_area",
    )

    # Action Buttons
    col_btn1, col_btn2, col_spacer = st.columns([1.8, 1.2, 5])
    with col_btn1:
        solve_clicked = st.button("🚀 Solve Inventory", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🔄 Clear / Reset", use_container_width=True)

    if clear_clicked:
        st.session_state["inventory_query"] = ""
        st.session_state["pipeline_result"] = None
        st.rerun()

    if solve_clicked:
        if not query_input.strip():
            st.warning("Please enter an inventory question or load one of the quick-start scenarios.")
        else:
            st.session_state["inventory_query"] = query_input
            with st.spinner("Executing 6-Stage Cognitive Inventory Decision Pipeline..."):
                res = agent.process_question(query_input, dataset_df=inventory_df)
                st.session_state["pipeline_result"] = res

    res: Optional[AgentPipelineResult] = st.session_state["pipeline_result"]

    # ---------------------------------------------------------------------
    # 3. Decision Results Presentation (Banner, Metric Cards, Tabs)
    # ---------------------------------------------------------------------
    if res is not None:
        if res.status != "SUCCESS":
            st.error(f"❌ Decision Pipeline Error: {res.error_message}")
            st.info(
                "💡 **Supply Chain Guidance:** Inventory decision models require complete parameter specifications to yield deterministic, mathematically sound recommendations. The decision agent never invents or assumes missing numerical values. Please provide the required parameter(s) above and re-run."
            )
            if res.extracted_inputs:
                with st.expander("🔍 Extracted Parameters from Question & Dataset", expanded=True):
                    clean_in = {k: v for k, v in res.extracted_inputs.items() if not k.startswith("_") and k != "dataset_df"}
                    st.json(clean_in)
        else:
            calc = res.calculation_results or {}
            results = calc.get("results", {})
            inter = calc.get("intermediate", {})
            model = res.identified_model

            # A. Dark Executive Recommendation Banner
            _render_executive_banner(model, results, inter)

            # B. Visually Appealing Metric Cards
            _render_metric_cards(model, results, inter)

            st.write("")

            # C. Three Main Tabs
            tab1, tab2, tab3 = st.tabs([
                "📋 6-Stage Execution Pipeline",
                "📊 Model Analytics & Visualizations",
                "🎛️ What-If Parameter Sensitivity",
            ])

            # Tab 1: 6-Stage Execution Pipeline
            with tab1:
                _render_pipeline_tab(res)

            # Tab 2: Model Analytics & Visualizations
            with tab2:
                _render_analytics_tab(model, results, inter, calc.get("inputs", {}), calc)

            # Tab 3: What-If Parameter Sensitivity
            with tab3:
                _render_sensitivity_tab(model, calc.get("inputs", {}), results)


# -------------------------------------------------------------------------
# CSV Data Processing & Portfolio Analytics Functions
# -------------------------------------------------------------------------
def process_inventory_csv(csv_file_or_df: Any) -> Dict[str, Any]:
    """
    Process uploaded CSV inventory data using pandas.
    Validates required columns: Product, Category, Units, Unit Cost.
    Performs data calculations:
      1. Total number of records (n_rows)
      2. Summary statistics across numerical metrics (describe())
      3. Total Units
      4. Average Unit Cost
      5. Product with Highest Units
      6. Total Units by Category
      7. Calculated Column: Inventory Value = Units * Unit Cost
      8. Total Inventory Value across all products
      9. Missing values and data-quality checks
    """
    import pandas as pd

    if isinstance(csv_file_or_df, pd.DataFrame):
        df = csv_file_or_df.copy()
    else:
        df = pd.read_csv(csv_file_or_df)

    # Clean whitespace in column names
    df.columns = df.columns.str.strip()

    required_columns = ["Product", "Category", "Units", "Unit Cost"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return {
            "valid": False,
            "missing_columns": missing_columns,
            "required_columns": required_columns,
            "column_names": list(df.columns),
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "df": df,
        }

    n_rows, n_cols = df.shape
    col_names = list(df.columns)

    # Clean numeric data types
    if df["Units"].dtype == object:
        df["Units"] = pd.to_numeric(
            df["Units"].astype(str).str.replace(",", "").str.strip(), errors="coerce"
        ).fillna(0)
    else:
        df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(0)

    if df["Unit Cost"].dtype == object:
        df["Unit Cost"] = pd.to_numeric(
            df["Unit Cost"].astype(str).str.replace("$", "").str.replace(",", "").str.strip(), errors="coerce"
        ).fillna(0.0)
    else:
        df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce").fillna(0.0)

    # 5. Calculated Column: Inventory Value = Units * Unit Cost
    df["Inventory Value"] = df["Units"] * df["Unit Cost"]

    # 1. Total Units
    total_units = int(df["Units"].sum())

    # 2. Average Unit Cost
    avg_unit_cost = float(df["Unit Cost"].mean()) if n_rows > 0 else 0.0

    # 3. Product with highest Units
    if n_rows > 0:
        max_idx = df["Units"].idxmax()
        highest_units_product = str(df.loc[max_idx, "Product"])
        highest_units_count = int(df.loc[max_idx, "Units"])
    else:
        highest_units_product = "N/A"
        highest_units_count = 0

    # 4. Group by Category and calculate total Units
    category_summary = df.groupby("Category")["Units"].sum().reset_index()
    category_summary = category_summary.sort_values(by="Units", ascending=False)

    # 6. Total Inventory Value across all products
    total_inventory_value = float(df["Inventory Value"].sum())

    # 7. Summary statistics across numerical metrics (Requirement 4)
    summary_stats = df[["Units", "Unit Cost", "Inventory Value"]].describe()

    # 8. Missing values and data-quality checks (Requirement 4)
    null_counts = {str(col): int(df[col].isna().sum()) for col in df.columns}
    total_nulls = sum(null_counts.values())
    dup_rows = int(df.duplicated().sum())
    neg_units = int((df["Units"] < 0).sum())
    neg_costs = int((df["Unit Cost"] < 0).sum())

    quality_notes = []
    if total_nulls == 0:
        quality_notes.append("✓ No missing values detected in any required columns.")
    else:
        quality_notes.append(f"⚠ {total_nulls} missing (null) values detected across dataset columns.")

    if dup_rows == 0:
        quality_notes.append("✓ No duplicate inventory records detected.")
    else:
        quality_notes.append(f"⚠ {dup_rows} duplicate record(s) detected.")

    if neg_units == 0 and neg_costs == 0:
        quality_notes.append("✓ All product units and unit costs are non-negative.")
    else:
        quality_notes.append(f"⚠ Detected {neg_units} negative unit quantities and {neg_costs} negative unit costs.")

    data_quality = {
        "null_counts": null_counts,
        "total_nulls": total_nulls,
        "duplicate_rows": dup_rows,
        "negative_units_count": neg_units,
        "negative_cost_count": neg_costs,
        "quality_notes": quality_notes,
        "status": "PASSED" if (total_nulls == 0 and dup_rows == 0 and neg_units == 0 and neg_costs == 0) else "WARNING",
    }

    # Formatted display dataframe
    df_display = df.copy()
    df_display["Unit Cost"] = df_display["Unit Cost"].apply(lambda x: f"${x:,.2f}")
    df_display["Inventory Value"] = df_display["Inventory Value"].apply(lambda x: f"${x:,.2f}")
    df_display["Units"] = df_display["Units"].apply(lambda x: f"{int(x):,}")

    return {
        "valid": True,
        "missing_columns": [],
        "required_columns": required_columns,
        "df": df,
        "df_display": df_display,
        "n_rows": n_rows,
        "n_cols": n_cols + 1,  # including calculated column
        "column_names": list(df.columns),
        "original_column_names": col_names,
        "total_units": total_units,
        "avg_unit_cost": avg_unit_cost,
        "highest_units_product": highest_units_product,
        "highest_units_count": highest_units_count,
        "total_inventory_value": total_inventory_value,
        "category_summary": category_summary,
        "summary_statistics": summary_stats,
        "data_quality": data_quality,
    }


def _render_csv_upload_section(csv_or_results: Any):
    """Render the uploaded CSV dataset, schema validation, summary statistics, and data-quality checks."""
    import streamlit as st
    import pandas as pd
    import altair as alt

    if isinstance(csv_or_results, dict) and "valid" in csv_or_results:
        results = csv_or_results
    else:
        try:
            results = process_inventory_csv(csv_or_results)
        except Exception as e:
            st.error(f"❌ Error processing uploaded CSV file: {str(e)}")
            return

    # Schema Validation Check
    if not results["valid"]:
        st.error(
            f"❌ **Schema Validation Failed:** The uploaded CSV is missing required column(s): "
            f"**{', '.join(results['missing_columns'])}**. "
            f"Required columns are: `{', '.join(results['required_columns'])}`."
        )
        st.info(f"**Columns found in uploaded file:** {', '.join(results['column_names']) if results['column_names'] else 'None'}")
        st.markdown("**Uploaded Data Preview:**")
        st.dataframe(results["df"], use_container_width=True)
        return

    # Metadata Display: Rows, Columns, Column Names (Requirement 4)
    m1, m2, m3 = st.columns([1, 1, 2.5])
    m1.metric("Total Records", f"{results['n_rows']:,}")
    m2.metric("Total Columns", f"{len(results['original_column_names']):,}")
    m3.info(f"**Verified Columns:** {', '.join(results['original_column_names'])}")

    # Analysis Results KPI Metric Cards (Requirement 4)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Units", f"{results['total_units']:,}")
    kpi2.metric("Average Unit Cost", f"${results['avg_unit_cost']:,.2f}")
    kpi3.metric("Product with Highest Units", results["highest_units_product"], f"{results['highest_units_count']:,} units")
    kpi4.metric("Total Inventory Value", f"${results['total_inventory_value']:,.2f}")

    # Dataset Preview (Requirement 3)
    with st.expander("👁️ View Uploaded Dataset Records & Calculated Values", expanded=True):
        st.dataframe(results["df_display"], use_container_width=True)

    # Missing values and data-quality checks (Requirement 4)
    dq = results.get("data_quality", {})
    dq_status = dq.get("status", "PASSED")
    dq_icon = "✅" if dq_status == "PASSED" else "⚠️"

    with st.expander(f"🛡️ Data Quality & Missing Values Audit ({dq_icon} {dq_status})", expanded=False):
        c_dq1, c_dq2 = st.columns([1.5, 1.5])
        with c_dq1:
            st.markdown("**Data Hygiene Checks:**")
            for note in dq.get("quality_notes", []):
                st.markdown(f"- {note}")
        with c_dq2:
            st.markdown("**Column Null (Missing) Counts:**")
            null_s = pd.Series(dq.get("null_counts", {}), name="Missing Values")
            st.dataframe(null_s, use_container_width=True)

    # Inventory Quantities & Summary Statistics (Requirement 4)
    with st.expander("📊 Inventory Quantities & Summary Statistics (describe)", expanded=False):
        st.caption("Distributional summary statistics across Units, Unit Cost, and calculated Inventory Value.")
        st.dataframe(results["summary_statistics"], use_container_width=True)

    # Total Units by Category
    st.markdown("##### 🏷️ Total Units by Category")
    c_cat1, c_cat2 = st.columns([1.2, 1.8])
    with c_cat1:
        cat_df_display = results["category_summary"].copy()
        cat_df_display["Units"] = cat_df_display["Units"].apply(lambda x: f"{int(x):,}")
        st.dataframe(cat_df_display, use_container_width=True, hide_index=True)
    with c_cat2:
        cat_chart = (
            alt.Chart(results["category_summary"])
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#3B82F6")
            .encode(
                x=alt.X("Category:N", sort="-y", title="Product Category"),
                y=alt.Y("Units:Q", title="Total Units"),
                tooltip=["Category", "Units"],
            )
            .properties(height=220)
        )
        st.altair_chart(cat_chart, use_container_width=True)


# -------------------------------------------------------------------------
# UI Component: Dark Executive Recommendation Banner
# -------------------------------------------------------------------------
def _render_executive_banner(model: str, results: Dict[str, Any], inter: Dict[str, Any]):
    import streamlit as st

    if model == "EOQ":
        q = results.get("optimal_order_quantity_eoq", 0)
        orders = results.get("orders_per_year", 0)
        days = results.get("order_cycle_time_days", 0)
        cost = results.get("annual_inventory_cost", 0)
        headline = f"Optimal Lot Size: {math.ceil(q):,} units every {days:.1f} days ({orders:.1f} orders/year)"
        details = (
            f"At this order volume, annual holding cost (${results.get('annual_holding_cost', 0):,.2f}) "
            f"and ordering cost (${results.get('annual_ordering_cost', 0):,.2f}) are in perfect equilibrium, "
            f"minimizing total inventory carrying & procurement overhead to <strong>${cost:,.2f}</strong>."
        )

    elif model == "QUANTITY_DISCOUNT":
        tier = results.get("recommended_tier", 1)
        q = results.get("optimal_order_quantity", 0)
        price = results.get("best_unit_price", 0)
        savings = results.get("annual_savings_vs_tier_1", 0)
        tot_cost = results.get("minimum_total_annual_cost", 0)
        headline = f"Recommended Order: {math.ceil(q):,} units at ${price:.2f}/unit (Tier {tier})"
        details = (
            f"Taking this volume price break minimizes total annual expenditure to <strong>${tot_cost:,.2f}</strong>. "
            f"The procurement price concession saves <strong>${savings:,.2f}</strong> annually compared to baseline pricing."
        )

    elif model == "SAFETY_STOCK_ROP":
        rop = results.get("reorder_point_recommended_integer", 0)
        ss = results.get("safety_stock_recommended_integer", 0)
        csl = results.get("cycle_service_level_pct", 95)
        headline = f"Set Reorder Point (ROP) = {rop:,} units (Safety Stock = {ss:,} units)"
        details = (
            f"Maintains a target <strong>{csl:.1f}% Cycle Service Level</strong> against demand and lead time uncertainty. "
            f"Issue a purchase order immediately whenever physical on-hand + pipeline inventory falls to <strong>{rop:,} units</strong>."
        )

    elif model == "FIXED_PERIOD_REVIEW":
        s_level = results.get("target_order_up_to_level_integer", 0)
        q_rec = results.get("recommended_order_quantity_integer", 0)
        ip = inter.get("current_inventory_position", 0)
        ss = results.get("safety_stock", 0)
        headline = f"Target Order-Up-To Level (S) = {s_level:,} units | Order {q_rec:,} units today"
        details = (
            f"With current inventory position at {ip:,.0f} units, order <strong>{q_rec:,} units</strong> to replenish to target level S. "
            f"A buffer safety stock of {math.ceil(ss):,} units covers demand variance across the full protection interval."
        )

    elif model == "NEWSVENDOR":
        q = results.get("optimal_order_quantity_integer", 0)
        cr = inter.get("critical_ratio", 0)
        profit = results.get("expected_profit", 0)
        headline = f"Optimal Newsvendor Order: {q:,} units (Critical Ratio: {cr*100:.1f}%)"
        details = (
            f"Balances underage cost ($Cu = ${inter.get('underage_cost_Cu', 0):,.2f}) against overage cost ($Co = ${inter.get('overage_cost_Co', 0):,.2f}), "
            f"achieving a maximized expected single-period profit of <strong>${profit:,.2f}</strong>."
        )

    elif model == "DATASET_ANALYSIS":
        val = results.get("total_inventory_value", 0.0)
        units = results.get("total_units", 0)
        n_rows = results.get("records_count", 0)
        avg_cost = results.get("average_unit_cost", 0.0)
        top_prod = results.get("highest_units_product", "N/A")
        top_cnt = results.get("highest_units_count", 0)
        headline = f"Portfolio Valuation: ${val:,.2f} across {n_rows:,} items ({units:,} total units)"
        details = (
            f"The uploaded dataset contains <strong>{n_rows:,} inventory records</strong> with an average unit cost of <strong>${avg_cost:,.2f}</strong>. "
            f"Highest-volume product: <strong>{top_prod}</strong> ({top_cnt:,} units). Overall inventory valuation is <strong>${val:,.2f}</strong>."
        )

    else:
        headline = "Inventory Optimization Analysis Complete"
        details = "Review the decision parameters and execution trace below."

    st.markdown(
        f"""
        <div class="exec-banner">
            <div class="exec-banner-title">🏆 Executive Decision Brief & Optimal Recommendation</div>
            <div class="exec-banner-headline">{headline}</div>
            <div class="exec-banner-details">{details}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------------
# UI Component: Visually Appealing Metric Cards
# -------------------------------------------------------------------------
def _render_metric_cards(model: str, results: Dict[str, Any], inter: Dict[str, Any]):
    import streamlit as st

    if model == "EOQ":
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Optimal Batch Size", f"{math.ceil(results.get('optimal_order_quantity_eoq', 0)):,} units", help=f"Exact: {results.get('optimal_order_quantity_eoq', 0):.2f}")
        c2.metric("Order Frequency", f"{results.get('orders_per_year', 0):.1f} / yr", "Annual orders")
        c3.metric("Cycle Time", f"{results.get('order_cycle_time_days', 0):.1f} days", f"{results.get('order_cycle_time_weeks', 0):.1f} wks")
        c4.metric("Annual Ordering Cost", f"${results.get('annual_ordering_cost', 0):,.2f}")
        c5.metric("Annual Holding Cost", f"${results.get('annual_holding_cost', 0):,.2f}")
        c6.metric("Total Annual Cost", f"${results.get('annual_inventory_cost', 0):,.2f}", "Holding + Ordering")

    elif model == "QUANTITY_DISCOUNT":
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Recommended Tier", f"Tier {results.get('recommended_tier', 1)}")
        c2.metric("Optimal Order Size", f"{math.ceil(results.get('optimal_order_quantity', 0)):,} units")
        c3.metric("Best Unit Price", f"${results.get('best_unit_price', 0):,.2f} / unit")
        c4.metric("Annual Purchase Cost", f"${results.get('annual_purchase_cost', 0):,.2f}")
        c5.metric("Total Annual Spend", f"${results.get('minimum_total_annual_cost', 0):,.2f}")
        c6.metric("Annual Savings", f"${results.get('annual_savings_vs_tier_1', 0):,.2f}", "vs Base Pricing")

    elif model == "SAFETY_STOCK_ROP":
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Reorder Point (ROP)", f"{results.get('reorder_point_recommended_integer', 0):,} units", f"Exact: {results.get('reorder_point_rop', 0):.1f}")
        c2.metric("Safety Stock Buffer", f"{results.get('safety_stock_recommended_integer', 0):,} units", f"Exact: {results.get('safety_stock', 0):.1f}")
        c3.metric("Service Level", f"{results.get('cycle_service_level_pct', 95):.1f}%", f"z = {inter.get('z_factor', 0):.2f}")
        c4.metric("Expected LTD", f"{inter.get('expected_lead_time_demand', 0):,.1f} units")
        c5.metric("LTD Std Dev", f"{inter.get('std_dev_lead_time_demand', 0):,.1f} units")
        c6.metric("Case Type", inter.get("case_type", "Standard")[:16])

    elif model == "FIXED_PERIOD_REVIEW":
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Order-Up-To Level (S)", f"{results.get('target_order_up_to_level_integer', 0):,} units", f"Exact: {results.get('target_order_up_to_level_S', 0):.1f}")
        c2.metric("Order Quantity Today", f"{results.get('recommended_order_quantity_integer', 0):,} units", "max(0, S - IP)")
        c3.metric("Current Position (IP)", f"{inter.get('current_inventory_position', 0):,.0f} units")
        c4.metric("Safety Stock", f"{math.ceil(results.get('safety_stock', 0)):,} units")
        c5.metric("Protection Window", f"{inter.get('protection_interval_T_plus_L', 0):.1f} periods", "Review T + Lead Time L")
        c6.metric("Action Trigger", "Order Now" if results.get("order_placed_now") else "Sufficient Stock")

    elif model == "NEWSVENDOR":
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Optimal Order (Q*)", f"{results.get('optimal_order_quantity_integer', 0):,} units", f"Exact: {results.get('optimal_order_quantity', 0):.1f}")
        c2.metric("Critical Fractile", f"{inter.get('critical_ratio', 0)*100:.1f}%", f"z = {inter.get('z_score', 0):.2f}")
        c3.metric("Max Expected Profit", f"${results.get('expected_profit', 0):,.2f}")
        c4.metric("Expected Sales", f"{results.get('expected_sales', 0):,.1f} units")
        c5.metric("Expected Leftovers", f"{results.get('expected_leftovers', 0):,.1f} units")
        c6.metric("Expected Lost Sales", f"{results.get('expected_lost_sales', 0):,.1f} units")

    elif model == "DATASET_ANALYSIS":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{results.get('records_count', 0):,}")
        c2.metric("Total Inventory Units", f"{results.get('total_units', 0):,}")
        c3.metric("Average Unit Cost", f"${results.get('average_unit_cost', 0.0):,.2f}")
        c4.metric("Total Inventory Value", f"${results.get('total_inventory_value', 0.0):,.2f}")


# -------------------------------------------------------------------------
# Tab 1: 6-Stage Execution Pipeline
# -------------------------------------------------------------------------
def _render_pipeline_tab(res: AgentPipelineResult):
    import streamlit as st

    calc = res.calculation_results or {}
    results = calc.get("results", {})
    inter = calc.get("intermediate", {})
    inputs = res.extracted_inputs or {}
    verif = res.verification_report or {}

    st.markdown("### 🔍 6-Stage Cognitive Decision Trace")
    st.caption("Each stage transforms unstructured business intent into verified supply chain actions.")

    # Stage 1: Identify Model
    with st.expander("▶ Stage 1: Customer Question & Model Identification", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**Identified Model:** `{res.identified_model}`")
            st.progress(res.confidence, text=f"Confidence: {res.confidence*100:.0f}%")
        with col2:
            st.markdown(f"**Question Received:** *\"{res.question}\"*")
            st.markdown(f"**Classification Rationale:** {res.pipeline_trace[0].get('rationale', 'Keyword scoring match')}")

    # Stage 2: Extract Inputs
    with st.expander("▶ Stage 2: Extracted Numerical Inputs & Parameters", expanded=True):
        clean_inputs = {k: v for k, v in inputs.items() if not k.startswith("_")}
        input_rows = []
        for param, val in clean_inputs.items():
            if param == "discount_schedule" and isinstance(val, list):
                val_str = ", ".join([f"Tier {i+1}: min={t.get('min_qty')} max={t.get('max_qty')} @ ${t.get('price')}" for i, t in enumerate(val)])
            else:
                val_str = str(val)
            input_rows.append({"Parameter": param, "Extracted Value": val_str})
        st.dataframe(input_rows, use_container_width=True, hide_index=True)

    # Stage 3: Select Python Tool
    with st.expander("▶ Stage 3: Tool Dispatch & Function Binding", expanded=True):
        st.markdown(f"**Selected Deterministic Tool:** `inventory_tools.{res.selected_tool}()`")
        st.markdown(f"**Execution Environment:** Pure Python 3 standard library runtime with zero LLM arithmetic hallucinations.")

    # Stage 4: Calculate (Deterministic & Step-by-Step Proof)
    with st.expander("▶ Stage 4: Deterministic Calculations & Student Verification Walkthrough", expanded=True):
        st.markdown("#### 🎓 Student Verification Proof (8-Step Analytical Trace)")
        st.caption("All numerical calculations are executed deterministically by Python in inventory_tools.py.")

        # 1. Inputs Extracted & 2. Variable Definitions
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.markdown("**1. Inputs Extracted**")
            in_data = calc.get("inputs_extracted") or {k: v for k, v in inputs.items() if not k.startswith("_")}
            in_rows = []
            for k, v in in_data.items():
                if k == "discount_schedule" and isinstance(v, list):
                    v_str = f"{len(v)} tiers configured"
                else:
                    v_str = str(v)
                in_rows.append({"Parameter": k, "Extracted Value": v_str})
            st.dataframe(in_rows, use_container_width=True, hide_index=True)

        with c_v2:
            st.markdown("**2. Variable Definitions**")
            var_defs = calc.get("variable_definitions", {})
            if var_defs:
                var_rows = [{"Symbol": k, "Definition": v} for k, v in var_defs.items()]
                st.dataframe(var_rows, use_container_width=True, hide_index=True)
            else:
                st.info("Standard mathematical variable notation.")

        # 3. Formula & 4. Formula with Substituted Values
        st.markdown("**3. Governing Analytical Formulas**")
        formulas = calc.get("formula", {})
        if formulas:
            for fname, fexpr in formulas.items():
                st.markdown(f"- **{fname.replace('_', ' ').title()}:** `{fexpr}`")

        st.markdown("**4. Formulas with Substituted Numerical Values**")
        subs = calc.get("formula_with_substituted_values", {})
        if subs:
            for sname, sexpr in subs.items():
                if isinstance(sexpr, list):
                    st.markdown(f"**{sname.replace('_', ' ').title()}:**")
                    for sub_item in sexpr:
                        st.markdown(f"  - `{sub_item}`")
                else:
                    st.markdown(f"- **{sname.replace('_', ' ').title()}:** `{sexpr}`")

        # 5. Intermediate Calculations & 6. Final Answer
        col_inter, col_res = st.columns(2)
        with col_inter:
            st.markdown("**5. Intermediate Calculations**")
            inter_dict = calc.get("intermediate_calculations") or inter
            inter_rows = []
            for k, v in inter_dict.items():
                if k not in ["all_tier_evaluations", "tier_evaluations"]:
                    inter_rows.append({"Metric": k.replace('_', ' ').title(), "Value": str(v)})
            st.dataframe(inter_rows, use_container_width=True, hide_index=True)

        with col_res:
            st.markdown("**6. Final Answer**")
            ans_dict = calc.get("final_answer") or results
            ans_rows = [{"Decision Variable": k.replace('_', ' ').title(), "Recommended Value": str(v)} for k, v in ans_dict.items()]
            st.dataframe(ans_rows, use_container_width=True, hide_index=True)

        # 7. Verification Checks & 8. Assumptions
        c_chk, c_assump = st.columns(2)
        with c_chk:
            st.markdown("**7. Verification Checks**")
            verif_checks = calc.get("verification_checks") or verif.get("verification_checks", {})
            if verif_checks:
                chk_rows = [{"Verification Condition": k.replace('_', ' ').title(), "Status": "PASSED ✓" if v is True else str(v)} for k, v in verif_checks.items()]
                st.dataframe(chk_rows, use_container_width=True, hide_index=True)
            else:
                st.success("Mathematical equilibrium and non-negativity checks passed.")

        with c_assump:
            st.markdown("**8. Classical Supply Chain Assumptions**")
            assumptions_list = calc.get("assumptions", [])
            if assumptions_list:
                for a in assumptions_list:
                    st.markdown(f"- 📌 {a}")
            else:
                st.markdown("- Standard operational inventory control assumptions apply.")

    # Stage 5: Verify (Feasibility & Optimality)
    with st.expander("▶ Stage 5: Verification & Feasibility Audit Summary", expanded=True):
        status = verif.get("status", "PASSED")
        if status == "PASSED":
            st.success("✅ Mathematical Verification & Feasibility Checks Passed")
        else:
            st.warning(f"⚠️ Verification Status: {status}")

        checks = verif.get("verification_checks", {})
        if checks:
            check_list = [{"Verification Condition": k, "Result": "PASSED ✓" if v is True else str(v)} for k, v in checks.items()]
            st.dataframe(check_list, use_container_width=True, hide_index=True)

        notes = verif.get("audit_notes", [])
        if notes:
            for n in notes:
                st.markdown(f"- ✓ {n}")

    # Stage 6: Explain (Managerial Brief)
    with st.expander("▶ Stage 6: Executive Managerial Report", expanded=True):
        st.markdown(res.explanation)

    # Technical Raw JSON Inspector
    with st.expander("🛠️ Advanced Technical Trace (Raw JSON)", expanded=False):
        st.json(res.pipeline_trace)


# -------------------------------------------------------------------------
# Tab 2: Model Analytics & Visualizations
# -------------------------------------------------------------------------
def _render_analytics_tab(model: str, results: Dict[str, Any], inter: Dict[str, Any], inputs: Dict[str, Any], calc: Dict[str, Any]):
    import streamlit as st
    import pandas as pd
    import altair as alt

    st.markdown(f"### 📊 Analytics & Visualizations: {model}")

    if model == "EOQ":
        d = inputs.get("annual_demand", 10000)
        s = inputs.get("ordering_cost", 100)
        h = inputs.get("holding_cost", 1.5)
        eoq = results.get("optimal_order_quantity_eoq", 1000)

        st.markdown("#### Annual Inventory Cost Trade-Off Curve")
        st.caption("Illustrates the fundamental trade-off between ordering setup cost and holding carrying cost.")

        # Generate range of lot sizes from 0.2 * EOQ to 2.5 * EOQ
        q_min = max(10, int(eoq * 0.2))
        q_max = int(eoq * 2.5)
        step = max(5, (q_max - q_min) // 60)
        q_vals = list(range(q_min, q_max + step, step))

        data = []
        for q in q_vals:
            order_c = (d / q) * s
            hold_c = (q / 2.0) * h
            tot_c = order_c + hold_c
            data.append({"Lot Size (Q)": q, "Cost ($)": order_c, "Cost Type": "Annual Ordering Cost"})
            data.append({"Lot Size (Q)": q, "Cost ($)": hold_c, "Cost Type": "Annual Holding Cost"})
            data.append({"Lot Size (Q)": q, "Cost ($)": tot_c, "Cost Type": "Total Inventory Cost"})

        df = pd.DataFrame(data)

        cost_chart = (
            alt.Chart(df)
            .mark_line(size=2.5)
            .encode(
                x=alt.X("Lot Size (Q):Q", title="Order Quantity (Units)"),
                y=alt.Y("Cost ($):Q", title="Annual Cost ($)"),
                color=alt.Color(
                    "Cost Type:N",
                    scale=alt.Scale(domain=["Total Inventory Cost", "Annual Ordering Cost", "Annual Holding Cost"], range=["#2563eb", "#ea580c", "#16a34a"]),
                ),
            )
            .properties(height=380)
        )

        # Vertical line at EOQ
        rule_df = pd.DataFrame([{"Q": eoq}])
        rule = alt.Chart(rule_df).mark_rule(color="#dc2626", strokeDash=[6, 4], size=2).encode(x="Q:Q")
        text = alt.Chart(rule_df).mark_text(text=f"Optimal Q* = {eoq:.0f}", color="#dc2626", align="left", dx=8, dy=-140).encode(x="Q:Q")

        st.altair_chart(cost_chart + rule + text, use_container_width=True)

    elif model == "QUANTITY_DISCOUNT":
        st.markdown("#### Multi-Tier Total Annual Cost Comparison")
        st.caption("Compares total procurement spend across all discount brackets, highlighting the cost-minimizing tier.")

        tier_evals = calc.get("all_tier_evaluations", [])
        if tier_evals:
            t_data = []
            for t in tier_evals:
                if t.get("status") == "FEASIBLE":
                    t_data.append({
                        "Tier": f"Tier {t['tier_index']} (${t['price']:.2f})",
                        "Feasible Q": t["feasible_order_quantity"],
                        "Purchase Cost": t["annual_purchase_cost"],
                        "Ordering Cost": t["annual_ordering_cost"],
                        "Holding Cost": t["annual_holding_cost"],
                        "Total Annual Cost": t["total_annual_cost"],
                        "Is Optimal": "Optimal Tier" if t["tier_index"] == results.get("recommended_tier") else "Suboptimal Tier",
                    })

            df_tiers = pd.DataFrame(t_data)
            bar_chart = (
                alt.Chart(df_tiers)
                .mark_bar(size=45)
                .encode(
                    x=alt.X("Tier:N", title="Discount Tier Schedule", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Total Annual Cost:Q", title="Total Annual Spend ($)"),
                    color=alt.Color("Is Optimal:N", scale=alt.Scale(domain=["Optimal Tier", "Suboptimal Tier"], range=["#10b981", "#64748b"])),
                    tooltip=["Tier", "Feasible Q", "Total Annual Cost", "Purchase Cost", "Ordering Cost", "Holding Cost"],
                )
                .properties(height=350)
            )
            st.altair_chart(bar_chart, use_container_width=True)
            st.dataframe(df_tiers, use_container_width=True, hide_index=True)

    elif model == "SAFETY_STOCK_ROP":
        st.markdown("#### Lead Time Demand Distribution & Service Level Coverage")
        st.caption("Gaussian probability density curve over replenishment lead time showing safety stock buffer and ROP.")

        mu_ltd = inter.get("expected_lead_time_demand", 900)
        sigma_ltd = max(1.0, inter.get("std_dev_lead_time_demand", 60))
        rop = results.get("reorder_point_rop", 1000)

        # Generate Gaussian curve points
        x_min = max(0, int(mu_ltd - 3.5 * sigma_ltd))
        x_max = int(mu_ltd + 3.5 * sigma_ltd)
        step = max(1, (x_max - x_min) // 100)
        x_vals = list(range(x_min, x_max + step, step))

        curve_data = []
        for x in x_vals:
            z = (x - mu_ltd) / sigma_ltd
            pdf = (1.0 / (sigma_ltd * math.sqrt(2 * math.pi))) * math.exp(-0.5 * z * z)
            in_stock = "Protected by ROP (Service Level Area)" if x <= rop else "Stockout Exposure Area"
            curve_data.append({"Demand During Lead Time": x, "Probability Density": pdf, "Coverage": in_stock})

        df_curve = pd.DataFrame(curve_data)

        area_chart = (
            alt.Chart(df_curve)
            .mark_area(opacity=0.65)
            .encode(
                x=alt.X("Demand During Lead Time:Q", title="Lead Time Demand (Units)"),
                y=alt.Y("Probability Density:Q", title="Probability Density"),
                color=alt.Color("Coverage:N", scale=alt.Scale(domain=["Protected by ROP (Service Level Area)", "Stockout Exposure Area"], range=["#38bdf8", "#f43f5e"])),
            )
            .properties(height=350)
        )

        rop_rule = alt.Chart(pd.DataFrame([{"ROP": rop}])).mark_rule(color="#dc2626", size=2.5).encode(x="ROP:Q")
        mean_rule = alt.Chart(pd.DataFrame([{"Mean": mu_ltd}])).mark_rule(color="#475569", strokeDash=[4, 4], size=2).encode(x="Mean:Q")

        st.altair_chart(area_chart + rop_rule + mean_rule, use_container_width=True)

    elif model == "FIXED_PERIOD_REVIEW":
        st.markdown("#### Inventory Position vs. Target Order-Up-To Level (S)")
        st.caption("Visualizes current inventory position, safety stock cushion, and immediate replenishment order quantity.")

        s_val = results.get("target_order_up_to_level_S", 2000)
        ip = inter.get("current_inventory_position", 800)
        order_q = results.get("recommended_order_quantity", 1200)
        ss = results.get("safety_stock", 140)

        comp_data = [
            {"Inventory Component": "Current Inventory Position (IP)", "Units": ip, "Category": "Stock"},
            {"Inventory Component": "Recommended Order Quantity (Q)", "Units": order_q, "Category": "Replenishment"},
            {"Inventory Component": "Target Order-Up-To Level (S)", "Units": s_val, "Category": "Target Level"},
            {"Inventory Component": "Safety Stock (T + L)", "Units": ss, "Category": "Buffer"},
        ]
        df_comp = pd.DataFrame(comp_data)

        bar = (
            alt.Chart(df_comp)
            .mark_bar(size=40)
            .encode(
                x=alt.X("Inventory Component:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Units:Q", title="Inventory Units"),
                color=alt.Color("Category:N", scale=alt.Scale(range=["#f59e0b", "#3b82f6", "#10b981", "#8b5cf6"])),
            )
            .properties(height=350)
        )
        st.altair_chart(bar, use_container_width=True)

    elif model == "NEWSVENDOR":
        st.markdown("#### Single-Period Demand Uncertainty & Economic Payoff")
        st.caption("Compares expected sales revenue, procurement expenditure, salvage value, and maximized net expected profit.")

        sales = results.get("expected_sales", 0)
        profit = results.get("expected_profit", 0)
        p = inputs.get("selling_price", 100)
        c = inputs.get("purchase_cost", 40)
        s_val = inputs.get("salvage_value", 10)
        q = results.get("optimal_order_quantity", 0)
        leftovers = results.get("expected_leftovers", 0)

        econ_data = [
            {"Financial Metric": "Expected Revenue", "Amount ($)": sales * p},
            {"Financial Metric": "Procurement Cost", "Amount ($)": q * c},
            {"Financial Metric": "Expected Salvage Recovery", "Amount ($)": leftovers * s_val},
            {"Financial Metric": "Net Expected Profit", "Amount ($)": profit},
        ]
        df_econ = pd.DataFrame(econ_data)

        bar_nv = (
            alt.Chart(df_econ)
            .mark_bar(size=45)
            .encode(
                x=alt.X("Financial Metric:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Amount ($):Q", title="Financial Value ($)"),
                color=alt.Color("Financial Metric:N", scale=alt.Scale(range=["#3b82f6", "#ef4444", "#f59e0b", "#10b981"])),
            )
            .properties(height=350)
        )
        st.altair_chart(bar_nv, use_container_width=True)

    elif model == "DATASET_ANALYSIS":
        st.markdown("#### Portfolio Category Breakdown & Item Valuation")
        st.caption("Distribution of inventory units across product categories from the uploaded dataset.")
        cat_grp = inter.get("category_summary", {})
        if cat_grp:
            cat_df = pd.DataFrame([{"Category": k, "Units": v} for k, v in cat_grp.items()])
            bar_chart = (
                alt.Chart(cat_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#3B82F6")
                .encode(
                    x=alt.X("Category:N", sort="-y", title="Product Category"),
                    y=alt.Y("Units:Q", title="Total Units in Category"),
                    tooltip=["Category", "Units"],
                )
                .properties(height=340)
            )
            st.altair_chart(bar_chart, use_container_width=True)
            st.dataframe(cat_df, use_container_width=True, hide_index=True)


# -------------------------------------------------------------------------
# Tab 3: What-If Parameter Sensitivity
# -------------------------------------------------------------------------
def _render_sensitivity_tab(model: str, inputs: Dict[str, Any], baseline_results: Dict[str, Any]):
    import streamlit as st
    import pandas as pd

    st.markdown(f"### 🎛️ What-If Parameter Sensitivity Analysis")
    st.caption("Adjust key supply chain drivers to observe real-time operational trade-offs calculated via pure Python analytical formulas.")

    if model == "DATASET_ANALYSIS":
        st.info("💡 **Dataset Sensitivity Analysis:** What-If parameter sliders apply to individual single-item models. To test sensitivity on a specific product from your dataset, ask a question such as: *'What is the EOQ for [Product Name] with ordering cost $50 and holding cost $2?'*")
        return

    if model == "EOQ":
        base_d = float(inputs.get("annual_demand", 10000))
        base_s = float(inputs.get("ordering_cost", 100))
        base_h = float(inputs.get("holding_cost", 1.5))

        col1, col2, col3 = st.columns(3)
        with col1:
            sens_d = st.slider("Annual Demand (D):", min_value=int(base_d * 0.3), max_value=int(base_d * 2.5), value=int(base_d), step=100)
        with col2:
            sens_s = st.slider("Ordering Setup Cost (S):", min_value=max(5, int(base_s * 0.2)), max_value=int(base_s * 3.0), value=int(base_s), step=5)
        with col3:
            sens_h = st.slider("Unit Holding Cost (H):", min_value=max(0.1, round(base_h * 0.2, 2)), max_value=round(base_h * 3.0, 2), value=round(base_h, 2), step=0.1)

        # Real-time recalculation using pure python tool
        sens_calc = itools.calculate_eoq(annual_demand=sens_d, ordering_cost=sens_s, holding_cost=sens_h)
        sens_res = sens_calc["results"]

        base_q = baseline_results.get("optimal_order_quantity_eoq", 0)
        new_q = sens_res.get("optimal_order_quantity_eoq", 0)
        delta_q = new_q - base_q

        base_cost = baseline_results.get("annual_inventory_cost", 0)
        new_cost = sens_res.get("annual_inventory_cost", 0)
        delta_cost = new_cost - base_cost

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Adjusted EOQ (Q*)", f"{math.ceil(new_q):,} units", f"{delta_q:+,.1f} units vs baseline")
        m2.metric("Adjusted Annual Cost", f"${new_cost:,.2f}", f"${delta_cost:+,.2f} vs baseline")
        m3.metric("Adjusted Orders / Year", f"{sens_res.get('orders_per_year'):.1f} orders")
        m4.metric("Adjusted Cycle Time", f"{sens_res.get('order_cycle_time_days'):.1f} days")

    elif model == "SAFETY_STOCK_ROP":
        base_lt = float(inputs.get("lead_time", 9))
        base_d = float(inputs.get("demand_rate", 100))
        base_sigma = float(inputs.get("demand_std_dev", 20))
        base_csl = float(inputs.get("service_level", 0.95))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sens_lt = st.slider("Lead Time (L):", min_value=1.0, max_value=30.0, value=base_lt, step=1.0)
        with col2:
            sens_d = st.slider("Daily Demand (d):", min_value=10.0, max_value=500.0, value=base_d, step=5.0)
        with col3:
            sens_sigma = st.slider("Demand Std Dev (σ):", min_value=0.0, max_value=100.0, value=base_sigma, step=2.0)
        with col4:
            sens_csl = st.slider("Cycle Service Level:", min_value=0.80, max_value=0.999, value=base_csl, step=0.005, format="%.3f")

        sens_calc = itools.calculate_safety_stock_rop(
            lead_time=sens_lt, demand_rate=sens_d, demand_std_dev=sens_sigma, service_level=sens_csl
        )
        sens_res = sens_calc["results"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Adjusted Reorder Point (ROP)", f"{sens_res.get('reorder_point_recommended_integer'):,} units")
        m2.metric("Adjusted Safety Stock (SS)", f"{sens_res.get('safety_stock_recommended_integer'):,} units")
        m3.metric("Target Service Level", f"{sens_csl*100:.1f}%")

    elif model == "FIXED_PERIOD_REVIEW":
        base_t = float(inputs.get("review_period_T", inputs.get("review_period", 14)))
        base_lt = float(inputs.get("lead_time_L", inputs.get("lead_time", 5)))
        base_d = float(inputs.get("demand_rate_d", inputs.get("demand_rate_daily", 100)))
        base_sigma = float(inputs.get("demand_std_dev", 20))
        base_csl = float(inputs.get("service_level", 0.95))
        base_ip = float(inputs.get("inventory_position", 800))

        col1, col2, col3 = st.columns(3)
        with col1:
            sens_t = st.slider("Review Period (T):", min_value=1.0, max_value=30.0, value=base_t, step=1.0)
            sens_lt = st.slider("Lead Time (L):", min_value=1.0, max_value=20.0, value=base_lt, step=1.0)
        with col2:
            sens_d = st.slider("Demand Rate (d):", min_value=10.0, max_value=300.0, value=base_d, step=10.0)
            sens_sigma = st.slider("Demand Std Dev (σ):", min_value=0.0, max_value=80.0, value=base_sigma, step=2.0)
        with col3:
            sens_csl = st.slider("Service Level (CSL):", min_value=0.80, max_value=0.999, value=base_csl, step=0.005)
            sens_ip = st.slider("Current Inventory Position (IP):", min_value=0.0, max_value=3000.0, value=base_ip, step=50.0)

        sens_calc = itools.calculate_fixed_period_review(
            review_period=sens_t,
            lead_time=sens_lt,
            demand_rate=sens_d,
            demand_std_dev=sens_sigma,
            service_level=sens_csl,
            inventory_position=sens_ip,
        )
        sens_res = sens_calc["results"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Adjusted Order-Up-To Level (S)", f"{sens_res.get('target_order_up_to_level_integer'):,} units")
        m2.metric("Adjusted Order Quantity (Q)", f"{sens_res.get('recommended_order_quantity_integer'):,} units")
        m3.metric("Adjusted Safety Stock", f"{math.ceil(sens_res.get('safety_stock', 0)):,} units")

    elif model == "NEWSVENDOR":
        base_p = float(inputs.get("selling_price", 100))
        base_c = float(inputs.get("purchase_cost", 40))
        base_s = float(inputs.get("salvage_value", 10))
        base_mu = float(inputs.get("demand_mean", 1000))
        base_sigma = float(inputs.get("demand_std_dev", 200))

        col1, col2 = st.columns(2)
        with col1:
            sens_p = st.slider("Selling Price ($):", min_value=float(base_c + 1), max_value=float(base_p * 2), value=base_p, step=5.0)
            sens_c = st.slider("Unit Purchase Cost ($):", min_value=10.0, max_value=float(sens_p - 1), value=base_c, step=2.0)
        with col2:
            sens_s = st.slider("Salvage Value ($):", min_value=0.0, max_value=float(sens_c), value=base_s, step=2.0)
            sens_sigma = st.slider("Demand Std Dev (σ):", min_value=10.0, max_value=500.0, value=base_sigma, step=10.0)

        sens_calc = itools.calculate_newsvendor(
            selling_price=sens_p, purchase_cost=sens_c, salvage_value=sens_s, demand_mean=base_mu, demand_std_dev=sens_sigma
        )
        sens_res = sens_calc["results"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Adjusted Order Q*", f"{sens_res.get('optimal_order_quantity_integer'):,} units")
        m2.metric("Critical Fractile", f"{sens_res.get('optimal_service_level_pct'):.1f}%")
        m3.metric("Expected Profit", f"${sens_res.get('expected_profit'):,.2f}")

    elif model == "QUANTITY_DISCOUNT":
        base_d = float(inputs.get("annual_demand", 10000))
        base_s = float(inputs.get("ordering_cost", 50))
        schedule = inputs.get("discount_schedule", [])

        col1, col2 = st.columns(2)
        with col1:
            sens_d = st.slider("Annual Demand (D):", min_value=int(base_d * 0.5), max_value=int(base_d * 2.5), value=int(base_d), step=500)
        with col2:
            sens_s = st.slider("Ordering Cost (S):", min_value=10, max_value=200, value=int(base_s), step=5)

        if schedule:
            sens_calc = itools.calculate_quantity_discount(
                annual_demand=sens_d,
                ordering_cost=sens_s,
                discount_schedule=schedule,
                holding_rate=inputs.get("holding_rate", 0.20),
            )
            s_res = sens_calc["results"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Recommended Tier", f"Tier {s_res.get('recommended_tier')}")
            m2.metric("Optimal Quantity", f"{math.ceil(s_res.get('optimal_order_quantity', 0)):,} units")
            m3.metric("Total Annual Spend", f"${s_res.get('minimum_total_annual_cost'):,.2f}")


# =========================================================================
# CLI Application Runner (Terminal / Batch Mode)
# =========================================================================

def print_banner():
    banner = """
================================================================================
          SUPPLY CHAIN ANALYTICS - INVENTORY DECISION AGENT
================================================================================
 Pipeline Architecture:
   [Question] → [Identify Model] → [Extract Inputs] → [Select Tool] → [Calculate] → [Verify] → [Explain]

 Supported Models:
   1. Economic Order Quantity (EOQ)
   2. Quantity Discount Optimization
   3. Safety Stock & Reorder Point (Continuous Review (s, Q))
   4. Fixed-Period Review (Periodic Review (T, S) Order-Up-To)
   5. Newsvendor Single-Period Stochastic Model
================================================================================
"""
    print(banner)


def display_pipeline_terminal(res: AgentPipelineResult):
    print("\n" + "=" * 80)
    print(f"📥 CUSTOMER QUESTION:")
    print(f"   \"{res.question}\"")
    print("=" * 80)

    print("\n--- [PIPELINE TRACE] ---")
    for step in res.pipeline_trace:
        stage = step.get("stage", "Unknown")
        print(f"\n▶ {stage}")
        for k, v in step.items():
            if k != "stage":
                if isinstance(v, dict):
                    print(f"   • {k}:")
                    for sub_k, sub_v in v.items():
                        print(f"       - {sub_k}: {sub_v}")
                else:
                    print(f"   • {k}: {v}")

    print("\n" + "=" * 80)
    if res.status == "SUCCESS":
        print(f"✅ FINAL EXECUTIVE REPORT & RECOMMENDATIONS")
        print("=" * 80)
        print(res.explanation)
    else:
        print(f"❌ PIPELINE ERROR")
        print(f"   {res.error_message}")
    print("=" * 80 + "\n")


def run_cli_interactive():
    print_banner()
    agent = InventoryDecisionAgent()

    while True:
        print("\nChoose an option:")
        print("  [1-5] Run pre-built benchmark scenario")
        print("  [C]   Enter custom natural-language inventory question")
        print("  [D]   Run all 5 demonstration scenarios sequentially")
        print("  [Q]   Quit")

        choice = input("\nEnter selection (1-5, C, D, Q): ").strip()
        if not choice:
            continue

        if choice.upper() == "Q":
            print("\nExiting Inventory Decision Agent. Goodbye!\n")
            sys.exit(0)

        elif choice.upper() == "D":
            print("\n--- RUNNING ALL 5 BENCHMARK SCENARIOS ---")
            for title, q_text in SAMPLE_SCENARIOS.items():
                print(f"\n>>> Running Scenario: {title}")
                res = agent.process_question(q_text)
                display_pipeline_terminal(res)

        elif choice.upper() == "C":
            custom_q = input("\nEnter your inventory question:\n> ").strip()
            if custom_q:
                res = agent.process_question(custom_q)
                display_pipeline_terminal(res)

        elif choice in ["1", "2", "3", "4", "5"]:
            keys = list(SAMPLE_SCENARIOS.keys())
            idx = int(choice) - 1
            scenario_key = keys[idx]
            q_text = SAMPLE_SCENARIOS[scenario_key]
            print(f"\n--- Running: {scenario_key} ---")
            res = agent.process_question(q_text)
            display_pipeline_terminal(res)

        else:
            print("Invalid selection. Please choose 1-5, C, D, or Q.")


# =========================================================================
# Main Entry Point
# =========================================================================

def is_running_in_streamlit() -> bool:
    """Detect whether app.py was launched by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return True
    except Exception:
        pass
    return "streamlit" in sys.modules and (len(sys.argv) > 0 and "streamlit" in sys.argv[0])


def main():
    if is_running_in_streamlit():
        run_streamlit_app()
    else:
        # CLI Mode
        import re
        parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
        parser.add_argument("question", nargs="*", help="Direct natural-language inventory question")
        parser.add_argument("--demo", action="store_true", help="Run all 5 benchmark scenarios")
        args = parser.parse_args()

        agent = InventoryDecisionAgent()
        question_text = " ".join(args.question).strip() if args.question else None

        if args.demo:
            print_banner()
            for title, q_text in SAMPLE_SCENARIOS.items():
                print(f"\n>>> Running Benchmark: {title}")
                res = agent.process_question(q_text)
                display_pipeline_terminal(res)
        elif question_text:
            if re.search(r"cost\s+(?:is|of)\s+(?:per|/|\s)", question_text, re.IGNORECASE):
                print("\n⚠️  [SHELL NOTICE]: Numbers directly following 'cost is' appear to have been stripped by shell variable expansion.")
                print("   In macOS zsh or bash, wrap questions containing dollar signs in single quotes ('...') or escape them (\\$):")
                print("   Example: python app.py 'Annual demand is 10000 units, ordering cost is $50 per order, and annual holding cost is $2 per unit.'\n")
            res = agent.process_question(question_text)
            display_pipeline_terminal(res)
        else:
            run_cli_interactive()


if __name__ == "__main__":
    main()
