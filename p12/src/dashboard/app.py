import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# ----------------------------
# Config UI
# ----------------------------
st.set_page_config(
    page_title="P12 • ETL KPI Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 P12 • ETL KPI Dashboard")
st.caption("KPI de pipeline basés sur la table `etl_metrics` (Airflow reste la source de vérité pour l’orchestration).")


# ----------------------------
# DB
# ----------------------------
DB_URL = os.getenv("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
if not DB_URL:
    st.error("Variable d'environnement manquante: AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
    st.stop()

engine = create_engine(DB_URL, pool_pre_ping=True)


@st.cache_data(ttl=30)
def fetch_metrics(days: int) -> pd.DataFrame:
    q = text("""
        SELECT
            created_at,
            pipeline_name,
            step,
            run_id,
            execution_date,
            nb_input,
            nb_output,
            nb_rejected,
            duration_seconds,
            rows_per_second,
            success
        FROM etl_metrics
        WHERE created_at >= (NOW() - (:days || ' days')::interval)
        ORDER BY created_at DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"days": days})

    # Normalisation types (au cas où execution_date arrive en string)
    for col in ["created_at", "execution_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Colonnes dérivées utiles
    df["valid_rate"] = (df["nb_output"] / df["nb_input"]).where(df["nb_input"] > 0)
    df["reject_rate"] = (df["nb_rejected"] / df["nb_input"]).where(df["nb_input"] > 0)

    return df


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Filtres")

days = st.sidebar.slider("Fenêtre d'historique (jours)", min_value=1, max_value=90, value=14, step=1)

df = fetch_metrics(days)

if df.empty:
    st.warning("Aucune métrique trouvée sur la période sélectionnée.")
    st.stop()

pipelines = ["(tous)"] + sorted([p for p in df["pipeline_name"].dropna().unique()])
steps = ["(tous)"] + sorted([s for s in df["step"].dropna().unique()])

pipeline_choice = st.sidebar.selectbox("Pipeline", pipelines, index=0)
step_choice = st.sidebar.selectbox("Step", steps, index=0)

only_success = st.sidebar.checkbox("Afficher uniquement les runs success", value=False)

# Date range (sur created_at, car c’est le timestamp réel d’écriture)
min_dt = df["created_at"].min()
max_dt = df["created_at"].max()
default_start = max(min_dt, max_dt - pd.Timedelta(days=min(7, days)))
date_range = st.sidebar.date_input(
    "Période (created_at)",
    value=(default_start.date(), max_dt.date()),
    min_value=min_dt.date(),
    max_value=max_dt.date(),
)

# ----------------------------
# Apply filters
# ----------------------------
df_f = df.copy()

if pipeline_choice != "(tous)":
    df_f = df_f[df_f["pipeline_name"] == pipeline_choice]

if step_choice != "(tous)":
    df_f = df_f[df_f["step"] == step_choice]

if only_success:
    df_f = df_f[df_f["success"] == True]

start_date, end_date = date_range
start_ts = pd.to_datetime(datetime.combine(start_date, datetime.min.time()))
end_ts = pd.to_datetime(datetime.combine(end_date, datetime.max.time()))
df_f = df_f[(df_f["created_at"] >= start_ts) & (df_f["created_at"] <= end_ts)]

if df_f.empty:
    st.warning("Aucune métrique ne correspond aux filtres.")
    st.stop()


# ----------------------------
# KPI section
# ----------------------------
latest = df_f.sort_values("created_at").iloc[-1]

total_runs = len(df_f)
success_rate = (df_f["success"].mean() * 100.0) if total_runs else 0.0

avg_duration = df_f["duration_seconds"].dropna().mean()
p95_duration = df_f["duration_seconds"].dropna().quantile(0.95) if df_f["duration_seconds"].notna().any() else None

avg_rows_sec = df_f["rows_per_second"].dropna().mean()
avg_valid = df_f["valid_rate"].dropna().mean() * 100.0 if df_f["valid_rate"].notna().any() else None
avg_reject = df_f["reject_rate"].dropna().mean() * 100.0 if df_f["reject_rate"].notna().any() else None

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Runs", f"{total_runs}")
c2.metric("Success rate", f"{success_rate:.1f}%")
c3.metric("Durée moyenne", f"{avg_duration:.2f}s" if pd.notna(avg_duration) else "—")
c4.metric("Débit moyen", f"{avg_rows_sec:.0f} rows/s" if pd.notna(avg_rows_sec) else "—")
c5.metric("Validité moyenne", f"{avg_valid:.1f}%" if avg_valid is not None else "—")

with st.expander("Dernier enregistrement (selon filtres)", expanded=False):
    st.write(
        {
            "created_at": str(latest.get("created_at")),
            "pipeline_name": latest.get("pipeline_name"),
            "step": latest.get("step"),
            "run_id": latest.get("run_id"),
            "nb_input": int(latest["nb_input"]) if pd.notna(latest["nb_input"]) else None,
            "nb_output": int(latest["nb_output"]) if pd.notna(latest["nb_output"]) else None,
            "nb_rejected": int(latest["nb_rejected"]) if pd.notna(latest["nb_rejected"]) else None,
            "duration_seconds": float(latest["duration_seconds"]) if pd.notna(latest["duration_seconds"]) else None,
            "rows_per_second": float(latest["rows_per_second"]) if pd.notna(latest["rows_per_second"]) else None,
            "success": bool(latest.get("success")),
        }
    )


# ----------------------------
# Charts
# ----------------------------
st.subheader("Tendances")

# Sécurise types numériques (au cas où Postgres/driver renvoie object)
num_cols = ["nb_input", "nb_output", "nb_rejected", "duration_seconds", "rows_per_second"]
for c in num_cols:
    if c in df_f.columns:
        df_f[c] = pd.to_numeric(df_f[c], errors="coerce")

# Agrégation par jour (en gardant un datetime, pas un .date)
df_daily = df_f.copy()
df_daily["day"] = df_daily["created_at"].dt.floor("D")

agg = (
    df_daily.groupby("day", as_index=False)
    .agg(
        runs=("run_id", "count"),
        success_rate=("success", "mean"),
        nb_input=("nb_input", "sum"),
        nb_output=("nb_output", "sum"),
        nb_rejected=("nb_rejected", "sum"),
        duration_avg=("duration_seconds", "mean"),
        rows_per_sec_avg=("rows_per_second", "mean"),
    )
    .sort_values("day")
)

agg["success_rate"] = agg["success_rate"] * 100.0
agg["valid_rate"] = (agg["nb_output"] / agg["nb_input"]).where(agg["nb_input"] > 0) * 100.0
agg["reject_rate"] = (agg["nb_rejected"] / agg["nb_input"]).where(agg["nb_input"] > 0) * 100.0

# Pour éviter des graphes "vides" quand 1 seul point : fallback en bar chart
single_point = len(agg) < 2

ch1, ch2 = st.columns(2)
with ch1:
    st.markdown("**Volume & qualité**")
    vol_df = agg.set_index("day")[["nb_input", "nb_output", "nb_rejected"]].fillna(0)
    (st.bar_chart if single_point else st.line_chart)(vol_df)

with ch2:
    st.markdown("**Durée & débit**")
    perf_df = agg.set_index("day")[["duration_avg", "rows_per_sec_avg"]]
    (st.bar_chart if single_point else st.line_chart)(perf_df)

ch3, ch4 = st.columns(2)
with ch3:
    st.markdown("**Taux de succès**")
    sr = agg.set_index("day")[["success_rate"]].clip(lower=0, upper=100)
    (st.bar_chart if single_point else st.line_chart)(sr)

with ch4:
    st.markdown("**Taux validité / rejet**")
    vr = agg.set_index("day")[["valid_rate", "reject_rate"]].clip(lower=0, upper=100)
    (st.bar_chart if single_point else st.line_chart)(vr)



# ----------------------------
# Detailed table
# ----------------------------
st.subheader("Détail des runs (filtrés)")

# Format simple pour lecture
df_table = df_f.copy().sort_values("created_at", ascending=False)

# colonnes “lisibles”
cols = [
    "created_at",
    "pipeline_name",
    "step",
    "run_id",
    "nb_input",
    "nb_output",
    "nb_rejected",
    "duration_seconds",
    "rows_per_second",
    "success",
]
df_table = df_table[cols]

st.dataframe(df_table, use_container_width=True, height=420)

# Export CSV (utile pour ton rendu)
csv = df_table.to_csv(index=False).encode("utf-8")
st.download_button(
    "Télécharger CSV (runs filtrés)",
    data=csv,
    file_name="etl_metrics_filtered.csv",
    mime="text/csv",
)