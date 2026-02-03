import os
import time
import requests
import streamlit as st
import gymnasium as gym
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_URL = os.getenv("API_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

st.set_page_config(page_title="LunarLander Control Center", layout="wide")
st.title("🚀 LunarLander AI Dashboard")

tab_live, tab_dash = st.tabs(["🎮 Live Mission", "📊 Dashboard"])

# -------------------------
# TAB 1 — LIVE MISSION
# -------------------------
with tab_live:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Live Mission Feed")
        frame_placeholder = st.empty()
    with col2:
        st.subheader("Telemetry")
        status_text = st.empty()
        reward_chart = st.empty()

    if st.button("🚀 Launch Mission", key="launch"):
        env = gym.make(MODEL_NAME, render_mode="rgb_array")
        obs, _ = env.reset()

        total_reward = 0.0
        rewards = []
        steps = 0
        terminated = False
        truncated = False
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        circumstances = []

        t0 = time.time()
        done = False
        last_action_name = None

        while not done:
            steps += 1

            # 1) API inference
            try:
                res = requests.post(f"{API_URL}/predict", json={"observation": obs.tolist()}, timeout=5)
                res.raise_for_status()
                payload = res.json()
                action = payload["action"]
                last_action_name = payload.get("action_name")
                action_counts[action] += 1
            except Exception as e:
                st.error(f"API /predict error: {e}")
                break

            # Logger les situations critiques (toutes les 10 steps)
            if steps % 10 == 0:
                x, y, vx, vy, angle, angular_vel, left_leg, right_leg = obs
                circumstances.append({
                    "step": steps,
                    "action": action,
                    "action_name": last_action_name,
                    "angle": round(float(angle), 2),
                    "vitesse_verticale": round(float(vy), 2),
                    "vitesse_horizontale": round(float(vx), 2),
                    "altitude": round(float(y), 2)
                })

            # 2) Env step
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            rewards.append(total_reward)

            # 3) Render + UI update
            frame = env.render()
            caption = f"Action: {last_action_name}" if last_action_name else "Action: ?"
            frame_placeholder.image(frame, caption=caption, width=700)

            status_text.metric("Total Reward", f"{total_reward:.2f}")
            reward_chart.line_chart(rewards)

            done = bool(terminated) or bool(truncated)

        duration_s = time.time() - t0
        env.close()

        # Enregistrer l'expérimentation
        try:
            exp_payload = {
                "run_name": "streamlit_demo",
                "nb_steps": steps,
                "total_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
                "duration_s": duration_s,
                "action_0": action_counts[0],
                "action_1": action_counts[1],
                "action_2": action_counts[2],
                "action_3": action_counts[3],
                "circumstances": circumstances
            }
            r = requests.post(f"{API_URL}/experiments", json=exp_payload, timeout=5)
            r.raise_for_status()
            st.success(f"✅ Mission Finished! Final Score: {total_reward:.2f}")
        except Exception as e:
            st.warning(f"⚠️ Mission Finished! Final Score: {total_reward:.2f} (NOT saved: {e})")

# -------------------------
# TAB 2 — DASHBOARD
# -------------------------
with tab_dash:
    st.subheader("📊 Suivi des expérimentations")

    if st.button("🔄 Refresh", key="refresh"):
        st.rerun()

    # Charger les stats
    try:
        r_stats = requests.get(f"{API_URL}/experiments/stats", timeout=5)
        r_stats.raise_for_status()
        stats = r_stats.json()
    except Exception as e:
        st.error(f"Erreur stats: {e}")
        stats = {}

    # Charger la liste des expérimentations
    try:
        r = requests.get(f"{API_URL}/experiments", timeout=5)
        r.raise_for_status()
        exps = r.json().get("experiments", [])
    except Exception as e:
        st.error(f"Erreur liste: {e}")
        exps = []

    if not exps:
        st.info("Aucune expérimentation enregistrée.")
    else:
        # === MÉTRIQUES PRINCIPALES ===
        st.subheader("📈 Statistiques globales")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Nb expérimentations", stats.get("total_experiments", 0))
        with col2:
            avg = stats.get('avg_reward')
            st.metric("Moyenne récompense", f"{avg:.2f}" if avg is not None else "N/A")
        with col3:
            std = stats.get('std_reward')
            st.metric("Écart-type", f"{std:.2f}" if std is not None else "N/A")
        with col4:
            max_r = stats.get('max_reward')
            st.metric("Max récompense", f"{max_r:.2f}" if max_r is not None else "N/A")
        with col5:
            success = stats.get('success_rate')
            st.metric("Taux de succès", f"{success:.1f}%" if success is not None else "N/A")

        st.divider()

        # === COURBE DES RÉCOMPENSES ===
        st.subheader("📉 Évolution des récompenses")
        rewards_data = [x.get("total_reward", 0) for x in exps]
        fig_rewards = go.Figure()
        fig_rewards.add_trace(go.Scatter(
            y=rewards_data,
            mode='lines+markers',
            name='Récompense',
            line=dict(color='royalblue', width=2)
        ))
        fig_rewards.add_hline(y=200, line_dash="dash", line_color="green", annotation_text="Seuil de succès (200)")
        fig_rewards.update_layout(
            xaxis_title="Numéro d'expérimentation",
            yaxis_title="Récompense totale",
            height=400
        )
        st.plotly_chart(fig_rewards, use_container_width=True)

        st.divider()

        # === SÉLECTION D'UNE MISSION ===
        st.subheader("🎯 Analyse d'une mission")

        exp_options = [f"Mission {i + 1} | Score: {exp.get('total_reward', 0):.1f} | Steps: {exp.get('nb_steps', 0)}"
                       for i, exp in enumerate(exps)]
        selected_exp_idx = st.selectbox("Choisir une mission à analyser", range(len(exps)),
                                        format_func=lambda x: exp_options[x])

        selected_exp = exps[selected_exp_idx]

        # === DISTRIBUTION DES ACTIONS (MISSION SÉLECTIONNÉE) ===
        st.write("**Distribution des actions pour cette mission**")

        actions = {
            "Ne rien faire": selected_exp.get("action_0", 0),
            "Moteur gauche": selected_exp.get("action_1", 0),
            "Moteur principal": selected_exp.get("action_2", 0),
            "Moteur droit": selected_exp.get("action_3", 0)
        }

        col_pie, col_table = st.columns([1, 1])

        with col_pie:
            fig_actions = px.pie(
                values=list(actions.values()),
                names=list(actions.keys()),
                hole=0.4
            )
            st.plotly_chart(fig_actions, use_container_width=True)

        with col_table:
            total_actions = sum(actions.values())
            actions_df = pd.DataFrame({
                "Action": list(actions.keys()),
                "Nombre": list(actions.values()),
                "Pourcentage": [f"{v / total_actions * 100:.1f}%" if total_actions > 0 else "0%" for v in
                                actions.values()]
            })
            st.dataframe(actions_df, hide_index=True, use_container_width=True)

        st.divider()

        # === DÉCISIONS EN FONCTION DES CIRCONSTANCES ===
        st.subheader("🧠 Décisions selon les circonstances")

        circumstances = selected_exp.get("circumstances", [])

        if circumstances:
            circ_df = pd.DataFrame(circumstances)

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                # Graphique : Actions en fonction de l'angle
                fig_angle = px.scatter(
                    circ_df,
                    x="angle",
                    y="action_name",
                    color="action_name",
                    title="Actions prises selon l'angle du vaisseau",
                    labels={"angle": "Angle (rad)", "action_name": "Action"}
                )
                st.plotly_chart(fig_angle, use_container_width=True)

            with col_g2:
                # Graphique : Actions en fonction de la vitesse verticale
                fig_vy = px.scatter(
                    circ_df,
                    x="vitesse_verticale",
                    y="action_name",
                    color="action_name",
                    title="Actions prises selon la vitesse verticale",
                    labels={"vitesse_verticale": "Vitesse verticale", "action_name": "Action"}
                )
                st.plotly_chart(fig_vy, use_container_width=True)

            # Tableau détaillé
            st.write("**Détail des circonstances (échantillon toutes les 10 steps)**")
            st.dataframe(circ_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune circonstance enregistrée pour cette mission.")

        st.divider()

        # === TABLEAU DES EXPÉRIMENTATIONS ===
        st.subheader("📋 Historique complet")

        # Formater les données
        for exp in exps:
            if "created_at" in exp and exp["created_at"]:
                dt = datetime.fromtimestamp(exp["created_at"])
                exp["date"] = dt.strftime("%d/%m/%Y %H:%M:%S")
            else:
                exp["date"] = "N/A"

        df = pd.DataFrame(exps)

        # Colonnes à afficher
        display_cols = ["date", "run_name", "total_reward", "nb_steps", "terminated", "truncated"]
        display_cols = [c for c in display_cols if c in df.columns]

        df_display = df[display_cols]

        # Renommer pour plus de clarté
        df_display = df_display.rename(columns={
            "date": "Date",
            "run_name": "Nom",
            "total_reward": "Récompense",
            "nb_steps": "Steps",
            "terminated": "Terminé",
            "truncated": "Timeout"
        })

        st.dataframe(df_display, use_container_width=True, hide_index=True)