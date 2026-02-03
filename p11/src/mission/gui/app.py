import os
import time
import requests
import streamlit as st
import gymnasium as gym
from datetime import datetime
import pandas as pd

API_URL = os.getenv("API_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_PATH = os.getenv("MODEL_PATH")

st.set_page_config(page_title="LunarLander Control Center", layout="wide")
st.title("LunarLander AI Dashboard")

tab_live, tab_dash = st.tabs(["Live Mission", "Dashboard (Expérimentations)"])

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

    if st.button("Launch Mission", key="launch"):
        env = gym.make(MODEL_NAME, render_mode="rgb_array")
        obs, _ = env.reset()

        total_reward = 0.0
        rewards = []
        steps = 0
        terminated = False
        truncated = False

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
            except Exception as e:
                st.error(f"API /predict error: {e}")
                break

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

        # Enregistrer l'expérimentation (1 appel à la fin, pas à chaque step)
        try:
            exp_payload = {
                "run_name": st.session_state.get("run_name", "streamlit_demo"),
                "nb_steps": steps,
                "total_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
                "duration_s": duration_s,
                "model_path": os.getenv("MODEL_PATH", None),
            }
            r = requests.post(f"{API_URL}/experiments", json=exp_payload, timeout=5)
            r.raise_for_status()
            st.success(f"Mission Finished! Final Score: {total_reward:.2f} (saved)")
        except Exception as e:
            st.warning(f"Mission Finished! Final Score: {total_reward:.2f} (NOT saved: {e})")

# -------------------------
# TAB 2 — DASHBOARD
# -------------------------
with tab_dash:
    st.subheader("Suivi des expérimentations")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Refresh", key="refresh"):
            st.session_state["refresh_ts"] = time.time()

    # Charger la liste des expérimentations
    try:
        r = requests.get(f"{API_URL}/experiments", timeout=5)
        r.raise_for_status()
        exps = r.json().get("experiments", [])
    except Exception as e:
        st.error(f"API /experiments error: {e}")
        exps = []

    if not exps:
        st.info("Aucune expérimentation enregistrée pour le moment.")
    else:
        # Formater les données pour l'affichage
        for exp in exps:
            # Convertir le timestamp en date lisible
            if "created_at" in exp and exp["created_at"]:
                dt = datetime.fromtimestamp(exp["created_at"])
                exp["date"] = dt.strftime("%d/%m/%Y %H:%M:%S")
            else:
                exp["date"] = "N/A"

        # Créer un DataFrame avec les colonnes dans l'ordre souhaité
        df = pd.DataFrame(exps)

        # Réorganiser les colonnes pour mettre 'date' en premier
        cols = ["date"] + [col for col in df.columns if col not in ["date", "created_at"]]
        df = df[cols]

        # Supprimer la colonne created_at (timestamp brut)
        if "created_at" in df.columns:
            df = df.drop(columns=["created_at"])

        # Afficher le tableau
        st.dataframe(df, use_container_width=True)

        # Stats rapides
        rewards = [x.get("total_reward", 0) for x in exps if x.get("total_reward") is not None]
        if rewards:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nb expérimentations", len(rewards))
            with col2:
                st.metric("Meilleur score", f"{max(rewards):.2f}")
            with col3:
                st.metric("Score moyen", f"{(sum(rewards) / len(rewards)):.2f}")

            # Courbe des scores au fil des runs
            st.line_chart(rewards)