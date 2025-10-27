import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL")

def ask_api(message):
    r = requests.post(API_URL + "/ask", json={"question": message})
    data = r.json()
    # Adapte selon ton API, par ex. data["answer"] si c'est le champ renvoyé
    return data if isinstance(data, str) else data.get("answer", str(data))

def on_submit(user_message, chat_history):
    if not user_message:
        return "", chat_history
    bot_reply = ask_api(user_message)
    chat_history = chat_history + [(user_message, bot_reply)]
    return "", chat_history

with gr.Blocks(title="Département du Var - Evènements 2025") as demo:
    gr.Markdown("# Département du Var - Evènements 2025")

    # Message initial du bot
    chatbot = gr.Chatbot(
        value=[(None, "Bonjour, je suis votre assistant pour le département du Var, dédié aux évènements 2025. Comment puis-je vous aider ?")]
    )

    msg = gr.Textbox(placeholder="Posez votre question…", label="Message")
    send = gr.Button("Envoyer")

    # Liaison des événements
    msg.submit(on_submit, [msg, chatbot], [msg, chatbot])
    send.click(on_submit, [msg, chatbot], [msg, chatbot])

demo.launch(server_name="0.0.0.0")