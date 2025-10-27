import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL")

def chat(message, history):
    response = requests.post(API_URL+"/ask", json={"question": message})
    return response.json()

gr.ChatInterface(chat).launch(server_name="0.0.0.0")