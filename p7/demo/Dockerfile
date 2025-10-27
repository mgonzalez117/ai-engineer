FROM python:3.11
WORKDIR /app
RUN pip install gradio requests
COPY chat.py .
CMD ["python", "chat.py"]