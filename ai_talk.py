# app.py

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
import os
from flask import Flask, request, jsonify
import requests

os.environ["OPENAI_API_KEY"] = "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"

filepath = r'C:\Users\User\Desktop\line chat bot\pdf_data\生活保護運用事例 集 2017（令和3年6月改訂版）.pdf'
loader = PyPDFLoader(filepath)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=vectordb.as_retriever())

from langchain import PromptTemplate

template = """
あなたは親切なアシスタントです。下記の質問に日本語で回答してください。
質問：{question}
回答：
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)

app = Flask(__name__)

@app.route("/", methods=["POST"])
def handle_message():
    data = request.get_json()
    question = data["events"][0]["message"]["text"]
    answer = qa.predict(question)
    reply_token = data["events"][0]["replyToken"]
    send_reply(reply_token, answer)
    return jsonify({})

def send_reply(reply_token, message):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "vtNEjP6IvEOZy/kGGKQ4trYobJ7cx2khewDnigkqXq9MsiqGeuk94AVQ4XckF12O/62oawSQaJqC+zrZ2DDEVOXI+Yo5LVxoSlm6XnsQD9UrQn30wDEgeJm6VuTTmWxrEAQRkdAsqetSNTeXzIjvuQdB04t89/1O/w1cDnyilFU="
    }
    data = {
        "replyToken": reply_token,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }
    requests.post("https://api.line.me/v2/bot/message/reply", headers=headers, json=data)

if __name__ == "__main__":
    app.run(debug=True)