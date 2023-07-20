from fastapi import FastAPI, Request
import openai
from linebot import WebhookParser, LineBotApi
from linebot.models import TextSendMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import os

# Constants
OPENAI_API_KEY = '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'
LINE_CHANNEL_ACCESS_TOKEN = 'vtNEjP6IvEOZy/kGGKQ4trYobJ7cx2khewDnigkqXq9MsiqGeuk94AVQ4XckF12O/62oawSQaJqC+zrZ2DDEVOXI+Yo5LVxoSlm6XnsQD9UrQn30wDEgeJm6VuTTmWxrEAQRkdAsqetSNTeXzIjvuQdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '3971a771a03a8d6a9f8ee09f38a4ce94'
OPENAI_CHARACTER_PROFILE = 'これから会話を行います。以下の条件を絶対に守って回答してください。敬語は使わない'
# Set API key
openai.api_key = OPENAI_API_KEY

# Initialize LINE Bot
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
line_parser = WebhookParser(LINE_CHANNEL_SECRET)
app = FastAPI()

# Load PDF using langchain
pdf_file_path = 'C:\\Users\\User\\Desktop\\chatbot\\生活保護運用事例 集 2017（令和3年6月改訂版）.pdf'
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

# Preprocess PDF
texts = load_pdf('生活保護運用事例 集 2017（令和3年6月改訂版）.pdf')
texts = [page.page_content for doc in texts for page in doc]

# Create QA chain using langchain (No need for Chroma)
embeddings = OpenAIEmbeddings()
qa = ChatOpenAI(model_name="gpt-3.5-turbo")

# Handle incoming messages and generate responses
@app.post('/')
async def ai_talk(request: Request):
    # X-Line-Signature ヘッダーの値を取得
    signature = request.headers.get('X-Line-Signature', '')

    # request body から event オブジェクトを取得
    events = line_parser.parse((await request.body()).decode('utf-8'), signature)

    # 各イベントの処理（※1つの Webhook に複数の Webhook イベントオブジェっｚクトが含まれる場合あるため）
    for event in events:
        if event.type != 'message':
            continue
        if event.message.type != 'text':
            continue

        # LINE パラメータの取得
        line_user_id = event.source.user_id
        line_message = event.message.text

        # ChatGPT からトークデータを取得
        response = qa.run(line_message, system_message=OPENAI_CHARACTER_PROFILE)
        ai_message = response["response"]

        # LINE メッセージの送信
        line_bot_api.push_message(line_user_id, TextSendMessage(ai_message))

    # LINE Webhook サーバーへ HTTP レスポンスを返す
    return 'ok'