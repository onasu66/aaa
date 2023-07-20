import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import openai
from linebot import WebhookParser, LineBotApi
from linebot.models import TextSendMessage

os.environ["OPENAI_API_KEY"] = "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"

from fastapi import FastAPI, Request

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

app = FastAPI()

line_bot_api = LineBotApi('vtNEjP6IvEOZy/kGGKQ4trYobJ7cx2khewDnigkqXq9MsiqGeuk94AVQ4XckF12O/62oawSQaJqC+zrZ2DDEVOXI+Yo5LVxoSlm6XnsQD9UrQn30wDEgeJm6VuTTmWxrEAQRkdAsqetSNTeXzIjvuQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('3971a771a03a8d6a9f8ee09f38a4ce94')

line_parser = WebhookParser('3971a771a03a8d6a9f8ee09f38a4ce94')  # '3971a771a03a8d6a9f8ee09f38a4ce94'はWebhookのChannel Secretに置き換える

# You MUST add your PDF to local files in this notebook (folder icon on left hand side of screen)

# Simple method - Split by pages 
loader = PyPDFLoader("./生活保護運用事例 集 2017（令和3年6月改訂版）.pdf")
pages = loader.load_and_split()
print(pages[0])

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages

# Advanced method - Split by chunk

# Step 1: Convert PDF to text
import textract
doc = textract.process("./生活保護運用事例 集 2017（令和3年6月改訂版）.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('生活保護運用事例 集 2017（令和3年6月改訂版）.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

with open('生活保護運用事例 集 2017（令和3年6月改訂版）.txt', 'r') as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])


# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
type(chunks[0]) 
langchain.schema.Document

# Quick data visualization to ensure chunking was successful

# Create a list of token counts
token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

# Create a DataFrame from the token counts
df = pd.DataFrame({'Token Count': token_counts})

# Create a histogram of the token count distribution
df.hist(bins=40, )

# Show the plot
plt.show()

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

# Check similarity search is working
query = "Who created transformers?"
docs = db.similarity_search(query)
docs[0]

# Create QA chain to integrate similarity search with user queries (answer query from knowledge base)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

query = "Who created transformers?"
docs = db.similarity_search(query)

chain.run(input_documents=docs, question=query)

chat_history = []

@app.post('/')
async def ai_talk(event: linebot.models.MessageEvent):
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
        response = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo'
            , temperature = 0.5
            , messages = [
                {
                    'role': 'system'
                    , 'content': OPENAI_CHARACTER_PROFILE.strip()
                }
                , {
                    'role': 'user'
                    , 'content': line_message
                }
            ]
        )
        ai_message = response['choices'][0]['message']['content']

        # LINE メッセージの送信
        line_bot_api.push_message(line_user_id, TextSendMessage(ai_message))

    # LINE Webhook サーバーへ HTTP レスポンスを返す
    return 'ok'
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # テキストメッセージを受信した場合の処理
    user_message = event.message.text
    # QAチェーンの実行
    qa_result = chain.run(input_documents=docs, question=user_message)
    # QA結果から回答を取得
    answer = qa_result.answer
    # Line Botに応答を送信
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=answer)
    )

if __name__ == "__main__":
    app.run()