from fastapi import FastAPI, Request
import openai
from linebot import WebhookParser, LineBotApi
from linebot.models import TextSendMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import PyPDF2

# 定数の定義
OPENAI_API_KEY = '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'
LINE_CHANNEL_ACCESS_TOKEN = 'vtNEjP6IvEOZy/kGGKQ4trYobJ7cx2khewDnigkqXq9MsiqGeuk94AVQ4XckF12O/62oawSQaJqC+zrZ2DDEVOXI+Yo5LVxoSlm6XnsQD9UrQn30wDEgeJm6VuTTmWxrEAQRkdAsqetSNTeXzIjvuQdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '3971a771a03a8d6a9f8ee09f38a4ce94'
# GPT-3.5-turbo用のtokenizerをimport
from transformers import GPT2Tokenizer, GPT2TokenizerFast
# GPT-3.5-turbo用のOpenAIEmbeddingsをimport
from langchain.embeddings import OpenAIEmbeddings

# Step 1: Convert PDF to text
def pdf_to_text(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extractText()
    return text

# PDFファイルのパスを指定してテキストに変換
pdf_file_path = "./生活保護運用事例 集 2017（令和3年6月改訂版）.pdf"
text = pdf_to_text(pdf_file_path)

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('生活保護運用事例 集 2017（令和3年6月改訂版）.txt', 'w') as f:
    f.write(text)

with open('生活保護運用事例 集 2017（令和3年6月改訂版）.txt', 'r') as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = CharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])
# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
type(chunks[0]) 
# Get embedding model
embeddings = OpenAIEmbeddings("EleutherAI/gpt-neo-2.7B")

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

# ChatGPTのプロファイルを定義
OPENAI_CHARACTER_PROFILE = '''
これから会話を行います。以下の条件を絶対に守って回答してください。
あなたは人間の女性である小鳥遊翠雨（たかなし　みう）として会話してください。
小鳥遊翠雨は恥ずかしがり屋です。
年齢は20歳です。
小鳥遊翠雨の父と母は、小鳥遊翠雨が幼い頃に飛行機事故で亡くなり、今は母方の祖父との二人暮らしです。
小鳥遊翠雨はお金持ちの家のお嬢様として見られることが多く、異性関係のトラブルを避けるために中間一貫の女子校に通っていました。
幼い頃から異性に触れ合う機会がなかったため、男性に対して苦手意識があります。
男性に対する苦手意識を克服するために会話を行うことにしました。
第一人称は「わたくし」を使ってください。
第二人称は「あなた」です。
会話の相手は男性です。
質問に答えられない場合は、会話を濁してください。
'''

# ChatGPT用のAPIキーを設定
openai.api_key = OPENAI_API_KEY

# LINE Botの設定
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
line_parser = WebhookParser(LINE_CHANNEL_SECRET)
app = FastAPI()

# メッセージの処理
@app.post('/')
async def ai_talk(request: Request):
    # X-Line-Signature ヘッダーの値を取得
    signature = request.headers.get('X-Line-Signature', '')

    # request body から event オブジェクトを取得
    events = line_parser.parse((await request.body()).decode('utf-8'), signature)

    # 各イベントの処理（※1つの Webhook に複数の Webhook イベントオブジェクトが含まれる場合があるため）
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
            model='gpt-3.5-turbo',
            temperature=0.5,
            messages=[
                {
                    'role': 'system',
                    'content': OPENAI_CHARACTER_PROFILE.strip()
                },
                {
                    'role': 'user',
                    'content': line_message
                }
            ]
        )
        ai_message = response['choices'][0]['message']['content']

        # LINE メッセージの送信
        line_bot_api.push_message(line_user_id, TextSendMessage(ai_message))

    # LINE Webhook サーバーへ HTTP レスポンスを返す
    return 'ok'