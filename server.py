from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq
import base64
from dotenv import load_dotenv
import os
load_dotenv()
app = Flask(__name__, static_folder="public", static_url_path="/")
CORS(app)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

last_analysis = None

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    global last_analysis

    if "image" not in request.files:
        return jsonify({"result": "❌ Файл не найден"})

    image_file = request.files["image"]
    image_bytes = image_file.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt_text = """
Ты ИИ для сортировки мусора.

Определи объект на фото и отнеси его ТОЛЬКО к одной категории:

- ПЛАСТИК
- СТЕКЛО
- БУМАГА
- МЕТАЛЛ
- ОРГАНИЧЕСКИЕ ОТХОДЫ
- ПРОЧЕЕ

Формат ответа строго:

Тип отхода: <категория>

Описание:
<что это за предмет>

Подготовка:
<как очистить, промыть, снять крышки и т.д.>

Утилизация:
<куда выбросить, как переработать, рекомендации>

Пиши понятно и короткими абзацами.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_completion_tokens=1024
        )

        analysis_text = response.choices[0].message.content

        last_analysis = analysis_text

        return jsonify({
            "result": analysis_text
        })

    except Exception as e:
        return jsonify({
            "result": f"❌ Ошибка анализа: {str(e)}"
        })


@app.route("/chat", methods=["POST"])
def chat():
    global last_analysis

    if not last_analysis:
        return jsonify({
            "answer": "⚠ Сначала загрузите изображение мусора."
        })

    data = request.get_json()

    question = data.get("question", "")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Ты помощник по сортировке и переработке мусора."
                },
                {
                    "role": "user",
                    "content": f"""
Результат анализа:
{last_analysis}

Вопрос пользователя:
{question}

Отвечай подробно и понятно.
"""
                }
            ],
            temperature=0.4,
            max_completion_tokens=1024
        )

        answer = response.choices[0].message.content

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        return jsonify({
            "answer": f"❌ Ошибка: {str(e)}"
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)