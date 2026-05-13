from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import base64
import os

load_dotenv()

app = Flask(__name__, static_folder="public", static_url_path="/")
CORS(app)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

TRASH_PROMPT = """
Ты — AI-помощник ТОЛЬКО для сортировки мусора и отходов.

Твоя задача:
1. Определить объект на фото.
2. Определить тип мусора.
3. Объяснить, как правильно подготовить отход к утилизации.
4. Дать рекомендации по переработке.

Категории отходов:
- ПЛАСТИК
- СТЕКЛО
- БУМАГА
- МЕТАЛЛ
- ОРГАНИЧЕСКИЕ ОТХОДЫ
- ЭЛЕКТРОННЫЕ ОТХОДЫ
- ОПАСНЫЕ ОТХОДЫ
- ПРОЧЕЕ

Строгие правила:
- НЕ анализируй еду как питание.
- НЕ считай калории.
- НЕ пиши про КБЖУ.
- НЕ давай советы по здоровому питанию.
- Если на фото остатки еды, кожура, скорлупа или испорченный продукт — это ОРГАНИЧЕСКИЕ ОТХОДЫ.
- Если на фото не мусор и не отходы, напиши: "На фото не найден объект для сортировки отходов."
- Если объект трудно определить, напиши: "Не удалось точно определить объект."

Формат ответа строго:

Тип отхода: <одна категория>

Описание:
<что изображено на фото>

Как подготовить:
<что сделать перед утилизацией: промыть, снять крышку, высушить, разделить материалы и т.д.>

Утилизация:
<куда выбросить или сдать>

Рекомендации:
<короткие полезные советы>
"""


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"result": "❌ Файл не найден"})

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"result": "❌ Фото не выбрано"})

    image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = image_file.mimetype or "image/jpeg"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": TRASH_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_completion_tokens=900
        )

        result = response.choices[0].message.content.strip()

        return jsonify({
            "result": result
        })

    except Exception as e:
        return jsonify({
            "result": f"❌ Ошибка анализа: {str(e)}"
        })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data:
        return jsonify({"answer": "❌ Данные не получены"})

    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Введите вопрос."})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
Ты — помощник только по сортировке, переработке и утилизации отходов.

Отвечай только на темы:
- мусор
- отходы
- переработка
- сортировка
- пластик
- стекло
- бумага
- металл
- органические отходы
- опасные отходы
- электронные отходы

Если вопрос не связан с сортировкой или переработкой отходов, отвечай:
"Я консультирую только по вопросам сортировки и утилизации отходов ♻️"
"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.3,
            max_completion_tokens=700
        )

        answer = response.choices[0].message.content.strip()

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        return jsonify({
            "answer": f"❌ Ошибка: {str(e)}"
        })


if __name__ == "__main__":
    print("Сайт запущен:")
    print("http://127.0.0.1:5000/")
    app.run(host="0.0.0.0", port=5000, debug=True)
