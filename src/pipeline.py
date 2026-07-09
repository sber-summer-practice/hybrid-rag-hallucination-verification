#pip install torch transformers datasets rank_bm25 sentence-transformers pymorphy3 bitsandbytes accelerate
import json
import os
import re
import numpy as np
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pymorphy3
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device.upper()}")
if device == "cpu":
    print("ВНИМАНИЕ: Смените тип среды выполнения на GPU (T4).")

# 1. Dataset SberQuAD
print("\nЗагрузка валидационной выборки SberQuAD...")
raw_dataset = load_dataset("kuznetsoffandrey/sberquad", split="validation")

# 2. Extracting contexts
print("Создание базы знаний...")
documents = []
for item in raw_dataset:
    context = item.get('context', '').strip()
    if context:
        documents.append(context)

documents = list(set(documents))
print(f"База знаний успешно создана! Загружено {len(documents)} уникальных чанков.")

# 3. Morphology and Tokenizer for BM25
print("Инициализация pymorphy3 и лемматизация документов для BM25...")
morph = pymorphy3.MorphAnalyzer()

def russian_lemmatized_tokenizer(text):
    words = re.findall(r'\b[а-яА-Яa-zA-Z0-9_]+\b', text.lower())
    return [morph.parse(word)[0].normal_form for word in words]

tokenized_documents = [russian_lemmatized_tokenizer(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_documents)

# 4. Embedding
print("Загрузка мультиязычной модели эмбеддингов LaBSE...")
embedder = SentenceTransformer("sentence-transformers/LaBSE", device=device)
doc_embeddings = embedder.encode(documents, convert_to_tensor=True, normalize_embeddings=True)

# 5. Loading LLM (Using Qwen 2.5)
print("\nЗагрузка квантованных моделей (4-bit NF4)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Qwen2.5-7B is well-suited for Russian-language RAG
model_id = "Qwen/Qwen2.5-7B-Instruct"
print(f"Загрузка единой базовой модели: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)

# Generation function
def ask_model(messages, max_tokens=150):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False  
    )
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

# Hybrid search function
def hybrid_search(query, top_k=3):
    tokenized_query = russian_lemmatized_tokenizer(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    query_embedding = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()

    # Avoid division by zero with empty scores
    bm25_max, bm25_min = np.max(bm25_scores), np.min(bm25_scores)
    bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-9) if (bm25_max - bm25_min) > 0 else bm25_scores

    cos_max, cos_min = np.max(cos_scores), np.min(cos_scores)
    cos_norm = (cos_scores - cos_min) / (cos_max - cos_min + 1e-9) if (cos_max - cos_min) > 0 else cos_scores

    hybrid_scores = 0.5 * bm25_norm + 0.5 * cos_norm
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    return [documents[idx] for idx in top_indices]

# 6. RAG Pipeline
# ==========================================
def run_rag_agent(question):
    print(f"\n[Агент]: Запускаю гибридный поиск по запросу: '{question}'...")
    retrieved_chunks = hybrid_search(question, top_k=3)
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Шаг 1: Генерация ответа 
    gen_messages = [
        {
            "role": "system",
            "content": (
                "Вы — точный русскоязычный ассистент. Ответьте на вопрос пользователя, строго основываясь на предоставленном Контексте.\n"
                "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
                "1. Отвечайте максимально кратко (1-2 предложения), формулируйте только прямой факт.\n"
                "2. Пишите ответ СТРОГО на русском языке. Используйте ТОЛЬКО кириллицу для всех имён и названий (например, 'Имхотеп', а не 'Имhotep').\n"
                "3. Запрещено додумывать логику, вводить причинно-следственные связи или детали, которых нет в тексте.\n"
                "4. Если в тексте нет прямого ответа, напишите строго одну фразу: 'Я не знаю'."
            )
        },
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"}
    ]
    initial_answer = ask_model(gen_messages, max_tokens=80) 
    print(f"[Первичный ответ]: {initial_answer}\n")

    # Шаг 2: Верификация
    eval_messages = [
        {
            "role": "system",
            "content": (
                "Вы — строгий эксперт по верификации ответов в RAG-системах.\n"
                "Ваша задача — проверить, подтверждается ли 'Ответ' предоставленным 'Контекстом'.\n"
                "ПРАВИЛА ВАЛИДАЦИИ:\n"
                "1. Если ответ обрывается на полуслове (например, 'содерж'), содержит латинские буквы там, где нужна кириллица, или явно додуман — это ГАЛЛЮЦИНАЦИЯ.\n"
                "2. Если ключевой факт и имена переданы верно, не придирайтесь к порядку слов или синонимам.\n"
                "3. Напишите 1-2 предложения краткого анализа, а на последней строке выведите строго одно слово: ДА (если ответ корректен) или НЕТ (если это галлюцинация/брак)."
            )
        },
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет модели: {initial_answer}\n\nВыведите обоснование и вердикт (ДА/НЕТ) в самом конце:"}
    ]
    verdict_raw = ask_model(eval_messages, max_tokens=100).upper()
    print(f"[Рассуждение и вердикт валидатора]:\n{verdict_raw}")

    print("\n=== ИТОГОВЫЙ РЕЗУЛЬТАТ ===")
    
    last_line = verdict_raw.split('\n')[-1].strip()

    if "ДА" in last_line or verdict_raw.endswith("ДА"):
        print("Перекрестная проверка пройдена успешно. Ответ верифицирован.")
        return initial_answer
    else:
        print("ОБНАРУЖЕНА ГАЛЛЮЦИНАЦИЯ! Сгенерированный ответ заблокирован.")
        return "Ошибка: Ответ заблокирован модулем безопасности."

# Тестовый запуск на проблемном месте
test_question = "Когда египетский врач Имхотеп впервые описал некоторые органы и их функции?"
result = run_rag_agent(test_question)

