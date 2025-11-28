#!/usr/bin/env python3
"""
camera_caption_no_token.py

Захват с веб-камеры, локальная генерация подписей (captioning) без токенов,
использует публичную модель BLIP из transformers.

Установка зависимостей:
  pip install --upgrade pip
  pip install torch torchvision transformers pillow opencv-python requests

Запуск:
  python camera_caption_no_token.py

Опции (через переменные в коде или расширьте для argparse):
  - FPS_SEND: сколько inference в секунду (по умолчанию 0.5 = 1 запрос каждые 2s)
  - KEYWORDS: список слов для "чеков" (если слово встретилось в caption — подсветка)
"""

import os
import time
import cv2
from PIL import Image
import numpy as np
import torch
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

# ------- Настройки --------
MODEL_ID = "Salesforce/blip-image-captioning-base"  # публичная модель, не требует HF_TOKEN
CAMERA_INDEX = 0
FPS_SEND = 0.5            # запросов в секунду (0.5 => каждые 2 секунды)
MAX_NEW_TOKENS = 40
FRAME_MAX_SIDE = 720      # уменьшение кадра перед инференсом, чтобы быстрее работало

# список ключевых слов для "чеков" (регистронезависимо)
# сюда добавили knife/gun и т.п.
KEYWORDS = [
    "person", "cat", "dog", "phone",
    "knife", "knives", "gun", "pistol", "revolver", "rifle",
    "weapon", "blade", "machete",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Куда шлём события в KIKO backend
KIKO_BACKEND_URL = os.getenv("KIKO_BACKEND_URL", "http://localhost:3000")
KIKO_EVENTS_ENDPOINT = f"{KIKO_BACKEND_URL}/alert-ai/video-caption"
# --------------------------

# Маппинг "сырых" слов из caption -> нормализованные ключевые слова
# то, что тебе нужно: knife -> "Knife", pistol -> "Gun" и т.д.
DANGEROUS_KEYWORDS_MAP = {
    "knife": "Knife",
    "knives": "Knife",
    "blade": "Knife",
    "machete": "Knife",

    "gun": "Gun",
    "pistol": "Gun",
    "revolver": "Gun",
    "rifle": "Gun",

    "weapon": "Weapon",
}

# Накопленные опасные ключевые слова за сессию (для дебага / аналитики)
SESSION_DANGEROUS_KEYWORDS = set()


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # поддержка Apple MPS
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = choose_device()
print("Используем device:", device)

print("Загружаем модель и процессор (может занять время при первом запуске)...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
model.to(device)
model.eval()


def pil_from_bgr(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def prepare_image_for_model(pil_img: Image.Image, max_side=FRAME_MAX_SIDE):
    w, h = pil_img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return pil_img


def generate_caption(pil_img: Image.Image):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    caption = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return caption


def draw_multiline_text(frame, lines, pos=(10, 30), font_scale=0.8,
                        color=(0, 255, 0), thickness=2, line_height=26):
    x, y = pos
    for i, line in enumerate(lines):
        y_line = y + i * line_height
        cv2.putText(frame, line, (x, y_line), FONT, font_scale, color, thickness, lineType=cv2.LINE_AA)


def contains_keyword(text, keywords):
    if not text:
        return None
    t = text.lower()
    for k in keywords:
        if k.lower() in t:
            return k
    return None


def extract_dangerous_keywords(text: str):
    """
    Вытаскивает нормализованные опасные ключевые слова из caption.
    Пример:
      "a man holding a knife and a gun" -> ["Knife", "Gun"]
    """
    if not text:
        return []

    t = text.lower()
    found = []
    for raw_word, normalized in DANGEROUS_KEYWORDS_MAP.items():
        if raw_word in t and normalized not in found:
            found.append(normalized)
    return found


def send_caption_event_to_kiko(
    camera_id: str,
    caption: str,
    keywords: list,
    dangerous_keywords: list,
    frame_id: int | None = None,
):
    """
    Отправка события в KIKO backend (localhost:3001/alert-ai/video-caption).
    Сейчас шлём только если есть dangerous_keywords.
    """
    if not dangerous_keywords:
        return

    payload = {
        "cameraId": camera_id,
        "caption": caption,
        "keywords": keywords,
        "dangerousKeywords": dangerous_keywords,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta": {
            "source": "blip-caption-service",
            "frameId": frame_id,
        },
    }

    try:
        resp = requests.post(KIKO_EVENTS_ENDPOINT, json=payload, timeout=1.0)
        if resp.status_code != 200:
            print("KIKO backend responded with", resp.status_code, resp.text)
    except Exception as e:
        print("Error sending event to KIKO:", e)


def main():
    global SESSION_DANGEROUS_KEYWORDS

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Не удалось открыть камеру", CAMERA_INDEX)
        return

    last_call = 0.0
    last_caption = ""
    last_keyword = None
    last_dangerous_keywords = []  # последние Knife/Gun/Weapon
    frame_id = 0

    print("Нажмите 'q' чтобы выйти.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Кадр не получен. Выход.")
                break

            frame_id += 1
            now = time.time()
            if now - last_call >= 1.0 / max(FPS_SEND, 1e-6):
                # подготовка
                pil = pil_from_bgr(frame)
                pil_small = prepare_image_for_model(pil)

                try:
                    caption = generate_caption(pil_small)
                except Exception as e:
                    caption = f"[Ошибка инференса: {e}]"

                last_caption = caption
                last_keyword = contains_keyword(caption, KEYWORDS)

                # Новый блок: вытащить опасные ключевые слова (Knife, Gun, Weapon)
                dangerous_keywords = extract_dangerous_keywords(caption)
                last_dangerous_keywords = dangerous_keywords

                if dangerous_keywords:
                    # сохранить в сессионный набор (для логов/аналитики)
                    for kw in dangerous_keywords:
                        SESSION_DANGEROUS_KEYWORDS.add(kw)

                    print("Dangerous keywords this frame:", dangerous_keywords)
                    print("Dangerous keywords total session:", list(SESSION_DANGEROUS_KEYWORDS))

                    # Список обычных keywords (по желанию)
                    keywords_for_send = []
                    if last_keyword:
                        keywords_for_send.append(last_keyword)

                    # Шлём ивент в KIKO
                    send_caption_event_to_kiko(
                        camera_id=str(CAMERA_INDEX),
                        caption=caption,
                        keywords=keywords_for_send,
                        dangerous_keywords=dangerous_keywords,
                        frame_id=frame_id,
                    )

                last_call = now

            # рисуем информацию на кадре
            out = frame.copy()

            lines = [f"Caption: {last_caption}"]
            if last_keyword:
                lines.append(f"Keyword detected: {last_keyword}")

            # если нашли опасные ключевые слова — покажем их
            if last_dangerous_keywords:
                lines.append("Danger: " + ", ".join(last_dangerous_keywords))

            # если последний кадр содержал ключевые слова из KEYWORDS — подсветим рамкой
            if last_keyword:
                h, w = out.shape[:2]
                cv2.rectangle(out, (5, 5), (w - 5, h - 5), (0, 0, 255), 4)

            draw_multiline_text(out, lines, pos=(10, 30), font_scale=0.7,
                                color=(0, 255, 0), thickness=2)

            cv2.imshow("Camera caption (no token)", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
