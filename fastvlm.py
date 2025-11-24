#!/usr/bin/env python3
"""
camera_caption_no_token.py

Захват с веб-камеры, локальная генерация подписей (captioning) без токенов,
использует публичную модель BLIP из transformers.

Установка зависимостей:
  pip install --upgrade pip
  pip install torch torchvision transformers pillow opencv-python

Если у вас macOS + Apple Silicon (M1/M2) — установите torch для MPS согласно инструкции на https://pytorch.org

Запуск:
  python camera_caption_no_token.py
Опции (через переменные в коде или расширьте для argparse):
  - FPS_SEND: сколько inference в секунду (по умолчанию 0.5 = 1 запрос каждые 2s)
  - KEYWORDS: список слов для "чеков" (если слово встретилось в caption — подсветка)
"""

import time
import cv2
from PIL import Image
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ------- Настройки --------
MODEL_ID = "Salesforce/blip-image-captioning-base"  # публичная модель, не требует HF_TOKEN
CAMERA_INDEX = 0
FPS_SEND = 0.5            # запросов в секунду (0.5 => каждые 2 секунды)
MAX_NEW_TOKENS = 40
FRAME_MAX_SIDE = 720      # уменьшение кадра перед инференсом, чтобы быстрее работало
KEYWORDS = ["person", "cat", "dog", "phone"]  # список ключевых слов для "чеков" (регистронезависимо)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# --------------------------

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

def draw_multiline_text(frame, lines, pos=(10,30), font_scale=0.8, color=(0,255,0), thickness=2, line_height=26):
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

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Не удалось открыть камеру", CAMERA_INDEX)
        return

    last_call = 0.0
    last_caption = ""
    last_keyword = None
    print("Нажмите 'q' чтобы выйти.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Кадр не получен. Выход.")
                break

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
                last_call = now

            # рисуем информацию на кадре
            out = frame.copy()
            # подпись
            lines = [f"Caption: {last_caption}"]
            if last_keyword:
                lines.append(f"Keyword detected: {last_keyword}")
                # подсветим рамкой красного цвета, если ключевое слово найдено
                h, w = out.shape[:2]
                cv2.rectangle(out, (5, 5), (w-5, h-5), (0,0,255), 4)
            draw_multiline_text(out, lines, pos=(10,30), font_scale=0.7, color=(0,255,0), thickness=2)

            cv2.imshow("Camera caption (no token)", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()