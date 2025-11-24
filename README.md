# Camera caption service (локальный, без Flask)

Кратко
------
Этот проект — простой локальный сервис/скрипт для захвата видео с веб‑камеры и генерации подписи (caption) для кадров без использования внешних API и без Flask. По умолчанию использует публичную модель BLIP (Salesforce/blip-image-captioning-base) через `transformers`. Включено логирование, регулировка частоты инференса и поддержка GPU (CUDA/MPS) при наличии.
<img width="642" height="507" alt="image" src="https://github.com/user-attachments/assets/7aa72814-9295-4ac7-aff2-eaa4c2d7b2a4" />



Файлы
-----
- `camera_caption_no_token.py` — основной скрипт захвата камеры и инференса.
- `requirements.txt` — список рекомендуемых зависимостей (см. ниже).

Возможности
----------
- Локальный инференс изображений с камеры без HF токена.
- Поддержка CUDA / MPS / CPU (авто‑выбор устройства).
- Настраиваемая частота инференса (FPS_SEND).
- Поиск ключевых слов в подписи и визуальная подсветка кадра при совпадении.
- Логирование этапов и времени инференса (уровень LOG_LEVEL).

Требования
---------
- Python 3.8+
- Виртуальное окружение (рекомендуется)
- Пакеты из requirements.txt (пример приведён ниже)

Пример requirements.txt:
```
torch
torchvision
transformers
pillow
opencv-python
huggingface_hub
safetensors
accelerate
```

Установка (Windows PowerShell)
------------------------------
1. Создайте и активируйте виртуальное окружение:
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Обновите pip и установите зависимости:
   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. (Опционально) Если у вас NVIDIA GPU — установите `torch` с нужной поддержкой CUDA по инструкции с https://pytorch.org/get-started/locally/
   Для Apple Silicon — следуйте инструкциям PyTorch для MPS.

Установка (Linux / macOS)
-------------------------
1. Создайте и активируйте venv:
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Обновите pip и установите зависимости:
   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

Запуск
-----
Запуск скрипта:
```
python camera_caption_no_token.py
```
По умолчанию:
- Скрипт откроет камеру с индексом 0.
- Инференс выполняется с частотой `FPS_SEND` (по умолчанию 0.5 — т.е. 1 запрос каждые 2 секунды).
- Для выхода — нажмите `q` в окне с видео.

Переменные конфигурации в коде
------------------------------
В начале `camera_caption_no_token.py` доступны параметры:
- MODEL_ID — идентификатор модели (по умолчанию `"Salesforce/blip-image-captioning-base"`).
- CAMERA_INDEX — индекс веб‑камеры (0 по умолчанию).
- FPS_SEND — частота инференса (запросов в секунду).
- MAX_NEW_TOKENS — макс. токенов при генерации подписи.
- FRAME_MAX_SIDE — ограничение стороны изображения перед отправкой в модель.
- KEYWORDS — список ключевых слов для "чеков".
- LOG_LEVEL — уровень логирования (по умолчанию INFO, можно задать через переменную окружения `LOG_LEVEL`).

Логи
----
По умолчанию логируются:
- выбор устройства (CUDA/MPS/CPU),
- этапы загрузки модели,
- попытки открыть камеру,
- начало и результат каждого инференса (включая время выполнения),
- ошибки инференса и исключения.

Для более подробных логов:
Windows PowerShell:
```
$env:LOG_LEVEL = "DEBUG"
python camera_caption_no_token.py
```

Оптимизация и советы
--------------------
- Для слабых машин уменьшите `FRAME_MAX_SIDE` (например до 320) и уменьшите `FPS_SEND`.
- Если используете GPU — убедитесь, что `torch` установлен с поддержкой вашей CUDA версии.
- При первом запуске модель будет скачана с Hugging Face Hub — это может занять время и трафик.

Как заменить BLIP на EZCon/FastVLM-1.5B-mlx (интеграция локальной модели)
-----------------------------------------------------------------------
Если вы хотите именно модель EZCon/FastVLM-1.5B-mlx (публичная) — общий план действий:

1. Скачайте репозиторий/веса:
   - c Git LFS:
     ```
     git lfs install
     git clone https://huggingface.co/EZCon/FastVLM-1.5B-mlx
     ```
   - или через Python:
     ```python
     from huggingface_hub import snapshot_download
     snapshot_download("EZCon/FastVLM-1.5B-mlx")
     ```

2. Откройте README этого репо и посмотрите, какой API они предоставляют (PyTorch/transformers, JAX, WebGPU, CLI и т.п.). Обычно там есть примеры запуска.

3. В `camera_caption_no_token.py` замените реализацию функции `generate_caption()` на реализацию, совместимую с FastVLM:
   - Вариант A — если репо предлагает Python API:
     - Импортируйте и загрузите модель (например `model = FastVLM.from_pretrained(local_path)`).
     - Реализуйте предобработку изображения в тот формат, который ожидает модель.
     - Вызовите `model.generate(...)` или эквивалент и верните строку подписи.
   - Вариант B — если есть CLI-скрипт:
     - Сохраните кадр во временный файл и вызовите CLI через `subprocess`, прочитайте результат из stdout.
   - Вариант C — если модель оптимизирована под WebGPU/Metal — следуйте инструкциям репо (возможно запуск в браузере или через специальный рантайм).

4. Убедитесь, что все зависимости, указанные в README FastVLM, установлены (весы, специфичные библиотеки, возможно `jax`, `torch`, `metalps`, и т.д.).

Пример шаблона замены (высокоуровневая идея)
```python
# вместо generate_caption из BLIP:
def generate_caption(pil_img):
    # 1) преобразовать
