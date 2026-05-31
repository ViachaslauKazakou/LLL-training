"""
OCR приложение для подготовки данных из бумажных задачников.

Функции:
- Загрузка сканов (JPEG/PNG)
- OCR распознавание (Tesseract - бесплатно)
- Опционально: улучшение с GPT-4 Vision
- Извлечение задач из текста
- Редактирование и валидация
- Сохранение в JSON формат
- Копирование в data/ для обучения

Запуск:
    streamlit run ocr_app.py
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Optional
import shutil
from PIL import Image
import io
import numpy as np

# Для OCR
try:
    import pytesseract
    import subprocess
    
    # Автоопределение пути к tesseract
    try:
        result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
        if result.returncode == 0:
            tesseract_path = result.stdout.strip()
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            TESSERACT_AVAILABLE = True
        else:
            TESSERACT_AVAILABLE = False
    except:
        TESSERACT_AVAILABLE = False
except ImportError:
    TESSERACT_AVAILABLE = False

# Для GPT-4 Vision (опционально)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# Конфигурация
# ═══════════════════════════════════════════════════════════════

# Папки
BASE_DIR = Path(__file__).parent
DATA_PREP_DIR = BASE_DIR / "data_preparation"
SCANS_DIR = DATA_PREP_DIR / "scans"
OCR_RESULTS_DIR = DATA_PREP_DIR / "ocr_results"
JSON_OUTPUT_DIR = DATA_PREP_DIR / "json_output"
TRAINING_DATA_DIR = BASE_DIR / "data"

# Создаём директории
for dir_path in [SCANS_DIR, OCR_RESULTS_DIR, JSON_OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Категории задач
TASK_CATEGORIES = [
    "уравнения_реакций",
    "молярная_масса",
    "расчеты",
    "валентность",
    "стехиометрия",
    "органика",
    "неорганика",
    "другое"
]

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# ═══════════════════════════════════════════════════════════════
# Предобработка изображений для улучшения OCR
# ═══════════════════════════════════════════════════════════════

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Улучшает изображение для лучшего OCR распознавания.
    
    - Конвертирует в grayscale
    - Увеличивает контраст
    - Повышает резкость
    - Бинаризация (черно-белое)
    """
    from PIL import ImageEnhance, ImageFilter, ImageOps
    
    # Конвертируем в RGB если нужно
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Конвертируем в grayscale
    image = ImageOps.grayscale(image)
    
    # Увеличиваем контраст
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Увеличиваем контраст в 2 раза
    
    # Повышаем резкость
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    # Бинаризация (адаптивная)
    # Конвертируем в черно-белое для лучшего распознавания
    image = image.point(lambda x: 0 if x < 128 else 255, '1')
    
    return image

def postprocess_ocr_text(text: str) -> str:
    """
    Постобработка OCR текста для исправления типичных ошибок.
    
    Исправляет:
    - Дроби (n=— = — → n = m/M)
    - Степени (10% → 10²³)
    - Химические формулы
    - Математические символы
    """
    import re
    
    # Исправляем дроби (— часто означает дробную черту)
    # n=— = — =0,25 → n = m/M = 0,25
    text = re.sub(r'n\s*=\s*—+\s*=\s*—+\s*=', 'n = m/M =', text)
    text = re.sub(r'=\s*—+\s*=', '= m/M =', text)
    
    # Исправляем степени
    # 10% → 10²³
    text = text.replace('10%', '10²³')
    text = text.replace('10°?', '10⁻²³')
    text = text.replace('107', '10²³')
    
    # Исправляем точки в числах на умножение
    # 1,506. 1023 → 1,506 · 10²³
    text = re.sub(r'(\d+[,.]?\d*)\.\s*(\d+)', r'\1 · \2', text)
    
    # Исправляем дроби в формулах
    # т(атома) = —— = ——————— → m(атома) = M/Nₐ
    text = re.sub(r'т\(атома\)\s*=\s*—+\s*=\s*—+', 'm(атома) = M/Nₐ', text)
    
    # Исправляем N, (Nₐ - число Авогадро)
    text = text.replace('N,', 'Nₐ')
    text = text.replace('№', 'Nₐ')
    
    # Исправляем es → ≈
    text = text.replace('es', '≈')
    
    # Исправляем двоеточие на умножение в формулах
    # 6,023: 10% → 6,023 · 10²³
    text = re.sub(r'(\d+[,.]?\d*):(\s*\d+)', r'\1 ·\2', text)
    
    # Исправляем дефис на минус в степенях
    text = re.sub(r'(\d+)\s*-\s*(\d+)\s+(атомов|г)', r'\1 · 10²³ \3', text)
    
    return text

# ═══════════════════════════════════════════════════════════════
# OCR функции
# ═══════════════════════════════════════════════════════════════

def ocr_with_tesseract(
    image: Image.Image,
    lang: str = "rus+eng",
    postprocess: bool = False
) -> str:
    """
    OCR распознавание с Tesseract.
    
    Args:
        image: PIL Image
        lang: Языки для распознавания (rus+eng)
        postprocess: Применять постобработку для исправления формул
    
    Returns:
        Распознанный текст
    """
    if not TESSERACT_AVAILABLE:
        return "❌ Tesseract не установлен. Установите: brew install tesseract\nЗатем перезапустите приложение."
    
    try:
        # Конвертируем изображение в RGB формат (pytesseract требует)
        # Это решает проблему "Unsupported image format/type"
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Конвертируем PIL Image в numpy array для лучшей совместимости с pytesseract
        # Некоторые версии pytesseract не работают с PIL Image напрямую
        image_array = np.array(image)
        
        # Конфигурация для лучшего распознавания химических формул
        # --oem 3: LSTM neural net mode (лучше для формул)
        # --psm 6: Assume uniform block of text
        custom_config = r'--oem 3 --psm 6'
        
        # Проверяем путь к tesseract
        tesseract_path = pytesseract.pytesseract.tesseract_cmd
        
        # Используем numpy array вместо PIL Image
        text = pytesseract.image_to_string(image_array, lang=lang, config=custom_config)
        
        if not text.strip():
            return "⚠️ Tesseract не распознал текст на изображении.\n\nПопробуйте:\n- Использовать GPT-4 Vision\n- Улучшить качество скана"
        
        # Постобработка для исправления типичных ошибок OCR
        if postprocess:
            text = postprocess_ocr_text(text)
        
        return text
    except pytesseract.TesseractNotFoundError:
        return f"❌ Tesseract не найден.\n\nПуть: {pytesseract.pytesseract.tesseract_cmd}\n\nУстановите:\nbrew install tesseract\nbrew install tesseract-lang"
    except Exception as e:
        error_msg = str(e)
        import traceback
        detailed_error = traceback.format_exc()
        return f"❌ Ошибка OCR: {error_msg}\n\nДетали:\n- Формат изображения: {image.mode}\n- Размер: {image.size}\n- Путь к tesseract: {pytesseract.pytesseract.tesseract_cmd}\n\nПолная ошибка:\n{detailed_error}\n\nПопробуйте:\n1. Использовать GPT-4 Vision\n2. Пересохранить изображение\n3. Проверить установку tesseract: brew reinstall tesseract"

def ocr_with_gpt4_vision(
    image: Image.Image,
    api_key: str,
    prompt: Optional[str] = None
) -> str:
    """
    OCR с GPT-4 Vision (платно, но точнее для химии).
    
    Args:
        image: PIL Image
        api_key: OpenAI API ключ
        prompt: Кастомный промпт
    
    Returns:
        Распознанный текст
    """
    if not OPENAI_AVAILABLE:
        return "❌ OpenAI библиотека не установлена. Установите: poetry add openai"
    
    if not api_key:
        return "❌ Введите OpenAI API ключ"
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Конвертируем изображение в base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        import base64
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Промпт для распознавания
        if not prompt:
            prompt = """Распознай текст с этого изображения задачника по химии.

ВАЖНО:
- Сохраняй химические формулы точно (H₂O, CO₂, H₂SO₄, etc)
- Сохраняй математические символы (→, ⇄, ±, etc)
- Разделяй задачи пустой строкой
- Нумеруй задачи как в оригинале

Верни только распознанный текст, без комментариев."""
        
        response = client.chat.completions.create(
            model="gpt-4o",  # или gpt-4-vision-preview
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"❌ Ошибка GPT-4 Vision: {str(e)}"

# ═══════════════════════════════════════════════════════════════
# Парсинг задач из текста
# ═══════════════════════════════════════════════════════════════

def parse_tasks_from_text(text: str) -> List[Dict]:
    """
    Извлекает задачи из распознанного текста.
    
    Ищет паттерны:
    - Нумерация: "1.", "Задача 1:", "№1"
    - Ключевые слова: "Задача:", "Решите:", "Найдите:"
    
    Returns:
        Список словарей с задачами
    """
    import re
    
    tasks = []
    
    # Разбиваем на блоки по нумерации или ключевым словам
    # Паттерн: номер + точка/скобка/двоеточие
    patterns = [
        r'(?:^|\n)\s*(\d+)[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|\Z)',  # "1. Задача..."
        r'(?:^|\n)\s*(?:Задача|№)\s*(\d+)[\.:]\s*(.+?)(?=\n\s*(?:Задача|№)\s*\d+|\Z)',  # "Задача 1: ..."
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
        for match in matches:
            number = match.group(1)
            content = match.group(2).strip()
            
            # Пропускаем слишком короткие
            if len(content) < 10:
                continue
            
            # Пытаемся найти ответ (если есть "Ответ:")
            answer = ""
            question = content
            
            answer_match = re.search(r'Ответ[:\s]+(.+?)(?:\n|$)', content, re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
                question = content[:answer_match.start()].strip()
            
            # Пытаемся найти решение
            solution = ""
            solution_match = re.search(
                r'Решение[:\s]+(.+?)(?=Ответ|$)',
                content,
                re.IGNORECASE | re.DOTALL
            )
            if solution_match:
                solution = solution_match.group(1).strip()
                question = content[:solution_match.start()].strip()
            
            tasks.append({
                "number": number,
                "question": question,
                "solution": solution,
                "answer": answer,
                "category": "другое",
                "difficulty": "medium"
            })
    
    # Если не нашли по паттернам, просто разбиваем по пустым строкам
    if not tasks:
        blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
        for i, block in enumerate(blocks, 1):
            if len(block) < 10:
                continue
            
            tasks.append({
                "number": str(i),
                "question": block,
                "solution": "",
                "answer": "",
                "category": "другое",
                "difficulty": "medium"
            })
    
    return tasks

# ═══════════════════════════════════════════════════════════════
# Сохранение и загрузка
# ═══════════════════════════════════════════════════════════════

def save_tasks_to_json(tasks: List[Dict], filename: str) -> Path:
    """Сохраняет задачи в JSON файл."""
    output_path = JSON_OUTPUT_DIR / filename
    
    # Убираем поле number (оно временное)
    cleaned_tasks = []
    for task in tasks:
        cleaned = {k: v for k, v in task.items() if k != 'number'}
        cleaned_tasks.append(cleaned)
    
    data = {
        "dataset_name": filename.replace('.json', ''),
        "created_at": datetime.now().isoformat(),
        "total_tasks": len(cleaned_tasks),
        "tasks": cleaned_tasks
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return output_path

def copy_to_training_data(json_path: Path) -> Path:
    """Копирует JSON в папку для обучения."""
    dest_path = TRAINING_DATA_DIR / json_path.name
    shutil.copy2(json_path, dest_path)
    return dest_path

# ═══════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="OCR для задачников по химии",
    page_icon="📚",
    layout="wide"
)

st.title("📚 OCR для подготовки данных из бумажных задачников")

# Инициализация session state
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = ""
if 'parsed_tasks' not in st.session_state:
    st.session_state.parsed_tasks = []

# ═══════════════════════════════════════════════════════════════
# Sidebar: Конфигурация
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Настройки")
    
    # OCR метод
    ocr_method = st.radio(
        "Метод OCR",
        ["Tesseract (бесплатно)", "GPT-4 Vision (платно, точнее)"],
        help="Tesseract быстрее и бесплатен, но GPT-4 Vision лучше распознаёт формулы"
    )
    
    # Настройки Tesseract
    if ocr_method == "Tesseract (бесплатно)":
        st.info("🆓 Tesseract — бесплатный OCR")
        if not TESSERACT_AVAILABLE:
            st.error("❌ Tesseract не установлен!")
            st.code("brew install tesseract\nbrew install tesseract-lang")
        else:
            st.success("✅ Tesseract установлен")
            # Показываем путь к tesseract
            try:
                tesseract_path = pytesseract.pytesseract.tesseract_cmd
                st.caption(f"📍 Путь: {tesseract_path}")
                
                # Проверяем версию
                import subprocess
                try:
                    version_result = subprocess.run(
                        [tesseract_path, '--version'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    version_line = version_result.stdout.split('\n')[0]
                    st.caption(f"🔖 {version_line}")
                except:
                    pass
            except:
                pass
        
        tesseract_lang = st.text_input(
            "Языки для распознавания",
            "rus+eng",
            help="rus+eng для русского и английского"
        )
        
        # Постобработка текста
        postprocess_text = st.checkbox(
            "🔧 Исправлять формулы и дроби",
            value=False,
            help="Автоматически исправляет типичные ошибки OCR: дроби (n=— → n=m/M), степени (10% → 10²³), химические символы (N, → Nₐ)"
        )
    
    # Настройки GPT-4 Vision
    else:
        st.info("💰 GPT-4 Vision — ~$0.01-0.02 за изображение")
        openai_api_key = st.text_input(
            "OpenAI API ключ",
            type="password",
            help="Получите на https://platform.openai.com"
        )
        
        if not OPENAI_AVAILABLE:
            st.error("❌ OpenAI не установлен!")
            st.code("poetry add openai")
    
    st.divider()
    
    # Статистика
    st.subheader("📊 Статистика")
    scans_count = len(list(SCANS_DIR.glob("*.jp*g"))) + len(list(SCANS_DIR.glob("*.png")))
    json_count = len(list(JSON_OUTPUT_DIR.glob("*.json")))
    
    st.metric("Сканов загружено", scans_count)
    st.metric("JSON файлов создано", json_count)
    
    # Быстрые ссылки
    st.divider()
    st.subheader("📁 Папки")
    st.markdown(f"• [Сканы]({SCANS_DIR})")
    st.markdown(f"• [OCR результаты]({OCR_RESULTS_DIR})")
    st.markdown(f"• [JSON выход]({JSON_OUTPUT_DIR})")
    st.markdown(f"• [Данные для обучения]({TRAINING_DATA_DIR})")

# ═══════════════════════════════════════════════════════════════
# Вкладки
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 1. Загрузка сканов",
    "🔍 2. OCR распознавание",
    "✏️ 3. Редактирование задач",
    "💾 4. Сохранение JSON",
    "🔗 5. Объединение датасетов"
])

# ═══════════════════════════════════════════════════════════════
# Вкладка 1: Загрузка сканов
# ═══════════════════════════════════════════════════════════════

with tab1:
    st.header("📤 Загрузка сканов")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Загрузите изображение")
        
        uploaded_file = st.file_uploader(
            "Выберите JPEG/PNG файл",
            type=["jpg", "jpeg", "png"],
            help="Загрузите отсканированную страницу из задачника"
        )
        
        if uploaded_file:
            # Загружаем изображение
            image = Image.open(uploaded_file)
            st.session_state.current_image = image
            
            # Сохраняем в папку сканов
            scan_filename = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
            scan_path = SCANS_DIR / scan_filename
            image.save(scan_path)
            
            st.success(f"✅ Сохранено: {scan_filename}")
            
            # Предпросмотр
            st.image(image, caption="Загруженное изображение", width='stretch')
            
            # Информация об изображении
            st.info(f"📏 Размер: {image.size[0]}x{image.size[1]} px | Формат: {image.format}")
    
    with col2:
        st.subheader("Загруженные сканы")
        
        # Список сканов
        scans = sorted(SCANS_DIR.glob("*.jp*g")) + sorted(SCANS_DIR.glob("*.png"))
        
        if scans:
            for scan_path in scans[-10:]:  # Последние 10
                col_btn, col_name = st.columns([1, 3])
                
                with col_btn:
                    if st.button("👁️", key=f"view_{scan_path.name}"):
                        st.session_state.current_image = Image.open(scan_path)
                        st.rerun()
                
                with col_name:
                    st.text(scan_path.name)
        else:
            st.info("Сканы ещё не загружены")

# ═══════════════════════════════════════════════════════════════
# Вкладка 2: OCR распознавание
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.header("🔍 OCR распознавание")
    
    if st.session_state.current_image is None:
        st.warning("⚠️ Сначала загрузите изображение во вкладке 1")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Изображение")
            st.image(
                st.session_state.current_image,
                caption="Текущее изображение",
                width='stretch'
            )
            
            # Кнопка распознавания
            if st.button("🚀 Распознать текст", type="primary", width='stretch'):
                with st.spinner("Распознаю текст..."):
                    if ocr_method == "Tesseract (бесплатно)":
                        text = ocr_with_tesseract(
                            st.session_state.current_image,
                            tesseract_lang,
                            postprocess=postprocess_text
                        )
                    else:
                        text = ocr_with_gpt4_vision(st.session_state.current_image, openai_api_key)
                    
                    st.session_state.ocr_text = text
                    
                    # Сохраняем результат OCR
                    ocr_filename = f"ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    ocr_path = OCR_RESULTS_DIR / ocr_filename
                    ocr_path.write_text(text, encoding='utf-8')
                    
                    st.success(f"✅ OCR завершён! Сохранено: {ocr_filename}")
                    st.rerun()
        
        with col2:
            st.subheader("Распознанный текст")
            
            if st.session_state.ocr_text:
                # HTML/JS редактор с WYSIWYG функционалом
                html_editor = f"""
                <style>
                    .editor-container {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    }}
                    .toolbar {{
                        background: #f0f2f6;
                        padding: 10px;
                        border-radius: 8px 8px 0 0;
                        border: 1px solid #e0e0e0;
                        display: flex;
                        flex-wrap: wrap;
                        gap: 5px;
                    }}
                    .toolbar button {{
                        padding: 8px 12px;
                        border: 1px solid #ccc;
                        background: white;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 14px;
                        transition: all 0.2s;
                    }}
                    .toolbar button:hover {{
                        background: #e8e8e8;
                        transform: translateY(-1px);
                    }}
                    .toolbar button.format-btn {{
                        font-weight: bold;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border: none;
                    }}
                    .toolbar button.format-btn:hover {{
                        background: linear-gradient(135deg, #5568d3 0%, #63408a 100%);
                    }}
                    .toolbar .separator {{
                        width: 1px;
                        background: #ccc;
                        margin: 0 5px;
                    }}
                    #textEditor {{
                        width: 100%;
                        min-height: 300px;
                        padding: 12px;
                        font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
                        font-size: 14px;
                        border: 1px solid #e0e0e0;
                        border-radius: 0 0 8px 8px;
                        border-top: none;
                        resize: vertical;
                        line-height: 1.6;
                    }}
                    #textEditor:focus {{
                        outline: none;
                        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.3);
                    }}
                    .info {{
                        margin-top: 8px;
                        font-size: 12px;
                        color: #666;
                    }}
                </style>
                
                <div class="editor-container">
                    <div class="toolbar">
                        <button class="format-btn" onclick="applyFormat('superscript')" title="Выделите текст и нажмите для преобразования в верхний индекс">
                            ⁿ Верхний индекс
                        </button>
                        <button class="format-btn" onclick="applyFormat('subscript')" title="Выделите текст и нажмите для преобразования в нижний индекс">
                            ₙ Нижний индекс
                        </button>
                        
                        <div class="separator"></div>
                        
                        <button onclick="insertText('H₂O')" title="Вода">H₂O</button>
                        <button onclick="insertText('CO₂')" title="Углекислый газ">CO₂</button>
                        <button onclick="insertText('H₂SO₄')" title="Серная кислота">H₂SO₄</button>
                        <button onclick="insertText('10²³')" title="10 в степени 23">10²³</button>
                        <button onclick="insertText('Nₐ')" title="Число Авогадро">Nₐ</button>
                        
                        <div class="separator"></div>
                        
                        <button onclick="insertText('·')" title="Умножение">·</button>
                        <button onclick="insertText('→')" title="Реакция">→</button>
                        <button onclick="insertText('⇌')" title="Равновесие">⇌</button>
                        <button onclick="insertText('±')" title="Плюс-минус">±</button>
                        <button onclick="insertText('≈')" title="Примерно">≈</button>
                        <button onclick="insertText('≠')" title="Не равно">≠</button>
                        <button onclick="insertText('Δ')" title="Дельта">Δ</button>
                        <button onclick="insertText('√')" title="Корень">√</button>
                        <button onclick="insertText('°C')" title="Градусы Цельсия">°C</button>
                    </div>
                    
                    <textarea id="textEditor" placeholder="Текст появится после OCR распознавания...">{st.session_state.ocr_text}</textarea>
                    
                    <div class="info">
                        💡 <b>Инструкция:</b> Выделите текст мышкой → нажмите "ⁿ Верхний индекс" или "ₙ Нижний индекс" → выделенный текст преобразуется!
                    </div>
                </div>
                
                <script>
                    const editor = document.getElementById('textEditor');
                    
                    // Таблицы конвертации
                    const superscriptMap = {{
                        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
                        'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ', 'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ',
                        'j': 'ʲ', 'k': 'ᵏ', 'l': 'ˡ', 'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ', 'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ',
                        't': 'ᵗ', 'u': 'ᵘ', 'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ'
                    }};
                    
                    const subscriptMap = {{
                        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
                        '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
                        'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ', 'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ',
                        'n': 'ₙ', 'o': 'ₒ', 'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ', 'v': 'ᵥ', 'x': 'ₓ'
                    }};
                    
                    function convertToScript(text, map) {{
                        return text.split('').map(char => map[char] || char).join('');
                    }}
                    
                    function applyFormat(type) {{
                        const start = editor.selectionStart;
                        const end = editor.selectionEnd;
                        const selectedText = editor.value.substring(start, end);
                        
                        if (!selectedText) {{
                            alert('⚠️ Сначала выделите текст мышкой!');
                            return;
                        }}
                        
                        const map = type === 'superscript' ? superscriptMap : subscriptMap;
                        const converted = convertToScript(selectedText, map);
                        
                        // Заменяем выделенный текст
                        const before = editor.value.substring(0, start);
                        const after = editor.value.substring(end);
                        editor.value = before + converted + after;
                        
                        // Восстанавливаем курсор после конвертированного текста
                        const newPos = start + converted.length;
                        editor.setSelectionRange(newPos, newPos);
                        editor.focus();
                        
                        // Отправляем обновление в Streamlit
                        syncToStreamlit();
                    }}
                    
                    function insertText(text) {{
                        const start = editor.selectionStart;
                        const before = editor.value.substring(0, start);
                        const after = editor.value.substring(start);
                        editor.value = before + text + after;
                        
                        const newPos = start + text.length;
                        editor.setSelectionRange(newPos, newPos);
                        editor.focus();
                        
                        syncToStreamlit();
                    }}
                    
                    function syncToStreamlit() {{
                        // Обновляем скрытое поле для синхронизации
                        const hiddenField = window.parent.document.querySelector('textarea[data-testid="stTextArea"]');
                        if (hiddenField) {{
                            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                            nativeInputValueSetter.call(hiddenField, editor.value);
                            const event = new Event('input', {{ bubbles: true}});
                            hiddenField.dispatchEvent(event);
                        }}
                    }}
                    
                    // Синхронизация при изменении текста
                    let timeout;
                    editor.addEventListener('input', () => {{
                        clearTimeout(timeout);
                        timeout = setTimeout(syncToStreamlit, 500);
                    }});
                    
                    // Копируем текст в буфер обмена при нажатии Ctrl+C
                    editor.addEventListener('keydown', (e) => {{
                        if ((e.ctrlKey || e.metaKey) && e.key === 'c') {{
                            const selectedText = editor.value.substring(editor.selectionStart, editor.selectionEnd);
                            if (selectedText) {{
                                navigator.clipboard.writeText(selectedText).catch(() => {{}});
                            }}
                        }}
                    }});
                </script>
                """
                
                # Встраиваем HTML компонент
                import streamlit.components.v1 as components
                components.html(html_editor, height=480, scrolling=False)
                
                # Дополнительное поле для ручной синхронизации (если JS не сработал)
                with st.expander("🔄 Ручная синхронизация (если нужно)", expanded=False):
                    st.caption("Если изменения в редакторе выше не сохраняются, скопируйте текст сюда:")
                    manual_sync = st.text_area(
                        "Синхронизация",
                        st.session_state.ocr_text,
                        height=100,
                        key="manual_sync_textarea",
                        help="Вставьте текст из редактора выше"
                    )
                    if st.button("💾 Применить изменения", width='stretch'):
                        st.session_state.ocr_text = manual_sync
                        st.success("✅ Текст обновлён!")
                        st.rerun()
                
                # Статистика (используем st.session_state.ocr_text)
                text = st.session_state.ocr_text
                lines_count = len([l for l in text.split('\n') if l.strip()])
                st.caption(f"📊 Строк: {lines_count} | Символов: {len(text)}")
                
                # Кнопка парсинга
                if st.button("➡️ Извлечь задачи", type="primary", width='stretch'):
                    with st.spinner("Извлекаю задачи..."):
                        tasks = parse_tasks_from_text(st.session_state.ocr_text)
                        st.session_state.parsed_tasks = tasks
                        st.success(f"✅ Найдено задач: {len(tasks)}")
                        st.rerun()
            else:
                st.info("Нажмите 'Распознать текст' слева")

# ═══════════════════════════════════════════════════════════════
# Вкладка 3: Редактирование задач
# ═══════════════════════════════════════════════════════════════

with tab3:
    st.header("✏️ Редактирование задач")
    
    if not st.session_state.parsed_tasks:
        st.warning("⚠️ Сначала извлеките задачи во вкладке 2")
    else:
        st.info(f"📋 Найдено задач: {len(st.session_state.parsed_tasks)}")
        
        # Редактирование каждой задачи
        for i, task in enumerate(st.session_state.parsed_tasks):
            with st.expander(f"Задача {i+1}: {task['question'][:50]}...", expanded=(i == 0)):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Вопрос
                    new_question = st.text_area(
                        "Вопрос",
                        task['question'],
                        key=f"q_{i}",
                        height=100
                    )
                    
                    # Решение
                    new_solution = st.text_area(
                        "Решение (необязательно)",
                        task.get('solution', ''),
                        key=f"s_{i}",
                        height=100
                    )
                    
                    # Ответ
                    new_answer = st.text_input(
                        "Ответ",
                        task.get('answer', ''),
                        key=f"a_{i}"
                    )
                
                with col2:
                    # Категория
                    new_category = st.selectbox(
                        "Категория",
                        TASK_CATEGORIES,
                        index=TASK_CATEGORIES.index(task.get('category', 'другое')),
                        key=f"cat_{i}"
                    )
                    
                    # Сложность
                    new_difficulty = st.selectbox(
                        "Сложность",
                        DIFFICULTY_LEVELS,
                        index=DIFFICULTY_LEVELS.index(task.get('difficulty', 'medium')),
                        key=f"diff_{i}"
                    )
                    
                    # Ключевые слова
                    keywords_str = st.text_input(
                        "Ключевые слова (через запятую)",
                        ", ".join(task.get('keywords', [])),
                        key=f"kw_{i}"
                    )
                    
                    # Кнопки действий
                    col_save, col_del = st.columns(2)
                    
                    with col_save:
                        if st.button("💾 Сохранить", key=f"save_{i}", width='stretch'):
                            st.session_state.parsed_tasks[i] = {
                                'question': new_question,
                                'solution': new_solution,
                                'answer': new_answer,
                                'category': new_category,
                                'difficulty': new_difficulty,
                                'keywords': [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                            }
                            st.success("✅ Сохранено!")
                    
                    with col_del:
                        if st.button("🗑️ Удалить", key=f"del_{i}", width='stretch'):
                            st.session_state.parsed_tasks.pop(i)
                            st.rerun()
        
        # Добавить новую задачу
        st.divider()
        if st.button("➕ Добавить задачу вручную", width='stretch'):
            st.session_state.parsed_tasks.append({
                'question': '',
                'solution': '',
                'answer': '',
                'category': 'другое',
                'difficulty': 'medium',
                'keywords': []
            })
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# Вкладка 4: Сохранение JSON
# ═══════════════════════════════════════════════════════════════

with tab4:
    st.header("💾 Сохранение JSON")
    
    if not st.session_state.parsed_tasks:
        st.warning("⚠️ Нет задач для сохранения")
    else:
        st.info(f"📋 Готово к сохранению: {len(st.session_state.parsed_tasks)} задач")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Предпросмотр JSON")
            
            # Имя файла
            default_filename = f"chemistry_tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filename = st.text_input(
                "Имя файла",
                default_filename,
                help="Без расширения .json"
            )
            
            if not filename.endswith('.json'):
                filename += '.json'
            
            # Предпросмотр
            preview_data = {
                "dataset_name": filename.replace('.json', ''),
                "created_at": datetime.now().isoformat(),
                "total_tasks": len(st.session_state.parsed_tasks),
                "tasks": st.session_state.parsed_tasks[:3]  # Первые 3 для превью
            }
            
            st.json(preview_data, expanded=True)
            
            if len(st.session_state.parsed_tasks) > 3:
                st.caption(f"... и ещё {len(st.session_state.parsed_tasks) - 3} задач")
        
        with col2:
            st.subheader("Действия")
            
            # Сохранить JSON
            if st.button("💾 Сохранить JSON", type="primary", width='stretch'):
                try:
                    json_path = save_tasks_to_json(st.session_state.parsed_tasks, filename)
                    st.success(f"✅ JSON сохранён: {json_path.name}")
                    
                    # Показываем путь
                    st.code(str(json_path))
                    
                except Exception as e:
                    st.error(f"❌ Ошибка сохранения: {e}")
            
            st.divider()
            
            # Копировать в data/ для обучения
            st.subheader("Копировать для обучения")
            st.info("Скопирует JSON в папку data/ для использования в обучении модели")
            
            json_files = list(JSON_OUTPUT_DIR.glob("*.json"))
            if json_files:
                selected_json = st.selectbox(
                    "Выберите JSON файл",
                    [f.name for f in json_files],
                    index=len(json_files) - 1  # Последний файл
                )
                
                if st.button("📋 Копировать в data/", width='stretch'):
                    try:
                        source = JSON_OUTPUT_DIR / selected_json
                        dest = copy_to_training_data(source)
                        st.success(f"✅ Скопировано в: {dest}")
                        st.code(str(dest))
                    except Exception as e:
                        st.error(f"❌ Ошибка копирования: {e}")
            else:
                st.warning("Нет JSON файлов для копирования")
            
            st.divider()
            
            # Очистить текущую сессию
            if st.button("🔄 Начать новую сессию", width='stretch'):
                st.session_state.current_image = None
                st.session_state.ocr_text = ""
                st.session_state.parsed_tasks = []
                st.success("✅ Сессия очищена!")
                st.rerun()

# ═══════════════════════════════════════════════════════════════
# Вкладка 5: Объединение датасетов
# ═══════════════════════════════════════════════════════════════

with tab5:
    st.header("🔗 Объединение JSON датасетов")
    
    st.markdown("""
    Объедините несколько JSON файлов с задачами в один большой датасет для тренировки.
    
    **Использование:**
    1. Выберите JSON файлы для объединения
    2. Укажите название нового датасета
    3. Нажмите "Объединить"
    """)
    
    # Список всех JSON файлов
    json_files = sorted(JSON_OUTPUT_DIR.glob("*.json"))
    
    if len(json_files) < 2:
        st.warning("⚠️ Нужно минимум 2 JSON файла для объединения. Сначала создайте несколько датасетов во вкладке 4.")
    else:
        st.subheader("📂 Доступные файлы")
        
        # Создаём DataFrame со статистикой файлов
        file_stats = []
        for f in json_files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    tasks_count = data.get('total_tasks', len(data.get('tasks', [])))
                    file_stats.append({
                        'Файл': f.name,
                        'Задач': tasks_count,
                        'Создан': data.get('created_at', 'неизвестно')[:19] if data.get('created_at') else 'неизвестно'
                    })
            except:
                file_stats.append({
                    'Файл': f.name,
                    'Задач': '?',
                    'Создан': 'ошибка чтения'
                })
        
        import pandas as pd
        df = pd.DataFrame(file_stats)
        st.dataframe(df, width='stretch', hide_index=True)
        
        st.divider()
        
        # Выбор файлов для объединения
        st.subheader("🔀 Объединение")
        
        selected_files = st.multiselect(
            "Выберите файлы для объединения (минимум 2):",
            [f.name for f in json_files],
            help="Задачи из всех выбранных файлов будут объединены в один датасет"
        )
        
        if len(selected_files) >= 2:
            # Название нового датасета
            default_name = f"combined_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset_name = st.text_input(
                "Название нового датасета:",
                value=default_name,
                help="Будет сохранено как {название}.json"
            )
            
            # Предпросмотр
            total_tasks = 0
            for fname in selected_files:
                try:
                    with open(JSON_OUTPUT_DIR / fname, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        total_tasks += data.get('total_tasks', len(data.get('tasks', [])))
                except:
                    pass
            
            st.info(f"📊 Итого задач после объединения: **{total_tasks}**")
            
            # Кнопка объединения
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔗 Объединить датасеты", type="primary", width='stretch'):
                    try:
                        # Объединяем файлы
                        combined_tasks = []
                        sources = []
                        
                        for fname in selected_files:
                            filepath = JSON_OUTPUT_DIR / fname
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                tasks = data.get('tasks', [])
                                combined_tasks.extend(tasks)
                                sources.append(fname)
                        
                        # Создаём новый JSON
                        combined_data = {
                            "dataset_name": dataset_name,
                            "created_at": datetime.now().isoformat(),
                            "total_tasks": len(combined_tasks),
                            "source_files": sources,
                            "tasks": combined_tasks
                        }
                        
                        # Сохраняем
                        output_path = JSON_OUTPUT_DIR / f"{dataset_name}.json"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(combined_data, f, ensure_ascii=False, indent=2)
                        
                        st.success(f"✅ Объединено! Создан файл: {output_path.name}")
                        st.success(f"📊 Всего задач: {len(combined_tasks)}")
                        
                        # Предложение скопировать в data/
                        if st.button("📋 Скопировать в data/ для тренировки"):
                            dest = copy_to_training_data(output_path)
                            st.success(f"✅ Скопировано в: {dest}")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка объединения: {e}")
            
            with col2:
                # Опция удаления исходных файлов
                delete_originals = st.checkbox(
                    "🗑️ Удалить исходные файлы после объединения",
                    value=False,
                    help="ВНИМАНИЕ: Исходные файлы будут удалены безвозвратно!"
                )
                
                if delete_originals:
                    st.warning("⚠️ Исходные файлы будут удалены!")
        
        else:
            st.info("👆 Выберите минимум 2 файла для объединения")
        
        st.divider()
        
        # Быстрое объединение всех файлов
        st.subheader("⚡ Быстрые действия")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔗 Объединить ВСЕ файлы", width='stretch'):
                try:
                    all_tasks = []
                    all_sources = []
                    
                    for f in json_files:
                        with open(f, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                            all_tasks.extend(data.get('tasks', []))
                            all_sources.append(f.name)
                    
                    combined_name = f"all_tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    combined_data = {
                        "dataset_name": combined_name,
                        "created_at": datetime.now().isoformat(),
                        "total_tasks": len(all_tasks),
                        "source_files": all_sources,
                        "tasks": all_tasks
                    }
                    
                    output_path = JSON_OUTPUT_DIR / f"{combined_name}.json"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(combined_data, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"✅ Объединено {len(json_files)} файлов!")
                    st.success(f"📊 Всего задач: {len(all_tasks)}")
                    st.info(f"💾 Сохранено: {output_path.name}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
        
        with col2:
            if st.button("📋 Скопировать последний в data/", width='stretch'):
                if json_files:
                    try:
                        latest = max(json_files, key=lambda f: f.stat().st_mtime)
                        dest = copy_to_training_data(latest)
                        st.success(f"✅ Скопировано: {latest.name} → data/")
                    except Exception as e:
                        st.error(f"❌ Ошибка: {e}")

# ═══════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════

st.divider()
st.caption("💡 Подсказка: Сначала загрузите скан → распознайте → отредактируйте → сохраните JSON")
