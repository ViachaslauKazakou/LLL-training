"""
app.py — Streamlit интерфейс для PyTorch LLM

Запуск:
    streamlit run app.py
"""

import streamlit as st
import torch
from pathlib import Path
import json
import inspect
import threading
import time
import math
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import List

from inference import load_model_for_inference, generate_text
from data import CharTokenizer, auto_select_stride, prepare_sample_data
from config import get_small_config, get_medium_config, get_base_config, TrainingConfig, get_device, benchmark_device, get_memory_stats
from utils.pdf_to_text import convert_pdf_to_text
from utils.merge_datasets import analyze_dataset, merge_text_files
from model import GPTModel
from training import Trainer
from data import load_data
from logger import app_logger


# Глобальное хранилище для состояния обучения (доступно из threads)
@dataclass
class TrainingState:
    active: bool = False
    stop_requested: bool = False  # Флаг для остановки обучения
    logs: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=lambda: {
        'epoch': 0,
        'step': 0,
        'train_loss': 0.0,
        'val_loss': 0.0,
        'best_val_loss': float('inf'),
        'progress': 0.0
    })

@st.cache_resource
def _get_training_state() -> TrainingState:
    return TrainingState()

# Синглтон — один объект на всё время жизни сервера, переживает reruns
training_state = _get_training_state()


# Конфигурация страницы
st.set_page_config(
    page_title="PyTorch LLM",
    page_icon="🤖",
    layout="wide"
)


def call_load_data_compat(**kwargs):
    """Вызывает load_data только с поддерживаемыми аргументами текущего рантайма."""
    supported_params = inspect.signature(load_data).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_params}
    dropped_params = sorted(set(kwargs) - set(filtered_kwargs))

    if dropped_params:
        app_logger.warning(
            "load_data в текущем процессе не поддерживает аргументы: %s. "
            "Вероятно, Streamlit держит старый импорт; перезапустите приложение.",
            ", ".join(dropped_params),
        )

    return load_data(**filtered_kwargs)


def estimate_dataset_tokens(stats: dict, tokenizer_type: str) -> int:
    """Грубая оценка числа токенов для UI-подсказок."""
    if tokenizer_type == "char":
        return max(1, stats.get('total_chars', 0))

    if tokenizer_type == "hybrid":
        total_chars = stats.get('total_chars', 0)
        total_words = stats.get('total_words', 0)
        return max(1, int(max(total_words * 1.0, total_chars / 1.6)))

    if tokenizer_type == "bpe":
        total_chars = stats.get('total_chars', 0)
        total_words = stats.get('total_words', 0)
        return max(1, int(max(total_words * 1.2, total_chars / 2.1)))

    # Для tiktoken на русских и технических текстах обычно 1 токен ~= 2-3 символа.
    total_chars = stats.get('total_chars', 0)
    total_words = stats.get('total_words', 0)
    return max(1, int(max(total_words * 1.3, total_chars / 2.2)))


def build_training_data_advice(
    file_path: str,
    tokenizer_type: str,
    context_len: int,
    batch_size: int,
    train_split: float = 0.9
) -> dict | None:
    """Строит рекомендации по гиперпараметрам на основе размера датасета."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return None

    stats = analyze_dataset(path)
    estimated_tokens = estimate_dataset_tokens(stats, tokenizer_type)
    train_tokens = max(1, int(estimated_tokens * train_split))
    val_tokens = max(1, estimated_tokens - train_tokens)

    train_stride = auto_select_stride(train_tokens, context_len, target_windows=1024)
    val_stride = auto_select_stride(val_tokens, context_len, target_windows=256)
    train_windows = max(0, (train_tokens - context_len) // train_stride)
    val_windows = max(0, (val_tokens - context_len) // val_stride)
    train_batches = math.ceil(train_windows / batch_size) if train_windows > 0 else 0

    recommended_context = context_len
    recommended_batch = batch_size

    if stats['size_kb'] < 100:
        recommended_context = min(context_len, 64)
        recommended_batch = min(batch_size, 8)
    elif stats['size_kb'] < 512:
        recommended_context = min(context_len, 128)
        recommended_batch = min(batch_size, 16)

    severity = "success"
    message = "Параметры выглядят адекватно для выбранного корпуса."
    if train_batches < 10:
        severity = "error"
        message = (
            f"Слишком мало train batch'ей: примерно {train_batches}. "
            "Обучение будет шумным и почти без полезной динамики."
        )
    elif train_batches < 30:
        severity = "warning"
        message = (
            f"Train batch'ей мало: примерно {train_batches}. "
            "Лучше уменьшить batch size или context length."
        )
    elif train_batches < 100:
        severity = "info"
        message = (
            f"Train batch'ей умеренно: примерно {train_batches}. "
            "Для стабильности можно слегка уменьшить batch size."
        )

    return {
        'stats': stats,
        'estimated_tokens': estimated_tokens,
        'train_windows': train_windows,
        'val_windows': val_windows,
        'train_batches': train_batches,
        'train_stride': train_stride,
        'val_stride': val_stride,
        'recommended_context': recommended_context,
        'recommended_batch': recommended_batch,
        'severity': severity,
        'message': message,
    }


def get_training_file_options(data_dir: Path) -> tuple[list[str], dict[str, str], int]:
    """Собирает опции файлов для формы обучения."""
    available_files = []

    if data_dir.exists():
        for ext in ["*.txt", "*.json"]:
            available_files.extend(data_dir.glob(ext))

        available_files = sorted(available_files, key=lambda f: f.stat().st_mtime, reverse=True)

    file_options: list[str] = []
    file_paths: dict[str, str] = {}

    for file_path in available_files:
        size_kb = file_path.stat().st_size / 1024
        size_mb = size_kb / 1024
        size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{size_kb:.1f} KB"
        display_name = f"{file_path.name} ({size_str})"
        file_options.append(display_name)
        file_paths[display_name] = str(file_path)

    if file_options:
        file_options.append("✏️ Ввести путь вручную...")

    default_file = "sample.txt"
    default_idx = 0
    for idx, option in enumerate(file_options[:-1] if file_options else []):
        if default_file in option:
            default_idx = idx
            break

    return file_options, file_paths, default_idx


def load_available_checkpoints():
    """Находит все доступные checkpoint файлы."""
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    return sorted([str(cp.relative_to(checkpoint_dir)) for cp in checkpoints])


@st.cache_resource
def load_model_cached(checkpoint_path: str):
    """Кэширует загруженную модель."""
    from config import get_device
    device = get_device()
    
    try:
        model, checkpoint = load_model_for_inference(checkpoint_path, device)
        
        # Создаём токенизатор из checkpoint
        if 'tokenizer_config' in checkpoint:
            # Новый формат: tokenizer_config содержит тип и параметры
            from tokenizer import TikTokenizer, HybridChemTokenizer, BPETokenizer, CharTokenizer as NewCharTokenizer
            tok_config = checkpoint['tokenizer_config']
            if tok_config['type'] == 'tiktoken':
                tokenizer = TikTokenizer.from_dict(tok_config)
                st.info(f"✓ TikTokenizer загружен: {tokenizer.vocab_size()} токенов ({tok_config.get('encoding_name', 'cl100k_base')})")
            elif tok_config['type'] == 'hybrid':
                tokenizer = HybridChemTokenizer.from_dict(tok_config)
                st.info(f"✓ HybridTokenizer загружен: {tokenizer.vocab_size()} токенов")
            elif tok_config['type'] == 'bpe':
                tokenizer = BPETokenizer.from_dict(tok_config)
                st.info(f"✓ BPETokenizer загружен: {tokenizer.vocab_size()} токенов")
            else:
                tokenizer = NewCharTokenizer.from_dict(tok_config)
                st.info(f"✓ CharTokenizer загружен: {tokenizer.vocab_size()} символов")
        elif 'vocab' in checkpoint:
            # Старый формат: vocab напрямую (CharTokenizer legacy)
            tokenizer = CharTokenizer.__new__(CharTokenizer)
            tokenizer.vocab = checkpoint['vocab']
            tokenizer.char_to_idx = {ch: idx for idx, ch in enumerate(tokenizer.vocab)}
            tokenizer.idx_to_char = {idx: ch for ch, idx in tokenizer.char_to_idx.items()}
            st.info(f"✓ CharTokenizer загружен (legacy): {len(tokenizer.vocab)} символов")
        else:
            # Fallback: создаём TikTokenizer по умолчанию
            from tokenizer import TikTokenizer
            tokenizer = TikTokenizer(encoding_name='cl100k_base')
            st.warning(f"⚠️ Checkpoint без токенизатора, создан TikTokenizer: {tokenizer.vocab_size()} токенов")
        
        return model, tokenizer, checkpoint, device
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None


def tab_generation():
    """Вкладка генерации текста."""
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    
    st.header("🎯 Генерация текста")
    
    # Получаем список checkpoints
    checkpoints = load_available_checkpoints()
    
    _gen_source_from_state = st.session_state.get("gen_model_source", "🔥 PyTorch checkpoint")
    if not checkpoints and "MLX" not in _gen_source_from_state:
        st.warning("⚠️ Нет доступных checkpoint файлов в checkpoints/")
        st.info(
            "Сначала обучите модель на вкладке '🎓 Обучение', "
            "или переключитесь на режим '🤖 MLX модель' ниже."
        )
        # Still show model source selector so user can switch to MLX
        st.radio(
            "Источник модели",
            ["🔥 PyTorch checkpoint", "🤖 MLX модель"],
            horizontal=True,
            key="gen_model_source",
        )
        return

    if checkpoints:
        # ═══════════════════════════════════════════════════════
        # ТАБЛИЦА МОДЕЛЕЙ (только для PyTorch)
        # ═══════════════════════════════════════════════════════
        st.subheader("📋 Доступные модели")

        # Собираем данные о всех моделях
        model_data = []
        for ckpt_name in checkpoints:
            ckpt_path = checkpoint_dir / ckpt_name
            try:
                import torch
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

                # Извлекаем dataset name из пути
                dataset_path = checkpoint.get('dataset_path', 'N/A')
                if dataset_path != 'N/A':
                    dataset_name = Path(dataset_path).stem
                else:
                    dataset_name = 'N/A'

                # Дата обучения
                training_date = checkpoint.get('training_date', 'N/A')
                if training_date != 'N/A':
                    # Форматируем дату (убираем время)
                    training_date = training_date.split('T')[0]

                model_data.append({
                    'Модель': ckpt_name,
                    'Датасет': dataset_name[:30] + '...' if len(dataset_name) > 30 else dataset_name,
                    'Step': checkpoint.get('global_step', 0),
                    'Val Loss': checkpoint.get('best_val_loss', float('inf')),
                    'Vocab': checkpoint.get('config').vocab_size if 'config' in checkpoint else 'N/A',
                    'Дата': training_date,
                    'Размер': f"{ckpt_path.stat().st_size / 1024 / 1024:.1f} MB",
                    '_path': str(ckpt_path)
                })
            except Exception as e:
                model_data.append({
                    'Модель': ckpt_name,
                    'Датасет': 'error',
                    'Step': 'error',
                    'Val Loss': float('inf'),
                    'Vocab': 'N/A',
                    'Дата': 'N/A',
                    'Размер': f"{ckpt_path.stat().st_size / 1024 / 1024:.1f} MB",
                    '_path': str(ckpt_path)
                })

        # Сортируем по Val Loss
        model_data_sorted = sorted(model_data, key=lambda x: x['Val Loss'])

        # Отображаем таблицу
        import pandas as pd
        df = pd.DataFrame([{k: v for k, v in m.items() if not k.startswith('_')} for m in model_data_sorted])

        # Форматируем Val Loss
        df['Val Loss'] = df['Val Loss'].apply(lambda x: f"{x:.4f}" if x != float('inf') else "∞")

        # Добавляем индикатор лучшей модели
        best_idx = 0
        df.insert(0, '⭐', ['🏆' if i == best_idx else '' for i in range(len(df))])

        st.dataframe(
            df,
            width='stretch',
            hide_index=True
        )

        st.caption("🏆 = лучший val_loss")

        # ═══════════════════════════════════════════════════════
        # УПРАВЛЕНИЕ МОДЕЛЯМИ
        # ═══════════════════════════════════════════════════════
        st.divider()
        st.subheader("🔧 Управление моделями")

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_for_action = st.selectbox(
                "Выберите модель",
                [m['Модель'] for m in model_data_sorted],
                key="action_select"
            )

        with col2:
            # Кнопка "Сделать базовой"
            if st.button("⭐ Сделать базовой", width='stretch'):
                if selected_for_action != "best_model.pt":
                    try:
                        import shutil
                        src = checkpoint_dir / selected_for_action
                        dst = checkpoint_dir / "best_model.pt"
                        shutil.copy2(src, dst)
                        st.success(f"✅ {selected_for_action} → best_model.pt")
                        app_logger.info(f"UI Модель {selected_for_action} установлена как базовая")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
                else:
                    st.info("Эта модель уже базовая!")

        with col3:
            # Кнопка удаления
            if st.button("🗑️ Удалить", width='stretch', type="secondary"):
                if selected_for_action == "best_model.pt":
                    st.error("❌ Нельзя удалить best_model.pt!")
                else:
                    try:
                        (checkpoint_dir / selected_for_action).unlink()
                        st.success(f"✅ {selected_for_action} удалён")
                        app_logger.info(f"UI Модель {selected_for_action} удалена")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
    
    # ═══════════════════════════════════════════════════════
    # ВЫБОР МОДЕЛИ ДЛЯ ГЕНЕРАЦИИ
    # ═══════════════════════════════════════════════════════
    st.divider()
    st.subheader("💬 Генерация текста")

    gen_model_source = st.radio(
        "Источник модели",
        ["🔥 PyTorch checkpoint", "🤖 MLX модель"],
        horizontal=True,
        key="gen_model_source",
        help=(
            "**PyTorch checkpoint** — использовать модель, обученную с нуля или дообученную в этом приложении.\n\n"
            "**MLX модель** — использовать готовую предобученную модель (Llama, Qwen, ...) "
            "с опциональным LoRA адаптером. Работает только на Apple Silicon."
        ),
    )
    is_mlx_gen = "MLX" in gen_model_source

    if is_mlx_gen:
        # ═══════════════════════════════════════════════════════
        # MLX GENERATION
        # ═══════════════════════════════════════════════════════
        from mlx_integration import (
            MLX_AVAILABLE, CURATED_MODELS, get_model_display_label,
            load_mlx_model, mlx_generate,
        )

        if not MLX_AVAILABLE:
            st.error("❌ mlx-lm не установлен. Выполните: `poetry add mlx-lm`")
        else:
            # Local fused models from models/ directory
            _models_dir = Path(__file__).parent / "models"
            _local_models = []
            if _models_dir.exists():
                _local_models = [
                    d for d in sorted(_models_dir.iterdir())
                    if d.is_dir() and (d / "config.json").exists()
                ]

            # Build combined options: local models first, then curated HF models
            _local_labels = [f"📁 {d.name}" for d in _local_models]
            _hf_labels = [get_model_display_label(m) for m in CURATED_MODELS]
            _all_labels = _local_labels + _hf_labels

            mlx_gen_selected_label = st.selectbox(
                "Модель",
                _all_labels,
                index=0,
                key="mlx_gen_model_select",
                help="📁 = локальная слитая модель | ✅/⬇️ = HF модель из кеша/для скачивания",
            )

            _sel_idx = _all_labels.index(mlx_gen_selected_label)
            if _sel_idx < len(_local_models):
                mlx_gen_model_id = str(_local_models[_sel_idx])
                st.caption(f"📁 `{mlx_gen_model_id}`")
            else:
                mlx_gen_model_id = CURATED_MODELS[_sel_idx - len(_local_models)].hf_id

            use_custom_gen_id = st.checkbox("Свой HF ID или путь", key="mlx_gen_custom_id")
            if use_custom_gen_id:
                mlx_gen_model_id = st.text_input(
                    "HuggingFace model ID или локальный путь",
                    value=mlx_gen_model_id,
                    key="mlx_gen_model_id_input",
                )

            # Adapters list
            adapters_dir = Path(__file__).parent / "adapters"
            available_adapters = []
            if adapters_dir.exists():
                available_adapters = [
                    d.name for d in adapters_dir.iterdir()
                    if d.is_dir() and any(d.glob("*.safetensors"))
                ]

            mlx_gen_adapter_path = None
            if available_adapters:
                use_adapter = st.checkbox(
                    "Использовать LoRA адаптер",
                    value=False,
                    key="mlx_gen_use_adapter",
                    help="Адаптер из папки adapters/ — результат MLX fine-tuning",
                )
                if use_adapter:
                    selected_adapter = st.selectbox(
                        "Выберите адаптер",
                        available_adapters,
                        key="mlx_gen_adapter_select",
                    )
                    mlx_gen_adapter_path = str(adapters_dir / selected_adapter)
            else:
                st.caption("ℹ️ Нет обученных адаптеров (adapters/). Сначала запустите MLX Fine-tuning.")

            # Generation params
            mlx_gen_prompt = st.text_area(
                "Промпт",
                value="Расскажи о реакции нейтрализации.",
                height=100,
                key="mlx_gen_prompt",
            )

            mlx_gen_system_prompt = st.text_input(
                "System prompt (опционально)",
                value="",
                key="mlx_gen_system",
                placeholder="Ты — эксперт-химик.",
            )

            gcol1, gcol2, gcol3 = st.columns(3)
            with gcol1:
                mlx_gen_max_tokens = st.slider(
                    "Max токенов", min_value=50, max_value=1000, value=200, step=50,
                    key="mlx_gen_max_tokens",
                )
            with gcol2:
                mlx_gen_temperature = st.slider(
                    "Temperature", min_value=0.1, max_value=1.0, value=0.8, step=0.1,
                    key="mlx_gen_temp",
                )
            with gcol3:
                mlx_gen_top_p = st.slider(
                    "Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05,
                    key="mlx_gen_top_p",
                )

            mlx_gen_rep_penalty = st.slider(
                "Repetition penalty", min_value=1.0, max_value=2.0, value=1.1, step=0.05,
                key="mlx_gen_rep_penalty",
                help=(
                    "Штраф за повторение уже сгенерированных токенов. "
                    "1.0 = без штрафа, 1.1–1.3 = умеренный штраф (рекомендуется)."
                ),
            )

            mlx_gen_use_chat_template = st.checkbox(
                "Применять chat template",
                value=True,
                key="mlx_gen_use_chat_template",
                help=(
                    "Включено — оборачивает промпт в формат модели (нужно для Instruct/Chat моделей).\n\n"
                    "Выключи для base моделей или если хочешь передать промпт напрямую."
                ),
            )

            if st.button("✨ Сгенерировать (MLX)", type="primary", width="stretch", key="mlx_gen_btn"):
                if not mlx_gen_prompt.strip():
                    st.warning("Введите промпт!")
                else:
                    # Cache model in session_state
                    cache_key = f"mlx_model_{mlx_gen_model_id}_{mlx_gen_adapter_path}"
                    if st.session_state.get("mlx_gen_cache_key") != cache_key:
                        with st.spinner(f"Загрузка модели {mlx_gen_model_id}..."):
                            try:
                                loaded_model, loaded_tokenizer = load_mlx_model(
                                    mlx_gen_model_id, mlx_gen_adapter_path
                                )
                                st.session_state["mlx_gen_model"] = loaded_model
                                st.session_state["mlx_gen_tokenizer"] = loaded_tokenizer
                                st.session_state["mlx_gen_cache_key"] = cache_key
                                st.success("✅ Модель загружена")
                            except Exception as e:
                                st.error(f"Ошибка загрузки: {e}")
                                st.stop()

                    with st.spinner("Генерация..."):
                        try:
                            generated, stats = mlx_generate(
                                st.session_state["mlx_gen_model"],
                                st.session_state["mlx_gen_tokenizer"],
                                mlx_gen_prompt,
                                max_tokens=mlx_gen_max_tokens,
                                temperature=mlx_gen_temperature,
                                top_p=mlx_gen_top_p,
                                repetition_penalty=mlx_gen_rep_penalty,
                                system_prompt=mlx_gen_system_prompt,
                                use_chat_template=mlx_gen_use_chat_template,
                            )
                            st.success("✅ Готово!")
                            sc1, sc2, sc3, sc4 = st.columns(4)
                            with sc1:
                                st.metric("Время", f"{stats['total_time']:.2f}s")
                            with sc2:
                                st.metric("Скорость", f"{stats['tokens_per_second']:.1f} tok/s")
                            with sc3:
                                st.metric("Токенов", stats['tokens_generated'])
                            with sc4:
                                st.metric("Промпт", stats['prompt_tokens'])

                            st.text_area("Результат", value=generated, height=300, disabled=True)
                            st.code(generated, language=None)

                            if "generation_history" not in st.session_state:
                                st.session_state.generation_history = []
                            st.session_state.generation_history.append(
                                (mlx_gen_prompt, generated, stats)
                            )
                        except Exception as e:
                            st.error(f"Ошибка генерации: {e}")
        return  # skip PyTorch generation section

    if not checkpoints:
        return

    selected_checkpoint = st.selectbox(
        "Модель для генерации",
        [m['Модель'] for m in model_data_sorted],
        index=0,
        key="gen_select"
    )
    
    checkpoint_path = str(checkpoint_dir / selected_checkpoint)
    
    # Загружаем модель
    if st.button("🔄 Загрузить модель", type="primary"):
        with st.spinner("Загрузка модели..."):
            model, tokenizer, checkpoint, device = load_model_cached(checkpoint_path)
            
            if model is not None:
                st.success(f"✅ Модель загружена на {device}")
                
                # Инфо о модели
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Параметров", f"{model.count_parameters():,}")
                with col2:
                    st.metric("Global step", f"{checkpoint.get('global_step', 'N/A'):,}")
                with col3:
                    val_loss = checkpoint.get('best_val_loss', float('inf'))
                    st.metric("Best val loss", f"{val_loss:.4f}" if val_loss != float('inf') else "N/A")
    
    # Проверяем, загружена ли модель
    try:
        model, tokenizer, checkpoint, device = load_model_cached(checkpoint_path)
        if model is None:
            return
    except:
        st.info("👆 Нажмите 'Загрузить модель' для начала работы")
        return
    
    st.divider()
    
    # Интерфейс генерации
    st.subheader("💬 Генерация")
    
    prompt = st.text_area(
        "Введите начальный текст (промпт)",
        value="Что такое молярная масса?",
        height=100
    )
    
    # Параметры генерации
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_tokens = st.slider(
            "Максимум новых токенов",
            min_value=10,
            max_value=1000,
            value=200,
            step=10,
            help=(
                "Максимальное количество токенов, которое модель сгенерирует в ответ на промпт. "
                "Генерация останавливается раньше, если модель выберет токен конца текста.\n\n"
                "Один токен ≈ 1 символ (char-токенизатор) или 3–6 символов (BPE/tiktoken).\n\n"
                "Рекомендации:\n"
                "• 50–100 — короткие ответы, завершение фразы\n"
                "• 200 — стандартный абзац (default)\n"
                "• 300–500 — развёрнутый ответ\n"
                "• 500+ — длинный текст, эссе\n\n"
                "⚠️ Модель не может выйти за пределы context_len, с которым обучалась. "
                "Длинные запросы + длинная генерация = обрезание начала промпта."
            )
        )
    
    with col2:
        temperature = st.slider(
            "Temperature (креативность)",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help=(
                "Управляет «случайностью» при выборе следующего токена.\n\n"
                "Перед сэмплированием логиты делятся на temperature. "
                "Это сжимает или растягивает распределение вероятностей:\n\n"
                "**Низкая температура (0.1–0.4)** → распределение острее, "
                "модель почти всегда выбирает наиболее вероятный токен. "
                "Текст предсказуемый, повторяющийся, фактический.\n\n"
                "**Средняя температура (0.5–0.7)** → баланс между связностью "
                "и разнообразием. Рекомендуется для большинства задач.\n\n"
                "**Высокая температура (0.8–1.0)** → распределение сглаживается, "
                "модель чаще выбирает неожиданные токены. "
                "Текст креативный, но может терять смысл.\n\n"
                "💡 Используйте вместе с Top-k и Top-p для тонкой настройки генерации."
            )
        )
    
    with col3:
        top_k = st.slider(
            "Top-k sampling",
            min_value=1,
            max_value=100,
            value=50,
            step=5,
            help=(
                "Ограничивает выбор следующего токена топ-K наиболее вероятными вариантами. "
                "Все остальные токены получают вероятность 0.\n\n"
                "**Маленький k (5–20)** → текст связный, предсказуемый, мало разнообразия.\n"
                "**Большой k (50–100)** → больше вариантов, текст разнообразнее, "
                "но могут проскакивать неуместные слова.\n\n"
                "Рекомендации:\n"
                "• 10–20 — для фактических, структурированных текстов\n"
                "• 40–50 — хороший баланс (default)\n"
                "• 80–100 — максимальное разнообразие\n\n"
                "💡 Top-k и Top-p работают вместе: сначала применяется Top-k, "
                "затем из оставшихся токенов Top-p дополнительно обрезает «хвост»."
            )
        )

    col4, col5 = st.columns(2)
    with col4:
        top_p = st.slider(
            "Top-p (nucleus sampling)",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help=(
                "Nucleus sampling: выбирает из минимального набора токенов, "
                "чья суммарная вероятность ≥ top_p. Отсекает маловероятный «хвост».\n\n"
                "В отличие от Top-k (фиксированное число токенов), Top-p адаптируется: "
                "когда модель уверена — берёт мало токенов, когда неуверена — больше.\n\n"
                "**0.7–0.8** → консервативно, только самые вероятные варианты.\n"
                "**0.9** → хороший баланс, обрезает явно неуместные токены (default).\n"
                "**0.95–1.0** → почти без фильтрации, максимальное разнообразие.\n\n"
                "Рекомендации:\n"
                "• 0.8 — структурированные, точные тексты\n"
                "• 0.9 — универсальный вариант\n"
                "• 1.0 — отключить Top-p, полагаться только на Top-k и Temperature\n\n"
                "💡 Top-p применяется после Top-k как дополнительный фильтр."
            )
        )
    with col5:
        st.caption("")  # spacer

    # Генерация
    if st.button("✨ Сгенерировать", type="primary", width='stretch'):
        if not prompt.strip():
            st.warning("Введите промпт!")
            return

        app_logger.info(
            f"UI Генерация запущена: prompt='{prompt[:50]}...', "
            f"max_tokens={max_tokens}, temp={temperature}, top_k={top_k}, top_p={top_p}"
        )

        with st.spinner("Генерация..."):
            try:
                generated, stats = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    device=device
                )
                
                app_logger.info(f"UI Генерация завершена: {len(generated)} символов, {stats['tokens_per_second']:.2f} tok/s")
                st.success("✅ Готово!")
                
                # Показываем статистику
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Время генерации", f"{stats['total_time']:.2f} сек")
                with col2:
                    st.metric("Скорость", f"{stats['tokens_per_second']:.1f} tok/s")
                with col3:
                    st.metric("Сгенерировано токенов", stats['tokens_generated'])
                with col4:
                    st.metric("Токенов в промпте", stats['prompt_tokens'])
                
                # Сохраняем в историю
                if 'generation_history' not in st.session_state:
                    st.session_state.generation_history = []
                st.session_state.generation_history.append((prompt, generated, stats))
                
                # Показываем результат
                st.text_area(
                    "Результат",
                    value=generated,
                    height=300,
                    disabled=True
                )
                
                # Кнопка копирования
                st.code(generated, language=None)
                
            except Exception as e:
                st.error(f"Ошибка генерации: {e}")
    
    # История генераций (если есть в session_state)
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if st.session_state.generation_history:
        st.divider()
        
        # Кнопка очистки истории и кэша
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📜 История генераций")
        with col2:
            if st.button("🗑️ Очистить", help="Очистить историю и кэш"):
                st.session_state.generation_history = []
                st.cache_data.clear()
                st.rerun()
        
        for i, item in enumerate(reversed(st.session_state.generation_history[-5:])):
            # Поддержка старого формата (2 элемента) и нового (3 элемента)
            if len(item) == 2:
                p, g = item
                stats = None
            else:
                p, g, stats = item
            
            idx = len(st.session_state.generation_history) - i
            with st.expander(f"#{idx}: {p[:50]}..."):
                st.text(g)
                if stats:
                    st.caption(f"⚡ {stats['tokens_per_second']:.1f} tok/s • {stats['total_time']:.2f}s • {stats['tokens_generated']} tokens")


def tab_training():
    """Вкладка обучения модели."""
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    
    st.header("🎓 Обучение модели")
    
    # Две колонки: настройки и запуск / логи и прогресс
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("⚙️ Настройки")
        
        # Режим обучения
        training_mode = st.radio(
            "Режим обучения",
            ["🆕 С нуля", "🔄 Дообучение (Fine-tuning)", "🤖 MLX Fine-tuning"],
            disabled=training_state.active,
            help=(
                "**С нуля** — обучить новую GPT-модель на PyTorch с нуля.\n\n"
                "**Дообучение** — продолжить обучение существующего PyTorch checkpoint.\n\n"
                "**MLX Fine-tuning** — LoRA fine-tuning готовой предобученной модели (Llama, Qwen, ...) "
                "на Apple Silicon через mlx-lm. Быстрее и эффективнее для специализации."
            )
        )

        is_finetuning = "Дообучение" in training_mode
        is_mlx = "MLX" in training_mode

        # ═══════════════════════════════════════════════════════
        # MLX Fine-tuning UI
        # ═══════════════════════════════════════════════════════
        # Default MLX variable values (used in run_training closure)
        mlx_model_id = None
        mlx_adapter_path = None
        mlx_data_dir = None
        mlx_source_path = None
        mlx_lora_layers = 8
        mlx_lora_rank = 8
        mlx_lora_scale = 20.0
        mlx_lora_dropout = 0.0
        mlx_lr = 2e-5
        mlx_batch_size_val = 4
        mlx_iters = 500
        mlx_max_seq_length = 1024
        mlx_steps_per_eval = 100
        mlx_save_every = 100
        mlx_grad_checkpoint = False
        mlx_format = "completion"
        mlx_system_prompt = ""
        mlx_clean_forum_data = False
        mlx_resume_adapter = None

        if is_mlx:
            from mlx_integration import (
                MLX_AVAILABLE, CURATED_MODELS, get_model_display_label,
                DataPrepConfig, prepare_mlx_dataset,
            )

            if not MLX_AVAILABLE:
                st.error(
                    "❌ **mlx-lm не установлен.**\n\n"
                    "Установите командой:\n```\npoetry add mlx-lm\n```\n\n"
                    "MLX работает только на Apple Silicon (M1/M2/M3/M4)."
                )
            else:
                st.info("🍎 MLX Fine-tuning работает на Apple Silicon через LoRA адаптеры.")

                # --- Выбор базовой модели ---
                st.markdown("**🤖 Базовая модель**")

                # Локальные модели из models/
                _train_models_dir = Path(__file__).parent / "models"
                _local_train_models = sorted([
                    d for d in _train_models_dir.iterdir()
                    if d.is_dir() and (d / "config.json").exists()
                ]) if _train_models_dir.exists() else []

                _local_train_labels = [f"📁 {d.name}" for d in _local_train_models]
                _hf_train_labels = [get_model_display_label(m) for m in CURATED_MODELS]
                _all_train_labels = _local_train_labels + _hf_train_labels

                selected_model_label = st.selectbox(
                    "Выберите модель",
                    _all_train_labels,
                    index=0,
                    disabled=training_state.active,
                    help=(
                        "📁 = локальная модель из папки models/ (слитая или дообученная)\n\n"
                        "✅ = HF модель уже в кеше | ⬇️ = будет скачана при запуске\n\n"
                        "Рекомендации:\n"
                        "• Qwen2.5 1.5B или SmolLM2 1.7B — лучший баланс для форумных данных\n"
                        "• Llama 3.2 1B — самый лёгкий вариант (~700 MB)\n"
                        "• Модели 3B+ — лучшее качество, но больше памяти"
                    ),
                )
                _sel_train_idx = _all_train_labels.index(selected_model_label)
                if _sel_train_idx < len(_local_train_models):
                    mlx_model_id = str(_local_train_models[_sel_train_idx])
                    st.caption(f"📁 Локальная модель: `{mlx_model_id}`")
                else:
                    selected_model = CURATED_MODELS[_sel_train_idx - len(_local_train_models)]
                    mlx_model_id = selected_model.hf_id
                    st.caption(f"📋 {selected_model.description}")
                    st.caption(f"🔗 HF ID: `{mlx_model_id}` | Контекст: {selected_model.context_length:,} токенов")

                use_custom_id = st.checkbox(
                    "Использовать свой HuggingFace ID",
                    value=False,
                    disabled=training_state.active,
                    help="Любая mlx-lm совместимая модель с HuggingFace (например mlx-community/...)",
                )
                if use_custom_id:
                    mlx_model_id = st.text_input(
                        "HuggingFace model ID",
                        value=mlx_model_id,
                        disabled=training_state.active,
                        placeholder="mlx-community/Llama-3.2-1B-Instruct-4bit",
                    )

                # --- Данные ---
                st.divider()
                st.markdown("**📁 Данные для fine-tuning**")

                mlx_data_dir_obj = Path(__file__).parent / "data"
                mlx_file_options, mlx_file_paths, mlx_default_idx = get_training_file_options(mlx_data_dir_obj)

                if mlx_file_options:
                    mlx_selected_option = st.selectbox(
                        "Исходный файл",
                        mlx_file_options,
                        index=mlx_default_idx,
                        disabled=training_state.active,
                        help="Файл .txt или .json из директории data/ для конвертации в JSONL",
                        key="mlx_data_select",
                    )
                    if mlx_selected_option == "✏️ Ввести путь вручную...":
                        mlx_source_path = st.text_input(
                            "Путь к файлу",
                            value="data/",
                            disabled=training_state.active,
                            key="mlx_manual_data_path",
                        )
                    else:
                        mlx_source_path = mlx_file_paths.get(mlx_selected_option, "")
                        st.caption(f"📁 Путь: `{mlx_source_path}`")
                else:
                    mlx_source_path = st.text_input(
                        "Путь к данным",
                        value="data/sample.txt",
                        disabled=training_state.active,
                    )

                col_fmt1, col_fmt2 = st.columns(2)
                with col_fmt1:
                    mlx_format = st.radio(
                        "Формат JSONL",
                        ["completion", "chat"],
                        horizontal=True,
                        disabled=training_state.active,
                        help=(
                            "**completion** — `{\"text\": \"...\"}`\n"
                            "Весь текст как одна последовательность. Проще, подходит для общего fine-tuning.\n\n"
                            "**chat** — `{\"messages\": [{role, content}, ...]}`\n"
                            "Структура диалога system/user/assistant. Лучше для instruction-следования."
                        ),
                    )
                with col_fmt2:
                    mlx_clean_forum_data = st.checkbox(
                        "🧹 Очистить форумный шум",
                        value=False,
                        disabled=training_state.active,
                        help="Удаляет никнеймы и форумные метки перед конвертацией",
                    )

                if mlx_format == "chat":
                    mlx_system_prompt = st.text_area(
                        "System prompt (опционально)",
                        value="",
                        height=80,
                        disabled=training_state.active,
                        placeholder="Например: Ты опытный участник форума по химии.",
                        help="Добавляется как system-сообщение в каждый chat-пример.",
                    )
                else:
                    mlx_system_prompt = ""

                mlx_max_seq_length = st.number_input(
                    "Max sequence length",
                    min_value=128,
                    max_value=4096,
                    value=1024,
                    step=128,
                    disabled=training_state.active,
                    help=(
                        "Максимальная длина последовательности токенов в одном примере.\n\n"
                        "Тексты длиннее этого значения будут разбиты на несколько примеров.\n\n"
                        "Рекомендации:\n"
                        "• 512–1024 — для коротких форумных сообщений\n"
                        "• 2048 — для длинных текстов\n\n"
                        "⚠️ Большие значения требуют больше памяти при обучении."
                    ),
                )

                # JSONL output directory
                mlx_data_dir = str(
                    Path(__file__).parent / "data" / "mlx_jsonl" /
                    (Path(mlx_source_path).stem if mlx_source_path else "dataset")
                )
                mlx_train_jsonl = Path(mlx_data_dir) / "train.jsonl"
                mlx_data_prepped = mlx_train_jsonl.exists()

                if mlx_data_prepped:
                    line_count = sum(1 for _ in open(mlx_train_jsonl, encoding="utf-8"))
                    st.success(f"✅ JSONL готов: `{mlx_data_dir}` ({line_count} train примеров)")
                else:
                    st.warning(f"⚠️ JSONL не подготовлен. Нажмите кнопку ниже.")

                if st.button(
                    "🔄 Подготовить данные → JSONL",
                    disabled=training_state.active or not mlx_source_path,
                    help="Конвертирует исходный файл в train.jsonl + valid.jsonl для mlx_lm",
                ):
                    with st.spinner("Подготовка данных..."):
                        try:
                            prep_config = DataPrepConfig(
                                source_path=mlx_source_path,
                                output_dir=mlx_data_dir,
                                format=mlx_format,
                                system_prompt=mlx_system_prompt,
                                max_seq_length=mlx_max_seq_length,
                                clean_forum=mlx_clean_forum_data,
                            )
                            stats = prepare_mlx_dataset(prep_config)
                            st.success(
                                f"✅ Готово! Train: {stats['train_count']} примеров, "
                                f"Valid: {stats['valid_count']} примеров"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка подготовки данных: {e}")

                # --- LoRA параметры ---
                st.divider()
                with st.expander("⚙️ LoRA параметры", expanded=False):
                    lora_col1, lora_col2 = st.columns(2)
                    with lora_col1:
                        mlx_lora_layers = st.number_input(
                            "LoRA Layers",
                            min_value=1,
                            max_value=32,
                            value=8,
                            disabled=training_state.active,
                            help=(
                                "Сколько последних слоёв трансформера адаптировать через LoRA.\n\n"
                                "Больше слоёв = больше обучаемых параметров, но медленнее.\n\n"
                                "Рекомендации: 4–8 для маленьких моделей, 8–16 для больших."
                            ),
                        )
                        mlx_lora_rank = st.number_input(
                            "LoRA Rank",
                            min_value=1,
                            max_value=64,
                            value=8,
                            disabled=training_state.active,
                            help=(
                                "Ранг матриц LoRA. Определяет ёмкость адаптера.\n\n"
                                "**Низкий rank (4–8)** → меньше параметров, быстрее, риск underfitting.\n"
                                "**Высокий rank (16–32)** → больше параметров, медленнее, лучше качество.\n\n"
                                "Рекомендации: 8 — хороший баланс для большинства задач."
                            ),
                        )
                        mlx_lora_scale = st.number_input(
                            "LoRA Scale",
                            min_value=1.0,
                            max_value=100.0,
                            value=20.0,
                            step=5.0,
                            disabled=training_state.active,
                            help=(
                                "Масштаб LoRA = alpha / rank. Управляет силой обновлений адаптера.\n\n"
                                "Обычно устанавливается равным rank или вдвое больше.\n\n"
                                "Рекомендации: 10–20 для стандартного fine-tuning."
                            ),
                        )
                        mlx_lora_dropout = st.number_input(
                            "LoRA Dropout",
                            min_value=0.0,
                            max_value=0.5,
                            value=0.0,
                            step=0.05,
                            format="%.2f",
                            disabled=training_state.active,
                            help=(
                                "Dropout внутри LoRA слоёв для регуляризации.\n\n"
                                "0.0 — без dropout (стандарт для маленьких датасетов).\n"
                                "0.05–0.1 — лёгкая регуляризация при переобучении."
                            ),
                        )
                    with lora_col2:
                        mlx_lr = st.number_input(
                            "Learning rate",
                            min_value=1e-6,
                            max_value=1e-3,
                            value=2e-5,
                            format="%.1e",
                            disabled=training_state.active,
                            help=(
                                "Learning rate для LoRA fine-tuning.\n\n"
                                "Значительно меньше, чем при обучении с нуля.\n\n"
                                "Рекомендации:\n"
                                "• 1e-5 — консервативный\n"
                                "• 2e-5 — стандарт (default)\n"
                                "• 5e-5 — более агрессивный"
                            ),
                        )
                        mlx_batch_size_val = st.number_input(
                            "Batch size",
                            min_value=1,
                            max_value=16,
                            value=4,
                            disabled=training_state.active,
                            help=(
                                "Размер батча для LoRA обучения.\n\n"
                                "Рекомендации:\n"
                                "• 2–4 — для моделей 1–2B при 8–16 GB памяти\n"
                                "• 1 — если не хватает памяти (включите grad checkpoint)\n\n"
                                "💡 Используйте grad checkpoint для экономии памяти при малом батче."
                            ),
                        )
                        mlx_iters = st.number_input(
                            "Iterations",
                            min_value=50,
                            max_value=5000,
                            value=500,
                            step=50,
                            disabled=training_state.active,
                            help=(
                                "Количество шагов обучения (не эпох).\n\n"
                                "В отличие от PyTorch-обучения, mlx-lm считает шаги, а не эпохи.\n\n"
                                "Рекомендации:\n"
                                "• 200–300 — быстрый тест\n"
                                "• 500 — стандарт (default)\n"
                                "• 1000+ — полноценное обучение\n\n"
                                "💡 Один шаг = один батч. "
                                "Общий объём данных = iters × batch_size × max_seq_length."
                            ),
                        )
                        mlx_steps_per_eval = st.number_input(
                            "Steps per eval",
                            min_value=10,
                            max_value=500,
                            value=100,
                            step=10,
                            disabled=training_state.active,
                            help="Как часто оценивать val loss (в шагах).",
                        )
                        mlx_save_every = st.number_input(
                            "Save every",
                            min_value=10,
                            max_value=500,
                            value=100,
                            step=10,
                            disabled=training_state.active,
                            help="Как часто сохранять промежуточный адаптер (в шагах).",
                        )
                        mlx_grad_checkpoint = st.checkbox(
                            "Gradient checkpointing",
                            value=False,
                            disabled=training_state.active,
                            help=(
                                "Экономит память за счёт пересчёта активаций при backprop.\n\n"
                                "Включайте, если не хватает памяти для выбранного batch size. "
                                "Обучение станет немного медленнее (~20–30%)."
                            ),
                        )

                # --- Название адаптера ---
                st.divider()
                st.session_state.setdefault("mlx_adapter_name", "my_style")
                mlx_adapter_name = st.text_input(
                    "📝 Название адаптера",
                    disabled=training_state.active,
                    key="mlx_adapter_name",
                    help=(
                        "Адаптер будет сохранён в adapters/{название}/\n\n"
                        "После обучения можно слить с базовой моделью (Fuse) "
                        "для получения автономной модели."
                    ),
                )
                mlx_adapter_path = str(Path(__file__).parent / "adapters" / mlx_adapter_name) if mlx_adapter_name else None

                # --- Продолжить с существующего адаптера ---
                _adapters_dir = Path(__file__).parent / "adapters"
                _existing_adapters = [
                    d for d in sorted(_adapters_dir.iterdir())
                    if d.is_dir() and any(d.glob("*.safetensors"))
                ] if _adapters_dir.exists() else []

                mlx_resume_adapter = None
                if _existing_adapters:
                    mlx_resume = st.checkbox(
                        "Продолжить обучение с существующего адаптера",
                        value=False,
                        disabled=training_state.active,
                        key="mlx_resume_check",
                        help=(
                            "Загружает веса из уже обученного адаптера и продолжает обучение.\n\n"
                            "Используй если:\n"
                            "• Обучение прервалось и хочешь продолжить\n"
                            "• Хочешь дообучить адаптер на новых данных"
                        ),
                    )
                    if mlx_resume:
                        _adapter_options = [d.name for d in _existing_adapters]
                        _selected_resume = st.selectbox(
                            "Адаптер для продолжения",
                            _adapter_options,
                            disabled=training_state.active,
                            key="mlx_resume_adapter_select",
                        )
                        _resume_dir = _adapters_dir / _selected_resume
                        _safetensors = list(_resume_dir.glob("*.safetensors"))
                        if _safetensors:
                            mlx_resume_adapter = str(_safetensors[0])
                            st.caption(f"📂 Продолжение с: `{mlx_resume_adapter}`")

        # Выбор checkpoint для дообучения
        checkpoint_to_load = None
        if is_finetuning and not is_mlx:
            st.divider()
            st.caption("**📥 Загрузка существующей модели:**")
            
            checkpoints = load_available_checkpoints()
            if checkpoints:
                checkpoint_to_load = st.selectbox(
                    "Выберите checkpoint",
                    checkpoints,
                    disabled=training_state.active,
                    help="Модель, которую будем дообучать"
                )
                
                # Показываем информацию о checkpoint
                if checkpoint_to_load:
                    checkpoint_path = str(checkpoint_dir / checkpoint_to_load)
                    try:
                        import torch
                        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"📊 Step: {cp.get('global_step', 'N/A')}")
                        with col2:
                            val_loss = cp.get('best_val_loss', float('inf'))
                            if val_loss != float('inf'):
                                st.caption(f"📉 Val loss: {val_loss:.4f}")
                    except:
                        pass
            else:
                st.warning("⚠️ Нет доступных checkpoint файлов для дообучения")
                st.info("Сначала обучите базовую модель в режиме 'С нуля'")
            
            st.divider()
        
        # ═══════════════════════════════════════════════════════
        # PyTorch-specific UI (hidden in MLX mode)
        # ═══════════════════════════════════════════════════════
        # Placeholder values so the run_training closure compiles in MLX mode
        if is_mlx:
            config_size = None
            data_path = ""
            epochs = 0
            batch_size = 4
            context_len = 256
            lr = 2e-5
            patience = 15
            min_epochs = 20
            dropout = 0.1
            weight_decay = 0.01
            grad_clip = 1.0
            warmup_steps = 200
            gradient_accumulation_steps = 1
            min_delta = 0.01
            checkpoint_interval = 500
            stride = None
            clean_forum = False
            tokenizer_type = "bpe"
            tokenizer_encoding = None
            bpe_vocab_size = 8000
            bpe_min_frequency = 2
            model_name = st.session_state.get("mlx_adapter_name", "my_adapter")

        # Размер модели (только для обучения с нуля)
        if not is_finetuning and not is_mlx:
            config_size = st.selectbox(
                "Размер модели",
                ["small", "medium", "base"],
                index=0,
                help=(
                "Архитектурный размер модели — определяет количество параметров, "
                "слоёв и размерность эмбеддингов.\n\n"
                "**small** (~13M параметров)\n"
                "d_model=256, layers=4, heads=4, context=256\n"
                "Быстрое обучение, мало памяти. Подходит для экспериментов "
                "и небольших датасетов (<1 MB).\n\n"
                "**medium** (~50M параметров)\n"
                "d_model=512, layers=6, heads=8, context=512\n"
                "Хороший баланс качества и скорости. Оптимален для корпусов 1–10 MB.\n\n"
                "**base** (~117M параметров, как GPT-2 small)\n"
                "d_model=768, layers=12, heads=12, context=1024\n"
                "Максимальное качество, но требует много памяти и времени. "
                "Рекомендуется от 10 MB данных.\n\n"
                "⚠️ Чем больше модель, тем больше данных нужно для хорошего обучения. "
                "На маленьком датасете большая модель переобучится быстрее."
            ),
                disabled=training_state.active
            )
        elif is_finetuning and not is_mlx:
            # При дообучении размер определяется checkpoint
            config_size = None
            st.info("💡 Архитектура модели будет загружена из checkpoint")

        data_dir = Path(__file__).parent / "data"
        file_options, file_paths, default_file_idx = get_training_file_options(data_dir)

        default_selected_option = file_options[default_file_idx] if file_options else None
        selected_option_for_defaults = st.session_state.get("training_data_select", default_selected_option)
        manual_input_option = "✏️ Ввести путь вручную..."

        if selected_option_for_defaults == manual_input_option:
            selected_data_path_for_defaults = st.session_state.get("training_manual_data_path", "data/")
        elif selected_option_for_defaults in file_paths:
            selected_data_path_for_defaults = file_paths[selected_option_for_defaults]
        else:
            selected_data_path_for_defaults = "data/sample.txt"

        tokenizer_type_for_defaults = "char" if is_finetuning else st.session_state.get("training_tokenizer_type", "hybrid")
        default_advice = build_training_data_advice(
            selected_data_path_for_defaults,
            tokenizer_type_for_defaults,
            256,
            32,
        )

        previous_auto_context = st.session_state.get("training_auto_context_len")
        previous_auto_batch = st.session_state.get("training_auto_batch_size")
        previous_auto_source = st.session_state.get("training_auto_source_path")
        current_context = st.session_state.get("train_context_len")
        current_batch = st.session_state.get("train_batch_size")

        can_auto_apply_defaults = (
            current_context is None
            or current_batch is None
            or (
                previous_auto_source != selected_data_path_for_defaults
                and current_context == previous_auto_context
                and current_batch == previous_auto_batch
            )
        )

        if default_advice is not None and can_auto_apply_defaults:
            st.session_state.train_context_len = int(default_advice['recommended_context'])
            st.session_state.train_batch_size = int(default_advice['recommended_batch'])
            st.session_state.training_auto_context_len = int(default_advice['recommended_context'])
            st.session_state.training_auto_batch_size = int(default_advice['recommended_batch'])
            st.session_state.training_auto_source_path = selected_data_path_for_defaults
        
        # Параметры обучения
        st.divider()
        epochs = st.number_input(
            "Количество эпох" if not is_finetuning else "Дополнительные эпохи", 
            min_value=1, 
            max_value=100, 
            value=10 if is_finetuning else 30,
            disabled=training_state.active,
            help=(
                "Сколько раз модель пройдёт через весь датасет целиком.\n\n"
                "Одна эпоха = один полный проход по всем обучающим батчам. "
                "После каждой эпохи веса модели обновляются, и она «видела» все данные ещё раз.\n\n"
                "**Мало эпох** → недообучение: модель не успела выучить паттерны.\n"
                "**Много эпох** → риск переобучения: модель запоминает данные наизусть, "
                "val loss растёт при хорошем train loss.\n\n"
                "Рекомендации:\n"
                "• 10–20 — быстрый эксперимент\n"
                "• 30–50 — стандартное обучение с early stopping (default 30)\n"
                "• 50–100 — маленький датасет, когда нужно много проходов\n\n"
                "💡 С включённым early stopping обучение остановится автоматически "
                "при отсутствии прогресса — можно смело ставить большое значение."
            ) if not is_finetuning else (
                "Сколько дополнительных эпох пройти поверх уже обученной модели.\n\n"
                "При дообучении веса уже хорошо инициализированы, поэтому "
                "нужно гораздо меньше эпох, чем при обучении с нуля.\n\n"
                "Рекомендации:\n"
                "• 3–5 — лёгкая адаптация к новым данным\n"
                "• 5–10 — стандартное дообучение (default)\n"
                "• 10+ — значительное изменение домена или большой новый датасет\n\n"
                "⚠️ Слишком много эпох при дообучении может привести к «забыванию» "
                "исходных знаний модели (catastrophic forgetting)."
            )
        )
        
        st.session_state.setdefault("train_batch_size", 32)
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=128,
            disabled=training_state.active,
            key="train_batch_size",
            help=(
                "Количество обучающих примеров, обрабатываемых за один шаг.\n\n"
                "**Больше** → стабильнее градиенты, быстрее эпоха, но больше памяти.\n"
                "**Меньше** → больший «шум» в градиентах (иногда помогает избежать локальных минимумов), меньше памяти.\n\n"
                "Рекомендации:\n"
                "• 8–16 — маленький датасет или мало памяти\n"
                "• 32 — хороший баланс (default)\n"
                "• 64–128 — если памяти достаточно и датасет большой\n\n"
                "⚠️ Если не хватает памяти GPU/MPS — уменьшите batch size или увеличьте Gradient Accumulation Steps."
            )
        )

        st.session_state.setdefault("train_context_len", 256)
        context_len = st.number_input(
            "Context Length",
            min_value=16,
            max_value=1024,
            step=16,
            help=(
                "Максимальная длина последовательности токенов, которую модель видит за один раз.\n\n"
                "**Больше** → модель улавливает более длинные зависимости в тексте, но квадратично растёт память (attention).\n"
                "**Меньше** → быстрее обучение, меньше памяти, но модель «не видит» дальний контекст.\n\n"
                "Рекомендации:\n"
                "• 64–128 — очень маленький датасет (<100 KB)\n"
                "• 256 — хороший баланс для корпусов 0.5–5 MB (default)\n"
                "• 512–1024 — большой датасет, достаточно памяти\n\n"
                "⚠️ Если датасет мал, слишком большой context_len даёт мало обучающих окон — модель быстро переобучается."
            ),
            disabled=training_state.active,
            key="train_context_len"
        )
        
        lr = st.number_input(
            "Learning rate",
            min_value=1e-5,
            max_value=1e-2,
            value=1.5e-4,
            format="%.5f",
            help=(
                "Скорость обновления весов модели на каждом шаге оптимизатора.\n\n"
                "**Больше** → быстрее обучение, но риск нестабильности (loss «прыгает» или расходится).\n"
                "**Меньше** → стабильнее, но обучение медленнее и можно застрять в локальном минимуме.\n\n"
                "Рекомендации:\n"
                "• 3e-4 — агрессивный старт, хорошо с warmup\n"
                "• 1e-4 .. 1.5e-4 — надёжный диапазон для трансформеров (default)\n"
                "• 1e-5 .. 5e-5 — дообучение (fine-tuning) поверх существующей модели\n\n"
                "💡 Используйте Warmup Steps для плавного разгона от 0 до целевого LR — "
                "это защищает от расходимости в начале обучения."
            ),
            disabled=training_state.active
        )
        
        patience = st.number_input(
            "Early stopping patience",
            min_value=1,
            max_value=50,
            value=15,
            help=(
                "Сколько проверок val_loss подряд без улучшения допустимо перед остановкой обучения.\n\n"
                "Проверка происходит каждые 150 шагов. При patience=15 обучение остановится, "
                "если val_loss не улучшился на протяжении 15 × 150 = 2250 шагов.\n\n"
                "**Маленькое значение** → быстрая остановка, риск недообучить модель.\n"
                "**Большое значение** → модель дольше ищет оптимум, но можно потратить время впустую.\n\n"
                "Рекомендации:\n"
                "• 5–10 — если датасет большой и эпохи длинные\n"
                "• 15 — хороший баланс (default)\n"
                "• 20–30 — если датасет маленький и loss медленно сходится\n\n"
                "💡 Используйте вместе с «Минимальное количество эпох», чтобы модель "
                "гарантированно прошла начальный этап обучения перед возможной остановкой."
            ),
            disabled=training_state.active
        )

        min_epochs = st.number_input(
            "Минимальное количество эпох",
            min_value=0,
            max_value=epochs if epochs else 10,
            value=20,
            help=(
                "Гарантированное число эпох до того, как early stopping сможет остановить обучение.\n\n"
                "В начале обучения val_loss часто нестабилен: он может временно расти, "
                "хотя модель ещё не сошлась. Без этого ограничения early stopping может "
                "сработать слишком рано и прервать обучение до достижения хорошего качества.\n\n"
                "Рекомендации:\n"
                "• 0 — early stopping может сработать с первых шагов (не рекомендуется)\n"
                "• 10 — минимальный порог для коротких экспериментов\n"
                "• 20 — хороший баланс (default)\n"
                "• 30+ — для маленьких датасетов, где модели нужно больше времени на разгон\n\n"
                "⚠️ Значение не должно превышать общее количество эпох."
            ),
            disabled=training_state.active
        )
        
        dropout = st.slider(
            "Dropout rate",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help=(
                "Вероятность случайного «выключения» нейрона во время каждого шага обучения.\n\n"
                "Это регуляризация: модель не может полагаться на конкретные нейроны и вынуждена "
                "учить более общие признаки. Во время инференса dropout автоматически отключается.\n\n"
                "**Симптом overfitting:** train loss падает, а val loss растёт — увеличьте dropout.\n\n"
                "Рекомендации:\n"
                "• 0.0 — без регуляризации (риск overfitting на малых датасетах)\n"
                "• 0.05 — лёгкая регуляризация для корпусов 1–10 MB\n"
                "• 0.1 — стандарт для трансформеров (default)\n"
                "• 0.2–0.3 — сильная регуляризация, если явный overfitting\n\n"
                "⚠️ При дообучении (fine-tuning) dropout берётся из загруженного checkpoint."
            ),
            disabled=training_state.active or is_finetuning
        )
        
        if is_finetuning:
            st.caption("⚠️ Dropout определяется загруженным checkpoint")
        
        # Расширенные параметры обучения
        with st.expander("⚙️ Расширенные параметры", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                weight_decay = st.number_input(
                    "Weight Decay",
                    min_value=0.0,
                    max_value=0.3,
                    value=0.01,
                    step=0.01,
                    format="%.3f",
                    help=(
                        "L2-регуляризация: штраф за слишком большие веса модели. "
                        "На каждом шаге веса немного «тянутся» к нулю, что мешает модели "
                        "запомнить обучающие данные наизусть.\n\n"
                        "В отличие от Dropout (случайное выключение нейронов), Weight Decay "
                        "действует постоянно и равномерно на все веса.\n\n"
                        "**Симптом overfitting:** большой разрыв между train loss и val loss — "
                        "попробуйте увеличить weight decay.\n\n"
                        "Рекомендации:\n"
                        "• 0.0 — без регуляризации\n"
                        "• 0.01 — стандарт для трансформеров (default)\n"
                        "• 0.05–0.1 — если явный overfitting\n\n"
                        "💡 Weight Decay и Dropout дополняют друг друга — можно использовать оба."
                    ),
                    disabled=training_state.active
                )
                
                grad_clip = st.number_input(
                    "Gradient Clipping",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f",
                    help=(
                        "Максимально допустимая норма вектора градиентов перед обновлением весов.\n\n"
                        "Если норма градиентов превышает это значение, они масштабируются вниз "
                        "пропорционально. Это защищает от «взрывного» роста градиентов (exploding "
                        "gradients), когда один неудачный батч может резко испортить все веса модели.\n\n"
                        "**Симптом проблемы:** loss внезапно скачет вверх или уходит в NaN — "
                        "уменьшите grad clip или learning rate.\n\n"
                        "Рекомендации:\n"
                        "• 1.0 — стандарт для трансформеров, подходит в большинстве случаев (default)\n"
                        "• 0.5 — более агрессивное ограничение, если обучение нестабильно\n"
                        "• 2.0–5.0 — мягкое ограничение, если градиенты в норме\n\n"
                        "💡 Gradient Clipping не замедляет обучение — он срабатывает только в "
                        "аномальных ситуациях."
                    ),
                    disabled=training_state.active
                )
                
                warmup_steps = st.number_input(
                    "Warmup Steps",
                    min_value=0,
                    max_value=2000,
                    value=200,
                    step=50,
                    help=(
                        "Число шагов, в течение которых learning rate плавно растёт от 0 до целевого значения.\n\n"
                        "В самом начале обучения веса случайны, градиенты нестабильны — "
                        "большой LR может сразу «разогнать» модель в неудачном направлении. "
                        "Warmup даёт модели время «освоиться» перед полноценным обучением.\n\n"
                        "После warmup LR плавно снижается по косинусному расписанию до конца обучения.\n\n"
                        "Рекомендации:\n"
                        "• 0 — без warmup (риск нестабильности в начале)\n"
                        "• 100–200 — маленький датасет или короткое обучение (default)\n"
                        "• 300–500 — большой датасет, длинное обучение\n\n"
                        "💡 Хорошее правило: warmup ≈ 1–5% от общего числа шагов обучения."
                    ),
                    disabled=training_state.active
                )
            
            with col2:
                gradient_accumulation_steps = st.number_input(
                    "Gradient Accumulation Steps",
                    min_value=1,
                    max_value=16,
                    value=1,
                    step=1,
                    help=(
                        "Количество шагов, на протяжении которых градиенты накапливаются перед "
                        "обновлением весов модели.\n\n"
                        "Эффективный batch size = batch_size × gradient_accumulation_steps. "
                        "То есть при batch_size=16 и accumulation=4 модель обновляется так, "
                        "как если бы batch был 64 — но без дополнительных затрат памяти.\n\n"
                        "**Когда использовать:**\n"
                        "• Не хватает памяти GPU/MPS для большого batch size\n"
                        "• Хочется стабильности больших батчей при ограниченном железе\n\n"
                        "Рекомендации:\n"
                        "• 1 — без накопления (default, если памяти достаточно)\n"
                        "• 2–4 — умеренное увеличение эффективного batch size\n"
                        "• 8–16 — если batch_size вынужденно маленький (4–8)\n\n"
                        "⚠️ Скорость обучения снижается пропорционально: при accumulation=4 "
                        "один «настоящий» шаг занимает в 4 раза больше времени."
                    ),
                    disabled=training_state.active
                )
                
                min_delta = st.number_input(
                    "Early Stopping Min Delta",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.01,
                    step=0.005,
                    format="%.3f",
                    help=(
                        "Минимальное улучшение val_loss, которое считается «настоящим» прогрессом "
                        "и сбрасывает счётчик patience.\n\n"
                        "Без этого порога даже улучшение на 0.0001 сбрасывало бы счётчик, "
                        "и обучение продолжалось бы бесконечно при очень медленном сходжении.\n\n"
                        "Пример: при min_delta=0.01 и patience=15 обучение остановится, если "
                        "val_loss не снизился хотя бы на 0.01 за последние 15 проверок.\n\n"
                        "Рекомендации:\n"
                        "• 0.0 — любое улучшение считается прогрессом\n"
                        "• 0.005–0.01 — стандартный порог (default)\n"
                        "• 0.02–0.05 — если хотите останавливаться только при значимом прогрессе\n\n"
                        "⚠️ Слишком большое значение может привести к преждевременной остановке, "
                        "если модель сходится медленно, но стабильно."
                    ),
                    disabled=training_state.active
                )
                
                checkpoint_interval = st.number_input(
                    "Checkpoint Interval (steps)",
                    min_value=100,
                    max_value=2000,
                    value=500,
                    step=100,
                    help=(
                        "Как часто сохранять промежуточный checkpoint на диск (в шагах).\n\n"
                        "Помимо этих периодических сохранений, лучший checkpoint по val_loss "
                        "всегда сохраняется автоматически как `{название}_best.pt`.\n\n"
                        "**Зачем нужны промежуточные checkpoint'ы:**\n"
                        "• Защита от прерывания обучения (сбой, выключение)\n"
                        "• Возможность вернуться к более ранней версии модели\n"
                        "• Продолжение обучения с любой точки через --continue-from\n\n"
                        "Рекомендации:\n"
                        "• 200–300 — короткие эксперименты или нестабильное обучение\n"
                        "• 500 — хороший баланс (default)\n"
                        "• 1000+ — длинное обучение, когда дисковое место ограничено\n\n"
                        "⚠️ Частое сохранение замедляет обучение и занимает место на диске."
                    ),
                    disabled=training_state.active
                )

                stride = st.number_input(
                    "Dataset Stride",
                    min_value=0,
                    max_value=2048,
                    value=0,
                    step=16,
                    help=(
                        "Шаг (в токенах) между соседними обучающими окнами при нарезке датасета.\n\n"
                        "Датасет нарезается на окна длиной context_len. Stride определяет, "
                        "насколько каждое следующее окно сдвигается относительно предыдущего.\n\n"
                        "• **0 (авто)** — окна не перекрываются (stride = context_len). "
                        "Максимальная скорость, минимум окон.\n"
                        "• **context_len // 2** — окна перекрываются наполовину, вдвое больше "
                        "обучающих примеров из того же текста.\n"
                        "• **Меньше stride** → больше окон, но соседние окна сильно похожи друг на друга.\n\n"
                        "Рекомендации:\n"
                        "• 0 — большой датасет, окон и так достаточно\n"
                        "• context_len // 2 — маленький датасет (<2 MB), нужно больше обучающих примеров\n\n"
                        "Пример: context_len=256, stride=128 → ~2× больше окон, каждое соседнее "
                        "окно делит половину токенов с предыдущим."
                    ),
                    disabled=training_state.active
                )
                stride = stride if stride > 0 else None

            clean_forum = st.checkbox(
                "🧹 Очистить форумный шум",
                value=False,
                help=(
                    "Удаляет форумный «шум» из текста перед обучением: имена авторов, "
                    "никнеймы вида `username#:` и строки вида `Имя Фамилия:`.\n\n"
                    "Форумные данные часто содержат паттерны вида:\n"
                    "```\nИванов123: Привет всем!\nPetrov#42: А я думаю...\n```\n"
                    "Модель начинает запоминать эти имена как часть языка, что ухудшает "
                    "качество генерации на нефорумных текстах.\n\n"
                    "**Что сохраняется** (семантически значимые префиксы):\n"
                    "• Вопрос:, Ответ:, Задача:, Решение:, Тема:, Пример:, Примечание:\n\n"
                    "**Что удаляется:**\n"
                    "• Произвольные имена и никнеймы перед двоеточием\n"
                    "• Строки вида `username#:` (форумный формат)\n\n"
                    "💡 Включайте, если датасет содержит форумные обсуждения или чаты."
                ),
                disabled=training_state.active
            )

        # Токенизатор
        tokenizer_type = st.selectbox(
            "Тип токенизатора",
            options=["char", "hybrid", "bpe", "tiktoken"],
            index=1 if not is_finetuning else 0,  # hybrid по умолчанию для новых моделей
            format_func=lambda x: {
                "char": "Character-level (legacy, 1 символ = 1 токен)",
                "hybrid": "Hybrid chemistry (доменные токены + fallback char)",
                "bpe": "BPE trainable (рекомендуемо 4k-16k словарь)",
                "tiktoken": "TikToken BPE (быстро, эффективно, GPT-4 tokenizer)"
            }[x],
            help=(
                "Определяет, как текст разбивается на токены — единицы, с которыми работает модель.\n\n"
                "**char** — каждый символ = один токен. Маленький словарь (~200–300 токенов), "
                "но длинные последовательности. Модель учится с нуля на уровне букв. "
                "Подходит для экспериментов, устарел для практики.\n\n"
                "**hybrid** — доменные токены (химические формулы, термины) + char-fallback. "
                "Оптимален для научных и специализированных текстов. Словарь ~8k–12k токенов.\n\n"
                "**bpe** — обучаемый BPE (Byte Pair Encoding) на вашем корпусе. "
                "Строит словарь из наиболее частых подслов. Гибкий, но требует времени на обучение. "
                "Словарь 4k–16k токенов.\n\n"
                "**tiktoken** — готовый BPE от OpenAI (GPT-4). Не обучается на вашем корпусе, "
                "зато очень быстрый и эффективен для русского языка. Словарь ~100k токенов.\n\n"
                "Рекомендации:\n"
                "• Научный/химический текст → **hybrid**\n"
                "• Русский общий текст → **tiktoken** или **bpe**\n"
                "• Эксперименты/отладка → **char**\n\n"
                "⚠️ При дообучении токенизатор берётся из checkpoint и не меняется."
            ),
            disabled=training_state.active or is_finetuning,
            key="training_tokenizer_type"
        )
        
        if tokenizer_type == "tiktoken":
            tokenizer_encoding = st.selectbox(
                "TikToken encoding",
                options=["cl100k_base", "o200k_base", "p50k_base"],
                index=0,
                format_func=lambda x: {
                    "cl100k_base": "cl100k_base (GPT-4, ~100K tokens, русский+химия)",
                    "o200k_base": "o200k_base (GPT-4o, ~200K tokens)",
                    "p50k_base": "p50k_base (Codex, ~50K tokens, для кода)"
                }[x],
                help="cl100k_base оптимален для русского языка и научных текстов",
                disabled=training_state.active or is_finetuning
            )
        else:
            tokenizer_encoding = None

        if tokenizer_type == "bpe":
            bpe_vocab_size = st.select_slider(
                "BPE vocab size",
                options=[4000, 6000, 8000, 12000, 16000],
                value=8000,
                help=(
                    "Целевое количество токенов в словаре BPE-токенизатора.\n\n"
                    "BPE (Byte Pair Encoding) итеративно объединяет наиболее частые пары символов "
                    "в один токен. Размер словаря определяет, насколько крупными будут токены.\n\n"
                    "**Маленький словарь (4k)** → токены короче, последовательности длиннее, "
                    "модель видит меньше контекста за раз. Хуже для редких слов.\n\n"
                    "**Большой словарь (12k–16k)** → токены крупнее, последовательности короче, "
                    "но требует больше данных для обучения токенизатора.\n\n"
                    "Рекомендации:\n"
                    "• 4000 — очень маленький датасет (<500 KB)\n"
                    "• 6000–8000 — датасет 0.5–5 MB (default)\n"
                    "• 12000–16000 — датасет >5 MB, богатый словарный состав\n\n"
                    "💡 Хорошее правило: vocab_size ≈ √(количество уникальных слов в корпусе). "
                    "Слишком большой словарь при малом корпусе приведёт к токенам-одиночкам."
                ),
                disabled=training_state.active or is_finetuning,
            )
            bpe_min_frequency = st.number_input(
                "BPE min frequency",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
                help=(
                    "Минимальное количество раз, которое пара символов должна встретиться "
                    "в корпусе, чтобы объединиться в один токен.\n\n"
                    "BPE строит словарь, объединяя наиболее частые пары. Этот порог "
                    "отфильтровывает редкие пары, которые встречаются случайно и не несут "
                    "смысловой нагрузки.\n\n"
                    "**Низкое значение (1–2)** → больше редких токенов в словаре, "
                    "токенизатор лучше покрывает редкие слова, но словарь «засоряется».\n\n"
                    "**Высокое значение (5–10)** → только частые, устойчивые токены. "
                    "Чище, но редкие слова разбиваются на множество мелких кусков.\n\n"
                    "Рекомендации:\n"
                    "• 2 — стандарт, подходит для большинства корпусов (default)\n"
                    "• 3–5 — большой корпус с богатым словарём\n"
                    "• 1 — очень маленький корпус, чтобы не терять редкие слова\n\n"
                    "⚠️ При дообучении параметр берётся из checkpoint и не меняется."
                ),
                disabled=training_state.active or is_finetuning,
            )
        else:
            bpe_vocab_size = 8000
            bpe_min_frequency = 2
        
        if is_finetuning:
            st.caption("⚠️ Токенизатор определяется загруженным checkpoint")
        
        # Данные
        st.markdown("**Данные для обучения**" if not is_finetuning else "**Новые данные для дообучения**")

        # Создаём список опций для selectbox
        if file_options:
            # Selectbox для выбора файла
            selected_option = st.selectbox(
                "Выберите файл данных",
                options=file_options,
                index=default_file_idx,
                disabled=training_state.active,
                help="Файлы из директории data/ или введите путь вручную",
                key="training_data_select"
            )
            
            # Определяем путь
            if selected_option == manual_input_option:
                data_path = st.text_input(
                    "Путь к файлу",
                    value="data/",
                    disabled=training_state.active,
                    help="Введите полный путь к файлу данных",
                    key="training_manual_data_path"
                )
            else:
                data_path = file_paths[selected_option]
                st.caption(f"📁 Путь: `{data_path}`")
        else:
            # Если нет файлов в data/ - показываем text_input
            st.info("📂 Директория data/ пуста или не найдена")
            data_path = st.text_input(
                "Путь к данным",
                value="data/sample.txt",
                disabled=training_state.active
            )
        
        # Проверка наличия данных
        if data_path and not Path(data_path).exists():
            st.warning(f"⚠️ Файл {data_path} не найден")
            if "sample.txt" in data_path and st.button("📝 Создать sample.txt"):
                prepare_sample_data(data_path)
                st.success(f"✅ Создан {data_path}")
                st.rerun()
        elif data_path and Path(data_path).exists():
            file_size = Path(data_path).stat().st_size / 1024
            file_size_mb = file_size / 1024
            if file_size_mb >= 1:
                st.success(f"✅ Файл готов к использованию ({file_size_mb:.1f} MB)")
            else:
                st.success(f"✅ Файл готов к использованию ({file_size:.1f} KB)")

            advice = build_training_data_advice(
                data_path,
                tokenizer_type if not is_finetuning else "char",
                context_len,
                batch_size,
            )

            if advice is not None:
                st.caption("📊 Быстрая оценка датасета")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Оценка токенов", f"{advice['estimated_tokens']:,}")
                with metric_col2:
                    st.metric("Train окон", f"{advice['train_windows']:,}")
                with metric_col3:
                    st.metric("Train batch'ей", f"{advice['train_batches']:,}")

                advice_message = (
                    f"{advice['message']} Stride≈{advice['train_stride']} "
                    f"(val≈{advice['val_stride']}), val окон≈{advice['val_windows']}."
                )
                if advice['severity'] == 'error':
                    st.error(advice_message)
                elif advice['severity'] == 'warning':
                    st.warning(advice_message)
                elif advice['severity'] == 'info':
                    st.info(advice_message)
                else:
                    st.success(advice_message)

                if (
                    advice['recommended_context'] != context_len
                    or advice['recommended_batch'] != batch_size
                ):
                    st.caption(
                        "Рекомендация для этого корпуса: "
                        f"context_len≈{advice['recommended_context']}, "
                        f"batch_size≈{advice['recommended_batch']}."
                    )

                    if st.button(
                        "✨ Применить рекомендации",
                        key="apply_training_recommendations",
                        disabled=training_state.active,
                        help="Подставить рекомендованные context_len и batch_size в форму"
                    ):
                        st.session_state.train_context_len = int(advice['recommended_context'])
                        st.session_state.train_batch_size = int(advice['recommended_batch'])
                        st.session_state.training_auto_context_len = int(advice['recommended_context'])
                        st.session_state.training_auto_batch_size = int(advice['recommended_batch'])
                        st.session_state.training_auto_source_path = data_path
                        st.rerun()
        
        # Название модели/эксперимента
        st.divider()
        default_model_name = "chemistry_model" if "chemistry" in str(data_path).lower() else "my_model"

        # В режиме дообучения подставляем базовое имя выбранного checkpoint без суффиксов
        # (_best, _epoch_N, _step_N), чтобы сохранять под тем же именем.
        if is_finetuning and checkpoint_to_load:
            checkpoint_stem = Path(checkpoint_to_load).stem
            default_model_name = re.sub(r'_(best|epoch_\d+|step_\d+)$', '', checkpoint_stem)

            prev_auto_name = st.session_state.get("training_auto_model_name")
            prev_checkpoint_name = st.session_state.get("training_model_name_source_checkpoint")
            current_model_name = st.session_state.get("training_model_name")

            should_apply_auto_name = (
                current_model_name is None
                or current_model_name == prev_auto_name
                or prev_checkpoint_name != checkpoint_to_load
            )

            if should_apply_auto_name:
                st.session_state.training_model_name = default_model_name
                st.session_state.training_auto_model_name = default_model_name
                st.session_state.training_model_name_source_checkpoint = checkpoint_to_load
        elif "training_model_name" not in st.session_state:
            st.session_state.training_model_name = default_model_name
            st.session_state.training_auto_model_name = default_model_name

        model_name = st.text_input(
            "📝 Название модели",
            disabled=training_state.active,
            key="training_model_name",
            help="Уникальное имя для этой модели. Лучший checkpoint будет сохранен как {название}_best.pt"
        )
        
        if not model_name or not model_name.strip():
            st.error("⚠️ Введите название модели!")
        
        # Device
        auto_device = get_device()
        device_options = ["auto", "mps", "cuda", "cpu"]
        default_idx = 0  # auto по умолчанию
        device = st.selectbox(
            "Device",
            device_options,
            index=default_idx,
            disabled=training_state.active,
            help=(
                "Вычислительное устройство для обучения модели.\n\n"
                "**auto** — автоматически выбирает лучшее доступное устройство "
                "(MPS → CUDA → CPU). Рекомендуется (default).\n\n"
                "**mps** — Apple Silicon (M1/M2/M3/M4). Использует unified memory, "
                "значительно быстрее CPU. Доступно только на Mac с Apple Silicon.\n\n"
                "**cuda** — NVIDIA GPU. Самый быстрый вариант при наличии. "
                "Требует CUDA-совместимую видеокарту и установленный CUDA toolkit.\n\n"
                "**cpu** — центральный процессор. Медленно, но работает везде. "
                "Подходит для отладки или если GPU недоступен.\n\n"
                "💡 Ориентировочная скорость: CUDA > MPS >> CPU. "
                "На CPU обучение может быть в 10–50× медленнее, чем на GPU."
            )
        )
        
        # Показываем конфиг модели
        if not is_finetuning:
            if config_size == "small":
                config = get_small_config()
            elif config_size == "medium":
                config = get_medium_config()
            else:
                config = get_base_config()
        else:
            # При дообучении показываем конфиг из checkpoint
            if checkpoint_to_load:
                try:
                    import torch
                    cp = torch.load(str(checkpoint_dir / checkpoint_to_load), map_location="cpu", weights_only=False)
                    config = cp['config']
                except:
                    config = None
            else:
                config = None
        
        st.divider()
        
        # Показываем конфиг модели
        if config:
            st.caption("**Архитектура модели:**")
            
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.caption(f"d_model: {config.d_model}")
                st.caption(f"n_layers: {config.n_layers}")
            with info_col2:
                st.caption(f"n_heads: {config.n_heads}")
                st.caption(f"context: {config.context_len}")
    
    with col_right:
        st.subheader("🚀 Запуск")
        
        # Кнопки управления
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            # Проверка готовности к запуску
            if is_mlx:
                mlx_data_prepped_check = bool(
                    mlx_data_dir and (Path(mlx_data_dir) / "train.jsonl").exists()
                )
                can_start = (
                    not training_state.active
                    and bool(mlx_model_id)
                    and bool(mlx_adapter_path)
                    and mlx_data_prepped_check
                )
                button_text = "▶️ Начать MLX Fine-tuning"
            else:
                can_start = not training_state.active and Path(data_path).exists() and model_name and model_name.strip()
                if is_finetuning:
                    can_start = can_start and checkpoint_to_load is not None
                button_text = "▶️ Начать дообучение" if is_finetuning else "▶️ Начать обучение"
            
            start_button = st.button(
                button_text,
                type="primary",
                disabled=not can_start,
                width='stretch'
            )
        
        with button_col2:
            stop_button = st.button(
                "⏹️ Остановить",
                disabled=not training_state.active,
                width='stretch'
            )
            
        # Обработка остановки
        if stop_button:
            training_state.stop_requested = True
            training_state.logs.append("🛑 Запрошена остановка обучения...")
            app_logger.info("UI Пользователь запросил остановку обучения")
        
        # Запуск обучения
        if start_button:
            training_state.active = True
            training_state.stop_requested = False  # Reset флага остановки
            training_state.logs = []
            training_state.metrics = {
                'epoch': 0,
                'step': 0,
                'train_loss': 0.0,
                'val_loss': 0.0,
                'best_val_loss': float('inf'),
                'progress': 0.0
            }
            
            if is_mlx:
                app_logger.info("UI Обучение запущено: mode=MLX Fine-tuning")
                app_logger.info(f"UI MLX Параметры: model={mlx_model_id}, iters={mlx_iters}, batch={mlx_batch_size_val}, lr={mlx_lr}, data={mlx_data_dir}")
            else:
                app_logger.info(f"UI Обучение запущено: mode={'fine-tuning' if is_finetuning else 'from scratch'}")
                if is_finetuning:
                    app_logger.info(f"UI Checkpoint для дообучения: {checkpoint_to_load}")
                app_logger.info(f"UI Параметры: epochs={epochs}, batch_size={batch_size}, lr={lr}, data={data_path}")
            
            # Запускаем обучение в отдельном потоке
            def run_training():
                if is_mlx:
                    # ═══════════════════════════════════════════════════════
                    # MLX LoRA Fine-tuning
                    # ═══════════════════════════════════════════════════════
                    from mlx_integration import MLXTrainingConfig, run_mlx_training
                    config = MLXTrainingConfig(
                        model_id=mlx_model_id,
                        adapter_path=mlx_adapter_path,
                        data_dir=mlx_data_dir,
                        lora_layers=mlx_lora_layers,
                        lora_rank=mlx_lora_rank,
                        lora_scale=mlx_lora_scale,
                        lora_dropout=mlx_lora_dropout,
                        learning_rate=mlx_lr,
                        batch_size=mlx_batch_size_val,
                        iters=mlx_iters,
                        max_seq_length=mlx_max_seq_length,
                        val_batches=25,
                        steps_per_eval=mlx_steps_per_eval,
                        save_every=mlx_save_every,
                        grad_checkpoint=mlx_grad_checkpoint,
                        resume_adapter_file=mlx_resume_adapter,
                    )
                    run_mlx_training(config, training_state)
                    return  # run_mlx_training manages its own thread + training_state

                try:
                    if is_finetuning and checkpoint_to_load:
                        # ═══════════════════════════════════════════════════════
                        # ДООБУЧЕНИЕ (Fine-tuning)
                        # ═══════════════════════════════════════════════════════
                        training_state.logs.append(f"🔄 Режим: Дообучение (Fine-tuning)")
                        training_state.logs.append(f"📥 Загрузка checkpoint: {checkpoint_to_load}")
                        
                        checkpoint_path = str(checkpoint_dir / checkpoint_to_load)
                        
                        # Загружаем checkpoint
                        import torch
                        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                        model_config = checkpoint['config']
                        
                        training_state.logs.append(f"✅ Checkpoint загружен (step {checkpoint.get('global_step', 'N/A')})")
                        
                        # Настройки обучения
                        train_config = TrainingConfig(
                            n_epochs=epochs,
                            data_path=data_path,
                            device=device if device != "auto" else get_device(),
                            eval_every=150,
                            save_every=checkpoint_interval,
                            log_every=50,
                            patience=patience,
                            min_epochs=min_epochs,
                            min_delta=min_delta,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            warmup_steps=warmup_steps
                        )
                        
                        # Проверка device и предупреждения
                        actual_device = train_config.device
                        device_emoji = "🍎" if actual_device == "mps" else ("🟢" if actual_device == "cuda" else "💻")
                        training_state.logs.append(f"{device_emoji} **Device: {actual_device.upper()}**")
                        
                        # Предупреждение о fallback на CPU
                        if device == "auto" and actual_device == "cpu":
                            training_state.logs.append("⚠️ **Внимание:** GPU не доступен, используется CPU (обучение будет медленным)")
                            if not torch.backends.mps.is_available() and not torch.cuda.is_available():
                                training_state.logs.append("💡 Для ускорения используйте Mac с Apple Silicon или GPU NVIDIA")
                        
                        # Быстрый бенчмарк device
                        training_state.logs.append("⚡ Бенчмарк производительности...")
                        bench = benchmark_device(actual_device, size=1024)
                        if bench['success']:
                            training_state.logs.append(f"   └─ Скорость: {bench['gflops']:.1f} GFLOPS ({bench['time_ms']:.2f} ms)")
                            
                            # Сравнение с CPU для наглядности (если не CPU)
                            if actual_device != "cpu":
                                cpu_bench = benchmark_device("cpu", size=1024)
                                if cpu_bench['success']:
                                    speedup = cpu_bench['time_ms'] / bench['time_ms']
                                    training_state.logs.append(f"   └─ Ускорение vs CPU: {speedup:.1f}x быстрее")
                        else:
                            training_state.logs.append(f"   └─ ⚠️ Ошибка бенчмарка: {bench['error']}")
                        
                        # Статистика памяти
                        mem_stats = get_memory_stats(actual_device)
                        if mem_stats and not mem_stats.get('error'):
                            training_state.logs.append(f"💾 Доступно памяти: {mem_stats['total_gb']:.1f} GB")

                        # Для fine-tuning используем тот же тип токенизатора, что в checkpoint.
                        tokenizer_config = checkpoint.get('tokenizer_config', {})
                        ft_tokenizer_type = 'char'
                        ft_tokenizer_encoding = 'cl100k_base'

                        if isinstance(tokenizer_config, dict):
                            tokenizer_kind = tokenizer_config.get('type')
                            if tokenizer_kind == 'tiktoken':
                                ft_tokenizer_type = 'tiktoken'
                                ft_tokenizer_encoding = tokenizer_config.get('encoding_name', 'cl100k_base')
                            elif tokenizer_kind == 'hybrid':
                                ft_tokenizer_type = 'hybrid'
                            elif tokenizer_kind == 'bpe':
                                ft_tokenizer_type = 'bpe'
                            elif tokenizer_kind == 'char':
                                ft_tokenizer_type = 'char'

                        training_state.logs.append(
                            f"🔤 Токенизатор fine-tuning: {ft_tokenizer_type}"
                            + (f" ({ft_tokenizer_encoding})" if ft_tokenizer_type == 'tiktoken' else "")
                        )
                        
                        # Загрузка данных
                        training_state.logs.append("📁 Загрузка новых данных...")
                        train_loader, val_loader, tokenizer = call_load_data_compat(
                            data_path=data_path,
                            context_len=model_config.context_len,
                            batch_size=batch_size,
                            tokenizer_type=ft_tokenizer_type,
                            tokenizer_encoding=ft_tokenizer_encoding,
                            bpe_vocab_size=int(tokenizer_config.get('vocab_size', 8000)) if isinstance(tokenizer_config, dict) else 8000,
                            bpe_min_frequency=int(tokenizer_config.get('min_frequency', 2)) if isinstance(tokenizer_config, dict) else 2,
                            tokenizer_config=tokenizer_config if isinstance(tokenizer_config, dict) else None,
                            normalize_chemistry=True,
                            stride=stride,
                            clean_forum=clean_forum,
                        )
                        
                        vocab_size = tokenizer.vocab_size() if hasattr(tokenizer, 'vocab_size') else len(tokenizer.vocab)
                        training_state.logs.append(f"✅ Данные загружены ({vocab_size} токенов)")
                        
                        # Создаём модель и загружаем веса
                        training_state.logs.append("🏗️ Восстановление модели...")
                        model = GPTModel(model_config).to(train_config.device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        
                        training_state.logs.append(f"✅ Модель восстановлена: {model.count_parameters():,} параметров")
                        
                        # Создаём trainer
                        trainer = Trainer(
                            model,
                            train_loader,
                            val_loader,
                            train_config,
                            device=train_config.device,
                            tokenizer=tokenizer,
                            model_name=model_name
                        )
                        
                        # Восстанавливаем optimizer state (опционально)
                        if 'optimizer_state_dict' in checkpoint:
                            try:
                                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                                training_state.logs.append("✅ Optimizer state восстановлен")
                            except:
                                training_state.logs.append("⚠️ Optimizer state не загружен (будет создан новый)")
                        
                        trainer.global_step = checkpoint.get('global_step', 0)
                        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                        
                        training_state.logs.append(f"🎓 Начало дообучения...")
                        
                    else:
                        # ═══════════════════════════════════════════════════════
                        # ОБУЧЕНИЕ С НУЛЯ
                        # ═══════════════════════════════════════════════════════
                        training_state.logs.append("🆕 Режим: Обучение с нуля")
                        
                        # Конфигурация
                        if config_size == "small":
                            model_config = get_small_config()
                        elif config_size == "medium":
                            model_config = get_medium_config()
                        else:
                            model_config = get_base_config()
                        
                        model_config.batch_size = batch_size
                        model_config.learning_rate = lr
                        model_config.context_len = context_len
                        model_config.dropout = dropout  # Применяем dropout из UI
                        model_config.weight_decay = weight_decay
                        model_config.grad_clip = grad_clip
                        
                        train_config = TrainingConfig(
                            n_epochs=epochs,
                            data_path=data_path,
                            device=device if device != "auto" else get_device(),
                            eval_every=150,
                            save_every=checkpoint_interval,
                            log_every=50,
                            patience=patience,
                            min_epochs=min_epochs,
                            min_delta=min_delta,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            warmup_steps=warmup_steps
                        )
                        
                        # Проверка device и предупреждения
                        actual_device = train_config.device
                        device_emoji = "🍎" if actual_device == "mps" else ("🟢" if actual_device == "cuda" else "💻")
                        training_state.logs.append(f"{device_emoji} **Device: {actual_device.upper()}**")
                        
                        # Предупреждение о fallback на CPU
                        if device == "auto" and actual_device == "cpu":
                            training_state.logs.append("⚠️ **Внимание:** GPU не доступен, используется CPU (обучение будет медленным)")
                            if not torch.backends.mps.is_available() and not torch.cuda.is_available():
                                training_state.logs.append("💡 Для ускорения используйте Mac с Apple Silicon или GPU NVIDIA")
                        
                        # Быстрый бенчмарк device
                        training_state.logs.append("⚡ Бенчмарк производительности...")
                        bench = benchmark_device(actual_device, size=1024)
                        if bench['success']:
                            training_state.logs.append(f"   └─ Скорость: {bench['gflops']:.1f} GFLOPS ({bench['time_ms']:.2f} ms)")
                            
                            # Сравнение с CPU для наглядности (если не CPU)
                            if actual_device != "cpu":
                                cpu_bench = benchmark_device("cpu", size=1024)
                                if cpu_bench['success']:
                                    speedup = cpu_bench['time_ms'] / bench['time_ms']
                                    training_state.logs.append(f"   └─ Ускорение vs CPU: {speedup:.1f}x быстрее")
                        else:
                            training_state.logs.append(f"   └─ ⚠️ Ошибка бенчмарка: {bench['error']}")
                        
                        # Статистика памяти
                        mem_stats = get_memory_stats(actual_device)
                        if mem_stats and not mem_stats.get('error'):
                            training_state.logs.append(f"💾 Доступно памяти: {mem_stats['total_gb']:.1f} GB")
                        
                        # Загрузка данных
                        training_state.logs.append("📁 Загрузка данных...")
                        train_loader, val_loader, tokenizer = call_load_data_compat(
                            data_path=data_path,
                            context_len=model_config.context_len,
                            batch_size=model_config.batch_size,
                            tokenizer_type=tokenizer_type,
                            tokenizer_encoding=tokenizer_encoding if tokenizer_type == 'tiktoken' else 'cl100k_base',
                            bpe_vocab_size=bpe_vocab_size,
                            bpe_min_frequency=bpe_min_frequency,
                            normalize_chemistry=True,
                            stride=stride,
                            clean_forum=clean_forum,
                        )
                        
                        model_config.vocab_size = tokenizer.vocab_size() if hasattr(tokenizer, 'vocab_size') else len(tokenizer.vocab)
                        
                        # Создание модели
                        training_state.logs.append(f"🏗️ Создание модели ({model_config.vocab_size} токенов)...")
                        model = GPTModel(model_config)
                        
                        training_state.logs.append(f"✅ Модель создана: {model.count_parameters():,} параметров")
                        
                        # Trainer
                        trainer = Trainer(
                            model,
                            train_loader,
                            val_loader,
                            train_config,
                            device=train_config.device,
                            tokenizer=tokenizer,
                            model_name=model_name
                        )
                        
                        training_state.logs.append(f"🎓 Начало обучения...")
                    
                    # ═══════════════════════════════════════════════════════
                    # ОБЩИЙ ЦИКЛ ОБУЧЕНИЯ (для обоих режимов)
                    # ═══════════════════════════════════════════════════════
                    
                    last_completed_epoch = 0
                    
                    for epoch in range(train_config.n_epochs):
                        if not training_state.active or training_state.stop_requested:
                            training_state.logs.append("⏹️ Обучение остановлено пользователем")
                            break
                        
                        train_loss, early_stop = trainer.train_epoch(epoch)
                        
                        # Проверка early stopping
                        if early_stop:
                            training_state.logs.append(f"⏹️ Early stopping на эпохе {epoch+1}")
                            app_logger.info(f"UI Early stopping на эпохе {epoch+1}")
                            break
                        
                        val_loss = trainer.evaluate()
                        
                        # Обновляем best_val_loss и метрики
                        if val_loss < trainer.best_val_loss:
                            trainer.best_val_loss = val_loss
                            trainer.steps_without_improvement = 0
                        
                        # Обновляем метрики
                        training_state.metrics['epoch'] = epoch + 1
                        training_state.metrics['step'] = trainer.global_step
                        training_state.metrics['train_loss'] = train_loss
                        training_state.metrics['val_loss'] = val_loss
                        training_state.metrics['best_val_loss'] = trainer.best_val_loss
                        training_state.metrics['progress'] = (epoch + 1) / train_config.n_epochs
                        
                        log_msg = f"Эпоха {epoch+1}/{train_config.n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                        if val_loss == trainer.best_val_loss:
                            log_msg += " ⭐ Новый рекорд!"
                        training_state.logs.append(log_msg)
                        
                        last_completed_epoch = epoch + 1
                    
                    # В конце обучения обновляем лучший checkpoint
                    if last_completed_epoch > 0:
                        # Обновляем лучший checkpoint с финальными весами
                        best_checkpoint = f"{model_name}_best.pt"
                        trainer.save_checkpoint(best_checkpoint)
                        
                        training_state.logs.append("✅ Обучение завершено!")
                        training_state.logs.append(f"📊 Завершено эпох: {last_completed_epoch}/{train_config.n_epochs}")
                        training_state.logs.append(f"⭐ Лучший checkpoint: {best_checkpoint}")
                        training_state.logs.append(f"📊 Best val_loss: {trainer.best_val_loss:.4f}")
                        training_state.logs.append(f"💾 Промежуточные checkpoint'ы сохранялись каждые {checkpoint_interval} шагов")
                        app_logger.info(f"UI Обучение завершено успешно ({last_completed_epoch} эпох)")
                    
                except Exception as e:
                    training_state.logs.append(f"❌ Ошибка: {str(e)}")
                    app_logger.error(f"UI Ошибка при обучении: {str(e)}")
                    import traceback
                    training_state.logs.append(traceback.format_exc())
                finally:
                    training_state.active = False
                    training_state.stop_requested = False
            
            # Запускаем в фоне
            thread = threading.Thread(target=run_training, daemon=True)
            thread.start()
            st.rerun()
        
        # Остановка
        if stop_button:
            training_state.active = False
            training_state.logs.append("⏸️ Остановка обучения...")
            st.rerun()
        
        st.divider()
        
        # Прогресс
        if training_state.active or training_state.logs:
            st.subheader("📊 Прогресс")
            
            if training_state.active:
                progress = training_state.metrics.get('progress', 0.0)
                st.progress(progress)

                if is_mlx:
                    # MLX progress: show step + losses
                    mlx_step = training_state.metrics.get('step', 0)
                    mlx_train_loss = training_state.metrics.get('train_loss', 0.0)
                    mlx_val_loss = training_state.metrics.get('val_loss', 0.0)
                    mlx_it_sec = training_state.metrics.get('it_per_sec', 0.0)

                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric("Шаг", f"{mlx_step} / {mlx_iters}")
                    with m_col2:
                        st.metric("Train Loss", f"{mlx_train_loss:.4f}" if mlx_train_loss else "—")
                        if mlx_val_loss:
                            st.metric("Val Loss", f"{mlx_val_loss:.4f}")
                    with m_col3:
                        if mlx_it_sec:
                            st.metric("Скорость", f"{mlx_it_sec:.1f} it/s")
                else:
                    # PyTorch progress
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Эпоха", f"{training_state.metrics['epoch']}")
                        st.metric("Шаг", f"{training_state.metrics['step']}")
                    with metric_col2:
                        st.metric("Train Loss", f"{training_state.metrics['train_loss']:.4f}")
                        current_val = training_state.metrics['val_loss']
                        st.metric("Val Loss", f"{current_val:.4f}")
                    with metric_col3:
                        best_val = training_state.metrics.get('best_val_loss', float('inf'))
                        if best_val < float('inf'):
                            st.metric("Best Val Loss", f"{best_val:.4f}")
                            # Индикатор - улучшается ли модель
                            if current_val <= best_val:
                                st.success("⭐ Новый рекорд!")
                            else:
                                delta = current_val - best_val
                                st.warning(f"📊 +{delta:.4f} от лучшего")
                        else:
                            st.metric("Best Val Loss", "—")

            # Fuse button for MLX after training completes
            if is_mlx and not training_state.active and training_state.metrics.get('finished'):
                st.success(f"✅ Обучение завершено! Адаптер: `{mlx_adapter_path}`")
                st.divider()
                st.caption("**🔀 Слить адаптер с базовой моделью (опционально)**")
                fuse_output = st.text_input(
                    "Путь для сохранения слитой модели",
                    value=f"models/{mlx_adapter_name}_fused" if mlx_adapter_path else "models/fused_model",
                    key="mlx_fuse_output_path",
                    help=(
                        "Директория для сохранения модели с уже встроенными LoRA весами.\n\n"
                        "После слияния адаптер не нужен — модель работает автономно. "
                        "Размер увеличится (~= размер базовой модели без квантизации)."
                    ),
                )
                if st.button(
                    "🔀 Слить адаптер с моделью (Fuse)",
                    help="Встраивает LoRA адаптер в базовую модель. Занимает ~1-3 минуты.",
                    key="mlx_fuse_button",
                ):
                    with st.spinner("Слияние адаптера..."):
                        from mlx_integration import fuse_adapter
                        success, msg = fuse_adapter(mlx_model_id, mlx_adapter_path, fuse_output)
                        if success:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")
            
            # Логи
            st.divider()
            st.caption("**Логи:**")
            
            logs_container = st.container(height=300)
            with logs_container:
                for log in training_state.logs[-20:]:  # Последние 20 строк
                    st.text(log)
            
            # Auto-refresh если обучение активно
            if training_state.active:
                time.sleep(1)
                st.rerun()
    
    # CLI команда (для справки)
    st.divider()
    st.subheader("💻 Альтернативно: через терминал")
    
    if is_finetuning and checkpoint_to_load:
        command = f"""python train.py \\
  --continue-from checkpoints/{checkpoint_to_load} \\
  --data {data_path} \\
  --epochs {epochs}"""
        
        with st.expander("Показать команду для дообучения"):
            st.code(command, language="bash")
            st.caption("Скопируйте и запустите в терминале для дообучения без UI")
            
            st.info("""
            💡 **Tip:** При дообучении:
            - Learning rate обычно меньше (1e-4 - 1e-5)
            - Меньше эпох (3-10 вместо 30-50)
            - Можно использовать новые данные для специализации модели
            """)
    else:
        command = f"""python train.py \\
  --config {config_size if config_size else 'small'} \\
  --epochs {epochs} \\
  --batch-size {batch_size} \\
  --lr {lr} \\
  --data {data_path} \\
  --device {device}"""
        
        with st.expander("Показать команду"):
            st.code(command, language="bash")
            st.caption("Скопируйте и запустите в терминале для обучения без UI")


def tab_data():
    """Вкладка подготовки данных."""
    data_dir = Path(__file__).parent / "data"
    
    st.header("📁 Подготовка данных")
    
    st.markdown("""
    Для обучения нужны текстовые данные. Модель учится предсказывать следующий символ.
    
    **Поддерживаемые форматы:**
    - 📄 `.txt` — обычный текст
    - 📋 `.json` — форумные сообщения (структура: `{"User": ..., "messages": {...}}`)
    
    **Рекомендации по размеру:**
    - Минимум: 1MB текста (~500K символов)
    - Хорошо: 10-100MB
    - Идеально: 1GB+
    """)
    
    st.divider()
    
    # Sample данные
    st.subheader("1. Создать sample датасет")
    
    st.markdown("""
    Sample датасет — небольшой текст на русском (~2KB) для быстрого тестирования.
    """)
    
    if st.button("🎲 Создать sample.txt", type="primary"):
        try:
            sample_path = data_dir / "sample.txt"
            prepare_sample_data(str(sample_path))
            st.success(f"✅ Создан: {sample_path}")
        except Exception as e:
            st.error(f"Ошибка: {e}")
    
    st.divider()
    
    # Загрузка своих данных
    st.subheader("2. Загрузить свои данные")
    
    uploaded_file = st.file_uploader(
        "Загрузите файл (.txt или .json)",
        type=["txt", "json"],
        help="Текстовый файл (.txt) или JSON с форумными сообщениями"
    )
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_ext = Path(file_name).suffix.lower()
        
        # Обработка в зависимости от типа файла
        if file_ext == '.json':
            # JSON с форумными сообщениями
            import json
            from data import convert_forum_json_to_text
            
            # Сохраняем временно JSON
            temp_json_path = data_dir / f"temp_{file_name}"
            temp_json_path.parent.mkdir(exist_ok=True)
            temp_json_path.write_bytes(uploaded_file.read())
            
            # Конвертируем в текст
            try:
                content = convert_forum_json_to_text(temp_json_path, include_topics=True)
                st.success(f"✅ JSON загружен и конвертирован: {len(content):,} символов")
                
                # Показываем статистику JSON
                with open(temp_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    topics = len(data.get('messages', {}))
                    messages = sum(len(msgs) for msgs in data.get('messages', {}).values())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Топиков", f"{topics:,}")
                with col2:
                    st.metric("Сообщений", f"{messages:,}")
                with col3:
                    st.metric("Символов", f"{len(content):,}")
                
            except Exception as e:
                st.error(f"Ошибка обработки JSON: {e}")
                content = None
        else:
            # Обычный текстовый файл
            content = uploaded_file.read().decode('utf-8')
            st.success(f"✅ Загружено: {len(content):,} символов")
            
            # Статистика
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Символов", f"{len(content):,}")
            with col2:
                st.metric("Слов", f"{len(content.split()):,}")
            with col3:
                st.metric("Размер", f"{len(content.encode('utf-8')) / 1024:.1f} KB")
        
        if content:
            # Превью
            with st.expander("Превью (первые 500 символов)"):
                st.text(content[:500] + "..." if len(content) > 500 else content)
            
            # Сохранение
            default_name = "my_corpus.txt" if file_ext == '.json' else file_name
            save_path = st.text_input("Сохранить как", value=str(data_dir / default_name))
            
            if st.button("💾 Сохранить", type="primary"):
                try:
                    Path(save_path).parent.mkdir(exist_ok=True)
                    Path(save_path).write_text(content, encoding='utf-8')
                    st.success(f"✅ Сохранено: {save_path}")
                except Exception as e:
                    st.error(f"Ошибка сохранения: {e}")
    
    st.divider()
    
    # Конвертация JSON
    st.subheader("4. Конвертировать JSON → TXT")
    
    st.markdown("""
    Если у вас есть JSON файл с форумными сообщениями, конвертируйте его в текст для обучения.
    """)
    
    data_dir = Path(__file__).parent / "data"
    if data_dir.exists():
        json_files = list(data_dir.glob("*.json"))
        
        if json_files:
            selected_json = st.selectbox(
                "Выберите JSON файл",
                options=[f.name for f in json_files]
            )
            
            include_topics = st.checkbox(
                "Включить названия топиков",
                value=True,
                help="Добавляет названия тем как контекст для каждой группы сообщений"
            )
            
            if st.button("🔄 Конвертировать", type="primary"):
                try:
                    from data import convert_forum_json_to_text
                    
                    json_path = data_dir / selected_json
                    text = convert_forum_json_to_text(str(json_path), include_topics=include_topics)
                    
                    # Сохраняем
                    output_name = selected_json.replace('.json', '_converted.txt')
                    output_path = data_dir / output_name
                    Path(output_path).write_text(text, encoding='utf-8')
                    
                    st.success(f"✅ Конвертировано: {output_path}")
                    st.info(f"📊 Размер: {len(text):,} символов ({len(text) / 1024:.1f} KB)")
                except Exception as e:
                    st.error(f"Ошибка конвертации: {e}")
        else:
            st.info("Нет JSON файлов в data/")
    
    st.divider()
    
    # Конвертация PDF
    st.subheader("5. Конвертер PDF → TXT")
    
    st.markdown("""
    Загрузите PDF книгу/учебник, конвертируем в чистый текст для обучения.
    """)
    
    uploaded_pdf = st.file_uploader(
        "Загрузите PDF файл",
        type=["pdf"],
        help="PDF книга будет конвертирована в текст",
        key="pdf_uploader"
    )
    
    if uploaded_pdf is not None:
        pdf_name = uploaded_pdf.name
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📄 Файл: {pdf_name}")
        with col2:
            size_mb = len(uploaded_pdf.getvalue()) / (1024 * 1024)
            st.info(f"📦 Размер: {size_mb:.2f} MB")
        
        clean_text = st.checkbox(
            "Очистить текст",
            value=True,
            help="Убрать лишние пробелы, восстановить параграфы",
            key="pdf_clean"
        )
        
        output_name = st.text_input(
            "Имя выходного файла",
            value=pdf_name.replace('.pdf', '_converted.txt'),
            key="pdf_output_name"
        )
        
        if st.button("🔄 Конвертировать PDF", type="primary", key="convert_pdf_btn"):
            try:
                # Создаём директории
                input_pdf_dir = data_dir / "input_pdf"
                input_pdf_dir.mkdir(exist_ok=True)
                
                # Сохраняем PDF
                pdf_path = input_pdf_dir / pdf_name
                pdf_path.write_bytes(uploaded_pdf.getvalue())
                st.info(f"💾 Сохранен PDF: {pdf_path}")
                
                # Конвертируем
                output_path = data_dir / output_name
                with st.spinner("⏳ Конвертация PDF в текст..."):
                    result = convert_pdf_to_text(
                        str(pdf_path),
                        str(output_path),
                        clean=clean_text
                    )
                
                st.success(f"✅ Конвертировано: {output_path}")
                
                # Статистика
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Страниц", result['pages'])
                with col2:
                    st.metric("Символов", f"{result['chars']:,}")
                with col3:
                    st.metric("Слов", f"{result['words']:,}")
                with col4:
                    st.metric("Размер", f"{result['size_kb']:.1f} KB")
                
                # Превью
                text_content = output_path.read_text(encoding='utf-8')
                with st.expander("Превью (первые 500 символов)"):
                    st.text(text_content[:500] + "..." if len(text_content) > 500 else text_content)
                    
            except Exception as e:
                st.error(f"❌ Ошибка конвертации: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    st.divider()
    
    # Объединение датасетов
    st.subheader("6. Объединить датасеты")
    
    st.markdown("""
    Объедините несколько текстовых файлов в один датасет для обучения.
    """)
    
    data_dir = Path(__file__).parent / "data"
    if data_dir.exists():
        txt_files = sorted([f.name for f in data_dir.glob("*.txt")])
        
        if len(txt_files) >= 2:
            selected_files = st.multiselect(
                "Выберите файлы для объединения (минимум 2)",
                options=txt_files,
                help="Выберите 2 или больше файлов",
                key="merge_files_select"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                add_headers = st.checkbox(
                    "Добавить заголовки",
                    value=True,
                    help="Добавить имя файла перед каждым датасетом",
                    key="merge_add_headers"
                )
            with col2:
                separator = st.text_input(
                    "Разделитель",
                    value="\n\n---\n\n",
                    help="Разделитель между датасетами",
                    key="merge_separator"
                )
            
            output_merged_name = st.text_input(
                "Имя объединенного файла",
                value="combined_dataset.txt",
                key="merge_output_name"
            )
            
            # Показываем превью выбранных файлов
            if selected_files:
                with st.expander(f"📊 Анализ выбранных файлов ({len(selected_files)})"):
                    total_size = 0
                    total_lines = 0
                    
                    for fname in selected_files:
                        fpath = data_dir / fname
                        size_kb = fpath.stat().st_size / 1024
                        lines = len(fpath.read_text(encoding='utf-8').splitlines())
                        total_size += size_kb
                        total_lines += lines
                        st.text(f"📄 {fname}: {size_kb:.1f} KB, {lines:,} строк")
                    
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Итого размер", f"{total_size:.1f} KB")
                    with col2:
                        st.metric("Итого строк", f"{total_lines:,}")
            
            if st.button("🔗 Объединить датасеты", type="primary", disabled=len(selected_files) < 2, key="merge_btn"):
                try:
                    file_paths = [str(data_dir / fname) for fname in selected_files]
                    output_path = str(data_dir / output_merged_name)
                    
                    with st.spinner("⏳ Объединение файлов..."):
                        result = merge_text_files(
                            file_paths,
                            output_path,
                            separator=separator,
                            add_headers=add_headers,
                            analyze=True
                        )
                    
                    st.success(f"✅ Объединено в: {output_path}")
                    
                    # Показываем анализ объединенного файла
                    st.markdown("**📊 Анализ объединенного датасета:**")
                    
                    stats = result['analysis']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Размер", f"{stats['size_mb']:.2f} MB")
                    with col2:
                        st.metric("Символов", f"{stats['total_chars']:,}")
                    with col3:
                        st.metric("Слов", f"{stats['total_words']:,}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Строк", f"{stats['total_lines']:,}")
                    with col2:
                        cyrillic_pct = stats['cyrillic_percent']
                        st.metric("Кириллица", f"{cyrillic_pct:.1f}%")
                    with col3:
                        latin_pct = stats['latin_percent']
                        st.metric("Латиница", f"{latin_pct:.1f}%")
                    
                    # Предупреждение о языке
                    if stats['size_mb'] > 1:
                        if cyrillic_pct > 80:
                            st.success("✅ Хороший русскоязычный датасет!")
                        elif latin_pct > 80:
                            st.info("ℹ️ Датасет на английском языке")
                        else:
                            st.warning("⚠️ Смешанный язык — модель может путаться")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка объединения: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("Нужно минимум 2 текстовых файла для объединения")
    
    st.divider()
    
    # Список существующих файлов
    st.subheader("7. Доступные датасеты")
    
    data_dir = Path(__file__).parent / "data"
    if data_dir.exists():
        txt_files = list(data_dir.glob("*.txt"))
        json_files = list(data_dir.glob("*.json"))
        
        if txt_files or json_files:
            st.markdown("**Текстовые файлы (.txt):**")
            if txt_files:
                for f in txt_files:
                    size_kb = f.stat().st_size / 1024
                    st.text(f"📄 {f.name} ({size_kb:.1f} KB)")
            else:
                st.caption("Нет .txt файлов")
            
            st.markdown("**JSON файлы (.json):**")
            if json_files:
                for f in json_files:
                    size_kb = f.stat().st_size / 1024
                    st.text(f"📋 {f.name} ({size_kb:.1f} KB)")
            else:
                st.caption("Нет .json файлов")
        else:
            st.info("Нет датасетов. Загрузите файл или создайте sample.")
    else:
        st.warning("Директория data/ не найдена")


def tab_info():
    """Вкладка с информацией."""
    st.header("ℹ️ О проекте")
    
    st.markdown("""
    # PyTorch LLM — полноценный трансформер
    
    Авторегрессионная языковая модель (GPT-подобная архитектура) на PyTorch.
    
    ## Особенности
    
    ✅ Полная архитектура трансформера  
    ✅ Multi-head self-attention  
    ✅ Production-ready training  
    ✅ Дообучение (fine-tuning)  
    ✅ Гибкая конфигурация  
    
    ## Размеры моделей
    
    | Конфиг | Параметры | Время обучения |
    |--------|-----------|----------------|
    | small | ~13M | 1 час (10 эпох) |
    | medium | ~50M | 4 часа (10 эпох) |
    | base | ~117M | 1-2 дня (10 эпох) |
    
    ## Быстрый старт
    
    ### 1. Подготовить данные
    ```bash
    python train.py --prepare-sample
    ```
    
    ### 2. Обучить модель
    ```bash
    python train.py --config small --epochs 10
    ```
    
    ### 3. Запустить интерфейс
    ```bash
    streamlit run app.py
    ```
    
    ## CLI инструменты
    
    - `train.py` — обучение и дообучение
    - `inference.py` — генерация в терминале
    - `example.py` — полный пример использования
    
    ## Документация
    
    - [README.md](README.md) — подробная документация
    - [QUICKSTART.md](QUICKSTART.md) — быстрый старт
    
    ## Технологии
    
    - PyTorch 2.0+
    - Streamlit
    - AdamW optimizer
    - Cosine LR schedule
    - Gradient clipping
    """)
    
    st.divider()

    # Управление скачанными моделями HF
    st.subheader("🗂️ Скачанные модели HF")
    _hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    _cached = sorted([d for d in _hf_cache.iterdir() if d.is_dir() and d.name.startswith("models--")]) if _hf_cache.exists() else []

    if not _cached:
        st.info("Нет скачанных моделей в ~/.cache/huggingface/hub/")
    else:
        for _model_dir in _cached:
            _display_name = _model_dir.name.replace("models--", "").replace("--", "/")
            _size_bytes = sum(f.stat().st_size for f in _model_dir.rglob("*") if f.is_file())
            _size_gb = _size_bytes / 1024**3
            _col1, _col2 = st.columns([4, 1])
            with _col1:
                st.text(f"📦 {_display_name}  ({_size_gb:.2f} GB)")
            with _col2:
                if st.button("🗑️ Удалить", key=f"del_{_model_dir.name}"):
                    st.session_state[f"confirm_del_{_model_dir.name}"] = True

            if st.session_state.get(f"confirm_del_{_model_dir.name}"):
                st.warning(f"Удалить **{_display_name}** ({_size_gb:.2f} GB)?")
                _c1, _c2 = st.columns(2)
                with _c1:
                    if st.button("✅ Да, удалить", key=f"yes_{_model_dir.name}"):
                        import shutil
                        shutil.rmtree(_model_dir)
                        st.session_state.pop(f"confirm_del_{_model_dir.name}", None)
                        st.success(f"Удалено: {_display_name}")
                        st.rerun()
                with _c2:
                    if st.button("❌ Отмена", key=f"no_{_model_dir.name}"):
                        st.session_state.pop(f"confirm_del_{_model_dir.name}", None)
                        st.rerun()

    st.divider()

    # System info
    st.subheader("💻 Системная информация")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("PyTorch версия", torch.__version__)
        
        from config import get_device
        device = get_device()
        device_emoji = {"mps": "🍎", "cuda": "🎮", "cpu": "💻"}
        st.metric("Устройство", f"{device_emoji.get(device, '')} {device.upper()}")
    
    with col2:
        if torch.backends.mps.is_available():
            st.success("✅ MPS доступен (Apple Silicon GPU)")
            st.info("Обучение будет на GPU через Metal")
        elif torch.cuda.is_available():
            st.metric("GPU", torch.cuda.get_device_name(0))
            st.metric("CUDA версия", torch.version.cuda)
        else:
            st.warning("⚠️ GPU не обнаружен")
            st.info("Обучение будет на CPU (медленнее в 10-50 раз)")


def tab_logs():
    """Вкладка просмотра логов в реальном времени."""
    st.header("📊 Просмотр логов")
    
    # Путь к логам
    logs_dir = Path(__file__).parent / "logs"
    
    # Инициализация session state для автообновления
    if 'logs_auto_refresh' not in st.session_state:
        st.session_state.logs_auto_refresh = False
    if 'logs_last_refresh' not in st.session_state:
        st.session_state.logs_last_refresh = time.time()
    
    # Проверка директории
    if not logs_dir.exists():
        st.error(f"📁 Директория логов не найдена: {logs_dir}")
        return
    
    # Получение списка файлов логов
    log_files = sorted([f.name for f in logs_dir.glob("*.log")])
    
    if not log_files:
        st.warning("⚠️ Нет файлов логов в директории")
        st.info(f"Путь: {logs_dir}")
        return
    
    # Контролы
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_log = st.selectbox(
            "📄 Выберите файл лога",
            options=log_files,
            help="Файлы из /logs директории",
            key="log_file_select"
        )
    
    with col2:
        log_level = st.selectbox(
            "🔍 Фильтр по уровню",
            options=["ALL", "INFO", "WARNING", "ERROR", "DEBUG"],
            index=0,
            help="Показывать только логи выбранного уровня",
            key="log_level_filter"
        )
    
    with col3:
        # Кнопка Start/Stop автообновления
        if st.session_state.logs_auto_refresh:
            if st.button("⏸️ Стоп", type="secondary", key="stop_refresh"):
                st.session_state.logs_auto_refresh = False
                st.rerun()
        else:
            if st.button("▶️ Старт", type="primary", key="start_refresh"):
                st.session_state.logs_auto_refresh = True
                st.session_state.logs_last_refresh = time.time()
                st.rerun()
    
    # Настройки отображения
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_lines = st.slider(
            "📏 Количество строк",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Показать последние N строк",
            key="log_num_lines"
        )
    
    with col2:
        show_timestamps = st.checkbox(
            "🕐 Показать время",
            value=True,
            help="Отображать временные метки",
            key="log_show_timestamps"
        )
    
    with col3:
        highlight_keywords = st.checkbox(
            "🎨 Подсветка",
            value=True,
            help="Цветная подсветка ключевых слов",
            key="log_highlight"
        )
    
    st.divider()
    
    # Статус автообновления
    if st.session_state.logs_auto_refresh:
        current_time = time.time()
        elapsed = int(current_time - st.session_state.logs_last_refresh)
        st.info(f"🔄 Автообновление активно (обновлено {elapsed} сек. назад)")
    
    # Чтение файла лога
    log_path = logs_dir / selected_log
    
    try:
        # Получаем размер файла
        file_size = log_path.stat().st_size
        
        if file_size == 0:
            st.warning("⚠️ Файл лога пустой")
            st.info("Логи появятся после запуска обучения или генерации")
            return
        
        # Читаем последние N строк
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Применяем фильтр по уровню
        if log_level != "ALL":
            lines = [line for line in lines if f"[{log_level}]" in line]
        
        # Берем последние N строк
        lines = lines[-num_lines:]
        
        # Разворачиваем порядок — свежие записи вверху
        lines = lines[::-1]
        
        # Информация о файле
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Файл", selected_log.replace('.log', ''))
        with col2:
            st.metric("📦 Размер", f"{file_size / 1024:.1f} KB")
        with col3:
            st.metric("📝 Всего строк", len(lines))
        with col4:
            if lines:
                # Извлекаем последнее время из лога (теперь первая строка — самая свежая)
                last_line = lines[0]
                if last_line.strip():
                    time_match = last_line.split()[0:2]
                    if len(time_match) == 2:
                        st.metric("🕐 Последняя запись", time_match[1])
        
        st.divider()
        
        # Отображение логов
        if not lines:
            st.info(f"ℹ️ Нет логов уровня {log_level}")
        else:
            # Создаем текст лога
            log_text = ""
            
            for line in lines:
                line = line.rstrip()
                
                if not show_timestamps:
                    # Убираем временные метки (первые 19 символов: "YYYY-MM-DD HH:MM:SS")
                    if len(line) > 19 and line[10] == ' ':
                        line = line[20:]
                
                if highlight_keywords:
                    # Применяем цветную подсветку через Markdown
                    # Epoch
                    if "Epoch" in line:
                        line = line.replace("Epoch", "**Epoch**")
                    
                    # Loss значения
                    import re
                    line = re.sub(r'(loss=[\d\.]+)', r'**\1**', line)
                    line = re.sub(r'(val_loss=[\d\.]+)', r'**\1**', line)
                    line = re.sub(r'(train_loss=[\d\.]+)', r'**\1**', line)
                    line = re.sub(r'(avg_loss=[\d\.]+)', r'**\1**', line)
                    
                    # Progress bars
                    if '█' in line or '100%' in line:
                        line = f"🔄 {line}"
                    
                    # Checkmarks
                    if '✓' in line or 'завершен' in line:
                        line = f"✅ {line}"
                    
                    # Errors
                    if '[ERROR]' in line or 'Error' in line or 'error' in line:
                        line = f"❌ {line}"
                    
                    # Warnings
                    if '[WARNING]' in line or 'Warning' in line:
                        line = f"⚠️ {line}"
                    
                    # Info
                    if '[INFO]' in line:
                        line = f"ℹ️ {line}"
                
                log_text += line + "\n"
            
            # Отображаем в text_area для возможности копирования
            st.text_area(
                "Логи:",
                value=log_text,
                height=500,
                disabled=False,
                label_visibility="collapsed"
            )
        
        # Автообновление
        if st.session_state.logs_auto_refresh:
            current_time = time.time()
            if current_time - st.session_state.logs_last_refresh >= 5:
                st.session_state.logs_last_refresh = current_time
                time.sleep(0.1)  # Небольшая задержка перед rerun
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ Ошибка чтения файла: {e}")
        import traceback
        with st.expander("Детали ошибки"):
            st.code(traceback.format_exc())


def main():
    st.title("🤖 PyTorch LLM")
    st.caption("Полноценный трансформер с нуля")
    
    # Создаём вкладки
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Генерация",
        "🎓 Обучение",
        "📁 Данные",
        "ℹ️ Инфо",
        "📊 Логи"
    ])
    
    with tab1:
        tab_generation()
    
    with tab2:
        tab_training()
    
    with tab3:
        tab_data()
    
    with tab4:
        tab_info()
    
    with tab5:
        tab_logs()


if __name__ == "__main__":
    main()
