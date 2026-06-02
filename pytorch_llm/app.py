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

# Создаём глобальный объект состояния
training_state = TrainingState()


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
            from tokenizer import TikTokenizer, CharTokenizer as NewCharTokenizer
            tok_config = checkpoint['tokenizer_config']
            if tok_config['type'] == 'tiktoken':
                tokenizer = TikTokenizer.from_dict(tok_config)
                st.info(f"✓ TikTokenizer загружен: {tokenizer.vocab_size()} токенов ({tok_config.get('encoding_name', 'cl100k_base')})")
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
    
    if not checkpoints:
        st.warning("⚠️ Нет доступных checkpoint файлов в checkpoints/")
        st.info("Сначала обучите модель на вкладке '🎓 Обучение'")
        return
    
    # ═══════════════════════════════════════════════════════
    # ТАБЛИЦА МОДЕЛЕЙ
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
            step=10
        )
    
    with col2:
        temperature = st.slider(
            "Temperature (креативность)",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="🌡️ Температура сэмплирования:\n0.1-0.3 = детерминированно, факты\n0.5-0.7 = сбалансированно (рекомендуется)\n0.8-1.0 = креативно, разнообразно"
        )
    
    with col3:
        top_k = st.slider(
            "Top-k sampling",
            min_value=1,
            max_value=100,
            value=50,
            step=5,
            help="Выбирать из топ-k наиболее вероятных токенов"
        )
    
    # Генерация
    if st.button("✨ Сгенерировать", type="primary", width='stretch'):
        if not prompt.strip():
            st.warning("Введите промпт!")
            return
        
        app_logger.info(f"UI Генерация запущена: prompt='{prompt[:50]}...', max_tokens={max_tokens}, temp={temperature}, top_k={top_k}")
        
        with st.spinner("Генерация..."):
            try:
                generated, stats = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
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
            ["🆕 С нуля", "🔄 Дообучение (Fine-tuning)"],
            disabled=training_state.active,
            help="С нуля: новая модель | Дообучение: продолжить обучение существующей модели"
        )
        
        is_finetuning = "Дообучение" in training_mode
        
        # Выбор checkpoint для дообучения
        checkpoint_to_load = None
        if is_finetuning:
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
        
        # Размер модели (только для обучения с нуля)
        if not is_finetuning:
            config_size = st.selectbox(
                "Размер модели",
                ["small", "medium", "base"],
                index=0,
                help="small: ~13M params, medium: ~50M params, base: ~117M params",
                disabled=training_state.active
            )
        else:
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

        tokenizer_type_for_defaults = "char" if is_finetuning else st.session_state.get("training_tokenizer_type", "tiktoken")
        default_advice = build_training_data_advice(
            selected_data_path_for_defaults,
            tokenizer_type_for_defaults,
            128,
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
            value=5 if is_finetuning else 3,
            disabled=training_state.active,
            help="Для дообучения обычно достаточно 3-10 эпох" if is_finetuning else None
        )
        
        batch_size = st.number_input(
            "Batch size", 
            min_value=1, 
            max_value=128, 
            value=32,
            disabled=training_state.active,
            key="train_batch_size"
        )
        
        context_len = st.number_input(
            "Context Length",
            min_value=16,
            max_value=1024,
            value=128,
            step=16,
            help="Длина контекста (окно токенов). Уменьшите для маленьких датасетов!",
            disabled=training_state.active,
            key="train_context_len"
        )
        
        lr = st.number_input(
            "Learning rate",
            min_value=1e-5,
            max_value=1e-2,
            value=3e-4,
            format="%.5f",
            help="Рекомендуется 3e-4 для трансформеров",
            disabled=training_state.active
        )
        
        patience = st.number_input(
            "Early stopping patience",
            min_value=1,
            max_value=50,
            value=10,
            help="Остановка после N проверок без улучшения val_loss. Проверка каждые 200 шагов. Рекомендуется 10-20.",
            disabled=training_state.active
        )
        
        min_epochs = st.number_input(
            "Минимальное количество эпох",
            min_value=0,
            max_value=epochs if epochs else 10,
            value=0,
            help="Гарантированное минимальное количество эпох. Early stopping не сработает до завершения этих эпох. Установите 3 для гарантии минимум 3 эпох.",
            disabled=training_state.active
        )
        
        dropout = st.slider(
            "Dropout rate",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="🛡️ Regularization: случайно выключает нейроны во время обучения. Помогает бороться с overfitting (когда train loss хороший, но val loss растет). Рекомендуется 0.1-0.2 для трансформеров.",
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
                    help="L2 regularization для весов модели. Помогает предотвратить overfitting. Рекомендуется 0.01-0.1.",
                    disabled=training_state.active
                )
                
                grad_clip = st.number_input(
                    "Gradient Clipping",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f",
                    help="Максимальная норма градиентов. Предотвращает exploding gradients. Рекомендуется 1.0.",
                    disabled=training_state.active
                )
                
                warmup_steps = st.number_input(
                    "Warmup Steps",
                    min_value=0,
                    max_value=2000,
                    value=100,
                    step=50,
                    help="Количество шагов для плавного увеличения learning rate от 0 до целевого значения. Стабилизирует начало обучения. Рекомендуется 100-500.",
                    disabled=training_state.active
                )
            
            with col2:
                gradient_accumulation_steps = st.number_input(
                    "Gradient Accumulation Steps",
                    min_value=1,
                    max_value=16,
                    value=1,
                    step=1,
                    help="Накопление градиентов перед обновлением весов. Эффективно увеличивает batch size без доп. памяти. Полезно если batch_size ограничен памятью GPU.",
                    disabled=training_state.active
                )
                
                min_delta = st.number_input(
                    "Early Stopping Min Delta",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.01,
                    step=0.005,
                    format="%.3f",
                    help="Минимальное улучшение val_loss для сброса счетчика early stopping. Рекомендуется 0.01.",
                    disabled=training_state.active
                )
                
                checkpoint_interval = st.number_input(
                    "Checkpoint Interval (steps)",
                    min_value=100,
                    max_value=2000,
                    value=500,
                    step=100,
                    help="Сохранять checkpoint каждые N шагов. Рекомендуется 500.",
                    disabled=training_state.active
                )
        
        # Токенизатор
        tokenizer_type = st.selectbox(
            "Тип токенизатора",
            options=["char", "hybrid", "tiktoken"],
            index=1 if not is_finetuning else 0,  # hybrid по умолчанию для новых моделей
            format_func=lambda x: {
                "char": "Character-level (legacy, 1 символ = 1 токен)",
                "hybrid": "Hybrid chemistry (доменные токены + fallback char)",
                "tiktoken": "TikToken BPE (быстро, эффективно, GPT-4 tokenizer)"
            }[x],
            help="Для химии: hybrid обычно стабильнее char и легче tiktoken на малых корпусах.",
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
        model_name = st.text_input(
            "📝 Название модели",
            value="chemistry_model" if "chemistry" in str(data_path).lower() else "my_model",
            disabled=training_state.active,
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
            disabled=training_state.active
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
            
            app_logger.info(f"UI Обучение запущено: mode={'fine-tuning' if is_finetuning else 'from scratch'}")
            if is_finetuning:
                app_logger.info(f"UI Checkpoint для дообучения: {checkpoint_to_load}")
            app_logger.info(f"UI Параметры: epochs={epochs}, batch_size={batch_size}, lr={lr}, data={data_path}")
            
            # Запускаем обучение в отдельном потоке
            def run_training():
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
                            eval_every=200,
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
                            tokenizer_config=tokenizer_config if isinstance(tokenizer_config, dict) else None,
                            normalize_chemistry=True,
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
                            eval_every=200,
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
                            normalize_chemistry=True,
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
                
                # Метрики в 3 колонки
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
