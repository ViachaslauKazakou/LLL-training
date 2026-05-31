"""
app.py — Streamlit интерфейс для MiniLLM

Запуск:
    streamlit run app.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import streamlit as st

# Добавляем текущую папку в sys.path чтобы импортировать model.py
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import MiniLLM  # noqa: E402

# ────────────────────────────────────────────────────────────────────── #
#  Конфигурация страницы
# ────────────────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="Mini LLM",
    page_icon="🤖",
    layout="wide",
)

HERE          = pathlib.Path(__file__).parent
DEFAULT_CORPUS = HERE / "corpus.json"
DEFAULT_MODEL  = HERE / "saved" / "model.npz"


# ────────────────────────────────────────────────────────────────────── #
#  Session state — хранит модель между перерисовками
# ────────────────────────────────────────────────────────────────────── #
if "llm" not in st.session_state:
    st.session_state.llm = MiniLLM(d_model=16, context_len=12)

llm: MiniLLM = st.session_state.llm


# ────────────────────────────────────────────────────────────────────── #
#  Sidebar — управление моделью
# ────────────────────────────────────────────────────────────────────── #
with st.sidebar:
    st.title("⚙️ Настройки")

    # ── Параметры архитектуры ───────────────────────────────────────── #
    with st.expander("Архитектура модели", expanded=True):
        d_model     = st.slider("d_model (размер эмбеддинга)", 4, 64, 16, step=4)
        context_len = st.slider("Длина контекста T", 2, 8, 4)

    # ── Подготовка корпуса ───────────────────────────────────────────── #
    st.subheader("📝 Подготовка корпуса")
    
    with st.expander("Из текстового файла"):
        uploaded_file = st.file_uploader(
            "Загрузите .txt с предложениями",
            type=['txt'],
            help="Каждая строка или предложение = отдельная фраза для обучения"
        )
        
        col_min, col_max = st.columns(2)
        with col_min:
            min_words = st.number_input("Мин слов", 2, 10, 2)
        with col_max:
            max_words = st.number_input("Макс слов", 5, 20, 15)
        
        corpus_name = st.text_input("Имя файла корпуса", "corpus_prepared.json")
        
        if st.button("🔨 Создать корпус", use_container_width=True):
            if uploaded_file:
                import re
                from collections import Counter
                
                text = uploaded_file.read().decode('utf-8')
                
                # Разбиваем на предложения
                sentences = re.split(r'[.!?\n]+', text)
                sentences = [re.sub(r'\s+', ' ', s).strip().lower() for s in sentences if s.strip()]
                
                # Фильтруем
                filtered = []
                for s in sentences:
                    words = s.split()
                    if min_words <= len(words) <= max_words and len(s) >= 5:
                        filtered.append(s)
                
                # Убираем дубликаты с сохранением частоты
                counts = Counter(filtered)
                expanded = []
                for sent, count in counts.items():
                    expanded.extend([sent] * min(count, 3))
                
                corpus = {"sentences": expanded}
                corpus_path = HERE / corpus_name
                
                import json
                with corpus_path.open('w', encoding='utf-8') as f:
                    json.dump(corpus, f, ensure_ascii=False, indent=2)
                
                vocab_size = len(set(' '.join(expanded).split()))
                
                st.success(f"✓ Создано: {len(filtered)} уникальных, {len(expanded)} в корпусе")
                st.info(f"Словарь: {vocab_size} слов\nСохранено: {corpus_name}")
            else:
                st.warning("Загрузите файл")

    # ── Загрузка корпуса ─────────────────────────────────────────────── #
    st.subheader("📂 Корпус")
    corpus_path = st.text_input("Путь к corpus.json", str(DEFAULT_CORPUS))
    if st.button("Загрузить корпус", use_container_width=True):
        st.session_state.llm = MiniLLM(d_model=d_model, context_len=context_len)
        llm = st.session_state.llm
        try:
            llm.load_corpus(corpus_path)
            st.success(f"Словарь: {llm.V} слов")
        except FileNotFoundError:
            st.error("Файл не найден")
        except Exception as e:
            st.error(str(e))

    # ── Сохранение / загрузка весов ──────────────────────────────────── #
    st.subheader("💾 Сохранение модели")
    model_path = st.text_input("Путь к .npz", str(DEFAULT_MODEL))

    col_s, col_l = st.columns(2)
    with col_s:
        if st.button("Сохранить", use_container_width=True):
            if llm.E is not None:
                llm.save(model_path)
                st.success("Сохранено!")
            else:
                st.warning("Модель не инициализирована")
    with col_l:
        if st.button("Загрузить", use_container_width=True):
            try:
                llm.load_weights(model_path)
                st.success(f"Загружено! Эпох: {len(llm.train_history)}")
            except FileNotFoundError:
                st.error("Файл не найден")
            except Exception as e:
                st.error(str(e))

    # ── Информация о модели ──────────────────────────────────────────── #
    if llm.vocab:
        st.subheader("📊 Статус")
        info = llm.info()
        st.metric("Параметров", f"{info['total_params']:,}")
        st.metric("Размер словаря", info["vocab_size"])
        st.metric("Обучена", "✅ Да" if info["is_trained"] else "❌ Нет")
        if info["last_loss"]:
            st.metric("Last loss", f"{info['last_loss']:.4f}")


# ────────────────────────────────────────────────────────────────────── #
#  Основная область — вкладки
# ────────────────────────────────────────────────────────────────────── #
st.title("🤖 Mini LLM — учебный трансформер")
st.caption("Авторегрессионный трансформер на NumPy: эмбеддинги → masked self-attention → softmax")

tab_train, tab_gen, tab_predict, tab_attn = st.tabs(
    ["🎓 Обучение", "✍️ Генерация", "🔮 Предсказание", "👁️ Внимание"]
)


# ────────────────────────────────────────────────────────────────────── #
#  Вкладка: Обучение
# ────────────────────────────────────────────────────────────────────── #
with tab_train:
    st.subheader("Обучение модели")

    if not llm.vocab:
        st.info("Сначала загрузите корпус в боковой панели.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            n_epochs = st.number_input("Эпох", 10, 1000, 300, step=10)
        with col2:
            lr = st.number_input("Learning rate", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
        with col3:
            clip = st.number_input("Gradient clip", 0.1, 10.0, 1.0, step=0.1)

        if st.button("▶ Обучить", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0, text="Начало обучения…")
            loss_placeholder = st.empty()

            epoch_losses: list[float] = []

            def on_epoch(epoch: int, loss: float) -> None:
                epoch_losses.append(loss)
                pct  = (epoch + 1) / n_epochs
                progress_bar.progress(pct, text=f"Эпоха {epoch + 1}/{n_epochs}  loss={loss:.4f}")
                if len(epoch_losses) > 1:
                    loss_placeholder.line_chart(
                        epoch_losses,
                        x_label="Эпоха",
                        y_label="Loss",
                        height=200,
                    )

            llm.train(n_epochs=n_epochs, lr=lr, clip=clip, on_epoch=on_epoch)
            progress_bar.progress(1.0, text="✅ Обучение завершено!")

        # График истории обучения (если уже обучалась)
        if llm.train_history:
            st.subheader("История loss")
            st.line_chart(llm.train_history, x_label="Эпоха", y_label="Loss", height=250)


# ────────────────────────────────────────────────────────────────────── #
#  Вкладка: Генерация
# ────────────────────────────────────────────────────────────────────── #
with tab_gen:
    st.subheader("Генерация текста")

    if not llm.vocab:
        st.info("Сначала загрузите корпус в боковой панели.")
    else:
        prompt = st.text_input(
            "Начало фразы",
            "кот сидел",
            help=f"Слова из словаря: {', '.join(llm.vocab)}",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            n_new = st.slider("Новых слов", 1, 20, 6)
            temperature = st.slider("Temperature", 0.1, 2.0, 0.6, step=0.05)
        with col_b:
            top_p = st.slider("Top-p (nucleus)", 0.1, 1.0, 0.85, step=0.05)
            rep_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.3, step=0.1)

        n_variants = st.slider("Сколько вариантов", 1, 8, 3)

        if st.button("✨ Генерировать", type="primary", use_container_width=True):
            st.subheader("Результаты")
            for seed in range(n_variants):
                result = llm.generate(
                    prompt, n_new=n_new, temperature=temperature, 
                    top_p=top_p, repetition_penalty=rep_penalty, seed=seed
                )
                # Подсвечиваем сгенерированные слова
                original_count = len([w for w in prompt.split() if w in llm.w2i])
                words = result.split()
                orig  = " ".join(words[:original_count])
                gen   = " ".join(words[original_count:])
                st.markdown(
                    f"**[{seed}]** {orig} "
                    f"<span style='color:#00c853;font-weight:bold'>{gen}</span>",
                    unsafe_allow_html=True,
                )


# ────────────────────────────────────────────────────────────────────── #
#  Вкладка: Предсказание следующего слова
# ────────────────────────────────────────────────────────────────────── #
with tab_predict:
    st.subheader("Предсказание следующего слова")

    if not llm.vocab:
        st.info("Сначала загрузите корпус в боковой панели.")
    else:
        context_str = st.text_input("Контекст (слова через пробел)", "кот сидел на")
        k           = st.slider("Топ-k вариантов", 1, llm.V, min(5, llm.V))

        if st.button("🔮 Предсказать", type="primary", use_container_width=True):
            context     = context_str.split()
            predictions = llm.predict_next(context, k=k)

            st.subheader(f"Контекст: *{' '.join(context)}*")
            for rank, (word, prob) in enumerate(predictions, 1):
                col_w, col_p, col_b = st.columns([2, 1, 5])
                with col_w:
                    st.markdown(f"**{rank}. {word}**")
                with col_p:
                    st.markdown(f"`{prob * 100:.1f}%`")
                with col_b:
                    st.progress(prob)


# ────────────────────────────────────────────────────────────────────── #
#  Вкладка: Матрица внимания
# ────────────────────────────────────────────────────────────────────── #
with tab_attn:
    st.subheader("Визуализация весов внимания")

    if not llm.vocab:
        st.info("Сначала загрузите корпус в боковой панели.")
    else:
        context_str = st.text_input(
            f"Контекст ({llm.T} слова)",
            " ".join(list(llm.vocab)[: llm.T]),
            key="attn_ctx",
        )

        if st.button("👁️ Показать внимание", type="primary", use_container_width=True):
            try:
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors

                weights, tokens = llm.attention_weights(context_str.split())

                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(weights, cmap="YlOrRd", vmin=0, vmax=1)
                fig.colorbar(im, ax=ax)

                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=12)
                ax.set_yticklabels(tokens, fontsize=12)
                ax.set_xlabel("Куда смотрим", fontsize=11)
                ax.set_ylabel("Откуда смотрим", fontsize=11)
                ax.set_title("Веса self-attention", fontsize=13)

                # Подписи значений в ячейках
                for i in range(len(tokens)):
                    for j in range(len(tokens)):
                        v = weights[i, j]
                        ax.text(
                            j, i, f"{v:.2f}",
                            ha="center", va="center",
                            color="black" if v < 0.6 else "white",
                            fontsize=9,
                        )

                fig.tight_layout()
                st.pyplot(fig)

            except ImportError:
                # matplotlib недоступен — показываем таблицу
                import pandas as pd
                df = pd.DataFrame(weights, index=tokens, columns=tokens)
                st.dataframe(df.style.background_gradient(cmap="YlOrRd"))

        st.caption(
            "Causal mask: нижнетреугольная матрица — токен может смотреть "
            "только на себя и предыдущие токены (как в GPT)."
        )
