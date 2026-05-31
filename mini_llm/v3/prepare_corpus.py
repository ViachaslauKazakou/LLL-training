"""
prepare_corpus.py — утилита для подготовки корпуса из текста

Использование:
    python prepare_corpus.py input.txt corpus_output.json

Берёт текст (сообщения с форума, чата) и:
1. Разбивает на предложения
2. Фильтрует слишком длинные/короткие
3. Сохраняет в формате corpus.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re


def split_into_sentences(text: str) -> list[str]:
    """Разбивает текст на предложения (упрощённая версия)."""
    # Разделители: точка, вопрос, восклицание, перенос строки
    sentences = re.split(r'[.!?\n]+', text)
    return [s.strip() for s in sentences if s.strip()]


def clean_sentence(s: str) -> str:
    """Чистит предложение — убирает лишние пробелы, приводит к нижнему регистру."""
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.lower()  # приводим к нижнему регистру для унификации
    return s


def filter_sentences(
    sentences: list[str],
    min_words: int = 2,
    max_words: int = 15,
    min_chars: int = 5,
) -> list[str]:
    """Фильтрует предложения по длине."""
    result = []
    for s in sentences:
        words = s.split()
        if min_words <= len(words) <= max_words and len(s) >= min_chars:
            result.append(s)
    return result


def prepare_corpus(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    min_words: int = 2,
    max_words: int = 15,
) -> None:
    """Обрабатывает входной текст и сохраняет corpus.json."""
    with input_path.open(encoding='utf-8') as f:
        text = f.read()

    sentences = split_into_sentences(text)
    sentences = [clean_sentence(s) for s in sentences]
    sentences = filter_sentences(sentences, min_words=min_words, max_words=max_words)

    # Убираем дубликаты (но сохраняем частоту через повторение)
    from collections import Counter
    counts = Counter(sentences)
    # Берём каждое предложение столько раз, сколько оно встречалось (но max 3)
    expanded = []
    for sent, count in counts.items():
        expanded.extend([sent] * min(count, 3))

    corpus = {"sentences": expanded}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"✓ Обработано: {len(sentences)} уникальных предложений")
    print(f"✓ В корпусе: {len(expanded)} предложений (с учётом частоты)")
    print(f"✓ Словарь: {len(set(' '.join(expanded).split()))} слов")
    print(f"✓ Сохранено: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Подготовка корпуса из текста для Mini LLM"
    )
    parser.add_argument("input", type=pathlib.Path, help="Входной текстовый файл")
    parser.add_argument("output", type=pathlib.Path, help="Выходной corpus.json")
    parser.add_argument(
        "--min-words", type=int, default=2, help="Минимум слов в предложении"
    )
    parser.add_argument(
        "--max-words", type=int, default=15, help="Максимум слов в предложении"
    )

    args = parser.parse_args()
    prepare_corpus(args.input, args.output, args.min_words, args.max_words)


if __name__ == "__main__":
    main()
