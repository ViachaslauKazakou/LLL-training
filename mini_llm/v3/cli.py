"""
cli.py — интерактивный консольный интерфейс MiniLLM

Запуск:
    python cli.py
    python cli.py --corpus corpus.json --model saved/model.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from model import MiniLLM

# ── Пути по умолчанию ───────────────────────────────────────────────── #
HERE         = pathlib.Path(__file__).parent
DEFAULT_CORPUS = HERE / "corpus.json"
DEFAULT_MODEL  = HERE / "saved" / "model.npz"

# ── Цвета для терминала ─────────────────────────────────────────────── #
C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "cyan":   "\033[36m",
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "red":    "\033[31m",
    "dim":    "\033[2m",
}


def color(text: str, *codes: str) -> str:
    """Оборачивает текст в ANSI-коды."""
    return "".join(C[c] for c in codes) + text + C["reset"]


def hr(char: str = "─", width: int = 60) -> str:
    return color(char * width, "dim")


# ── Отображение меню ─────────────────────────────────────────────────── #
MENU = """
{hr}
  {title}
{hr}
  {c0}  Подготовить корпус из текста
  {c1}  Загрузить корпус из JSON
  {t}  Обучить модель
  {g}  Генерировать текст
  {p}  Предсказать следующее слово
  {a}  Показать матрицу внимания
  {s}  Сохранить модель
  {l}  Загрузить модель
  {i}  Информация о модели
  {q}  Выход
{hr}
"""


def print_menu() -> None:
    print(
        MENU.format(
            hr=hr(),
            title=color("Mini LLM  —  учебный трансформер", "bold", "cyan"),
            c0=color("[P]", "green")  + " Подготовить корпус из текста",
            c1=color("[1]", "yellow") + " Загрузить корпус из JSON",
            t=color("[2]", "yellow")  + " Обучить модель",
            g=color("[3]", "yellow")  + " Генерировать текст",
            p=color("[4]", "yellow")  + " Предсказать следующее слово",
            a=color("[5]", "yellow")  + " Показать матрицу внимания",
            s=color("[6]", "yellow")  + " Сохранить модель",
            l=color("[7]", "yellow")  + " Загрузить модель",
            i=color("[8]", "yellow")  + " Информация о модели",
            q=color("[0]", "red")     + " Выход",
        )
    )


# ── Вспомогательные утилиты ввода ────────────────────────────────────── #

def ask(prompt: str, default: str = "") -> str:
    """Запрашивает строку; возвращает default при пустом вводе."""
    hint = f" [{default}]" if default else ""
    try:
        val = input(color(f"  ▶ {prompt}{hint}: ", "cyan")).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return default
    return val if val else default


def ask_int(prompt: str, default: int) -> int:
    raw = ask(prompt, str(default))
    try:
        return int(raw)
    except ValueError:
        print(color(f"  Неверное число, используется {default}", "red"))
        return default


def ask_float(prompt: str, default: float) -> float:
    raw = ask(prompt, str(default))
    try:
        return float(raw)
    except ValueError:
        print(color(f"  Неверное число, используется {default}", "red"))
        return default


# ── Обработчики пунктов меню ─────────────────────────────────────────── #

def cmd_prepare_corpus() -> None:
    """Подготовка корпуса из текстового файла."""
    import re
    from collections import Counter

    input_path = ask("Путь к текстовому файлу", "forum_messages.txt")
    output_path = ask("Сохранить корпус как", "corpus_prepared.json")
    min_words = ask_int("Минимум слов в предложении", 2)
    max_words = ask_int("Максимум слов в предложении", 15)

    try:
        with pathlib.Path(input_path).open(encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(color(f"  Файл не найден: {input_path}", "red"))
        return
    except Exception as e:
        print(color(f"  Ошибка чтения: {e}", "red"))
        return

    # Разбиваем на предложения
    sentences = re.split(r'[.!?\n]+', text)
    sentences = [re.sub(r'\s+', ' ', s).strip().lower() for s in sentences if s.strip()]

    # Фильтруем по длине
    filtered = []
    for s in sentences:
        words = s.split()
        if min_words <= len(words) <= max_words and len(s) >= 5:
            filtered.append(s)

    # Убираем дубликаты но сохраняем частоту (max 3 повтора)
    counts = Counter(filtered)
    expanded = []
    for sent, count in counts.items():
        expanded.extend([sent] * min(count, 3))

    corpus = {"sentences": expanded}

    try:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8') as f:
            import json
            json.dump(corpus, f, ensure_ascii=False, indent=2)

        print(color(f"\n  ✓ Обработано: {len(filtered)} уникальных предложений", "green"))
        print(color(f"  ✓ В корпусе: {len(expanded)} предложений", "green"))
        print(color(f"  ✓ Словарь: {len(set(' '.join(expanded).split()))} слов", "green"))
        print(color(f"  ✓ Сохранено: {output_path}\n", "green"))
    except Exception as e:
        print(color(f"  Ошибка сохранения: {e}", "red"))


def cmd_load_corpus(llm: MiniLLM) -> None:
    path = ask("Путь к corpus.json", str(DEFAULT_CORPUS))
    try:
        llm.load_corpus(path)
    except FileNotFoundError:
        print(color(f"  Файл не найден: {path}", "red"))
    except KeyError as e:
        print(color(f"  Неверный формат JSON: {e}", "red"))


def cmd_train(llm: MiniLLM) -> None:
    if not llm.vocab:
        print(color("  Сначала загрузите корпус [1]", "red"))
        return

    n_epochs = ask_int("Число эпох", 300)
    lr       = ask_float("Learning rate", 0.01)
    clip     = ask_float("Gradient clip", 1.0)

    print(color(f"\n  Обучение {n_epochs} эпох ...\n", "green"))

    def progress(epoch: int, loss: float) -> None:
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            bar_len = 30
            filled  = int(bar_len * (epoch + 1) / n_epochs)
            bar     = "█" * filled + "░" * (bar_len - filled)
            pct     = (epoch + 1) / n_epochs * 100
            print(f"  [{bar}] {pct:5.1f}%  epoch={epoch:4d}  loss={loss:.4f}")

    llm.train(n_epochs=n_epochs, lr=lr, clip=clip, on_epoch=progress)
    print(color("\n  Обучение завершено!", "green"))


def cmd_generate(llm: MiniLLM) -> None:
    if not llm.vocab:
        print(color("  Сначала загрузите корпус [1]", "red"))
        return

    prompt      = ask("Начало фразы", "кот сидел")
    n_new       = ask_int("Сколько новых слов", 6)
    temperature = ask_float("Temperature (0.1 – 2.0)", 0.6)
    top_p       = ask_float("Top-p nucleus (0.1 – 1.0)", 0.85)
    rep_penalty = ask_float("Repetition penalty (1.0 – 2.0)", 1.3)
    n_variants  = ask_int("Сколько вариантов показать", 3)

    print()
    for seed in range(n_variants):
        result = llm.generate(
            prompt, n_new=n_new, temperature=temperature, 
            top_p=top_p, repetition_penalty=rep_penalty, seed=seed
        )
        print(f"  {color(f'[{seed}]', 'yellow')} {result}")
    print()


def cmd_predict(llm: MiniLLM) -> None:
    if not llm.vocab:
        print(color("  Сначала загрузите корпус [1]", "red"))
        return

    context_str = ask("Контекст (слова через пробел)", "кот сидел на")
    context     = context_str.split()
    k           = ask_int("Топ-k вариантов", 5)

    predictions = llm.predict_next(context, k=k)

    print(f"\n  {color('Контекст:', 'bold')} {' '.join(context)}")
    print(f"  {color('Следующее слово:', 'bold')}")
    for rank, (word, prob) in enumerate(predictions, 1):
        bar  = "█" * int(prob * 40)
        pct  = prob * 100
        print(f"    {rank}. {color(f'{word:<12}', 'cyan')}  {pct:5.1f}%  {color(bar, 'green')}")
    print()


def cmd_attention(llm: MiniLLM) -> None:
    if not llm.vocab:
        print(color("  Сначала загрузите корпус [1]", "red"))
        return

    context_str = ask("Контекст (4 слова)", "кот сидел на коврике")
    context     = context_str.split()

    weights, tokens = llm.attention_weights(context)

    col_w = 12
    print(f"\n  {color('Веса внимания', 'bold')}  (строка = откуда, столбец = куда)\n")
    print("  " + " " * col_w, end="")
    for t in tokens:
        print(f"{t:>{col_w}}", end="")
    print()
    print("  " + hr("─", col_w * (len(tokens) + 1)))

    for i, row_tok in enumerate(tokens):
        print(f"  {color(f'{row_tok:<{col_w}}', 'yellow')}", end="")
        for j in range(len(tokens)):
            v   = weights[i, j]
            txt = f"{v:.3f}".rjust(col_w)
            # Подсвечиваем высокие значения
            if v > 0.5:
                txt = color(txt, "green", "bold")
            elif v > 0.2:
                txt = color(txt, "cyan")
            print(txt, end="")
        print()
    print()


def cmd_save(llm: MiniLLM) -> None:
    if llm.E is None:
        print(color("  Модель не инициализирована", "red"))
        return
    path = ask("Сохранить в файл", str(DEFAULT_MODEL))
    try:
        llm.save(path)
    except OSError as e:
        print(color(f"  Ошибка сохранения: {e}", "red"))


def cmd_load_weights(llm: MiniLLM) -> None:
    path = ask("Загрузить из файла", str(DEFAULT_MODEL))
    try:
        llm.load_weights(path)
    except FileNotFoundError:
        print(color(f"  Файл не найден: {path}", "red"))
    except Exception as e:
        print(color(f"  Ошибка загрузки: {e}", "red"))


def cmd_info(llm: MiniLLM) -> None:
    if not llm.vocab:
        print(color("  Модель не инициализирована", "red"))
        return

    info = llm.info()
    print()
    rows = [
        ("Размер словаря",   f"{info['vocab_size']} слов"),
        ("d_model",          str(info["d_model"])),
        ("Длина контекста",  str(info["context_len"])),
        ("Всего параметров", f"{info['total_params']:,}"),
        ("Обучена",          "да" if info["is_trained"] else "нет"),
        ("Эпох обучено",     str(info["epochs_trained"])),
        ("Последний loss",   f"{info['last_loss']:.4f}" if info["last_loss"] else "—"),
    ]
    for label, value in rows:
        print(f"  {color(f'{label}:', 'yellow')}  {value}")

    if llm.train_history:
        # Мини-график loss прямо в консоли
        history = llm.train_history
        n       = min(40, len(history))
        step    = max(1, len(history) // n)
        sampled = history[::step][:n]
        lo, hi  = min(sampled), max(sampled)
        span    = hi - lo or 1
        height  = 6
        cols    = [int((v - lo) / span * (height - 1)) for v in sampled]

        print(f"\n  {color('Loss (история обучения):', 'bold')}")
        for row in range(height - 1, -1, -1):
            line = "  "
            for c in cols:
                line += color("█", "green") if c >= row else " "
            print(line)
    print()


# ── Главный цикл ─────────────────────────────────────────────────────── #

COMMANDS = {
    "p": lambda llm: cmd_prepare_corpus(),
    "P": lambda llm: cmd_prepare_corpus(),
    "1": cmd_load_corpus,
    "2": cmd_train,
    "3": cmd_generate,
    "4": cmd_predict,
    "5": cmd_attention,
    "6": cmd_save,
    "7": cmd_load_weights,
    "8": cmd_info,
}


def main(args: argparse.Namespace) -> None:
    llm = MiniLLM(d_model=args.d_model, context_len=args.context_len)

    # Автозагрузка если переданы аргументы
    if args.model and pathlib.Path(args.model).exists():
        llm.load_weights(args.model)
    elif args.corpus and pathlib.Path(args.corpus).exists():
        llm.load_corpus(args.corpus)

    print_menu()

    while True:
        try:
            choice = input(color("\n  Выбор [P/0-8]: ", "bold")).strip().lower()
        except (KeyboardInterrupt, EOFError):
            choice = "0"

        if choice == "0":
            print(color("\n  До свидания!\n", "cyan"))
            sys.exit(0)

        handler = COMMANDS.get(choice)
        if handler:
            print()
            handler(llm)
        else:
            print(color("  Неизвестная команда. Введите P или 0–8.", "red"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mini LLM — интерактивный консольный интерфейс"
    )
    parser.add_argument(
        "--corpus", default="", help="Путь к corpus.json для автозагрузки"
    )
    parser.add_argument(
        "--model", default="", help="Путь к .npz файлу модели для автозагрузки"
    )
    parser.add_argument("--d-model", type=int, default=16, help="Размер эмбеддингов (default: 16)")
    parser.add_argument("--context-len", type=int, default=4, help="Длина контекста (default: 4)")

    main(parser.parse_args())
