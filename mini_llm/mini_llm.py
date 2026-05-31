import numpy as np

# ══════════════════════════════════════════════════════════
#  1. ДАННЫЕ И СЛОВАРЬ
# ══════════════════════════════════════════════════════════
corpus = "кот сидел на коврике кот смотрел в окно кот прыгнул на стол кот лежал на диване"
words  = corpus.split()

vocab  = sorted(set(words))
w2i    = {w: i for i, w in enumerate(vocab)}  # слово → индекс
i2w    = {i: w for w, i in w2i.items()}        # индекс → слово

V      = len(vocab)   # размер словаря
D      = 8            # размер эмбеддинга
T      = 4            # длина контекста (окно)

print(f"Словарь ({V} слов): {vocab}")

# Превращаем корпус в индексы
data = [w2i[w] for w in words]

# ══════════════════════════════════════════════════════════
#  2. ПАРАМЕТРЫ МОДЕЛИ (случайная инициализация)
# ══════════════════════════════════════════════════════════
rng = np.random.default_rng(42)

# Таблица эмбеддингов: каждое слово → вектор размера D
E  = rng.normal(0, 0.1, (V, D))   # (vocab_size, d_model)

# Веса self-attention
Wq = rng.normal(0, 0.1, (D, D))   # Query
Wk = rng.normal(0, 0.1, (D, D))   # Key
Wv = rng.normal(0, 0.1, (D, D))   # Value

# Выходной (проекционный) слой: D → V
Wo = rng.normal(0, 0.1, (D, V))

# ══════════════════════════════════════════════════════════
#  3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)   # стабилизация
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def positional_encoding(T, D):
    """Синусоидальное позиционное кодирование"""
    pe = np.zeros((T, D))
    pos = np.arange(T)[:, None]              # (T, 1)
    div = np.exp(np.arange(0, D, 2) * (-np.log(10000) / D))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe

# ══════════════════════════════════════════════════════════
#  4. SELF-ATTENTION
# ══════════════════════════════════════════════════════════
def self_attention(X):
    """
    X: (T, D) — матрица эмбеддингов для T токенов
    Возвращает: (T, D) — с учётом контекста
    """
    T, D = X.shape

    Q = X @ Wq          # (T, D) — что ищем
    K = X @ Wk          # (T, D) — что предлагаем
    V_mat = X @ Wv      # (T, D) — что передаём

    # Скоры внимания: насколько каждый токен смотрит на каждый
    scores = Q @ K.T / np.sqrt(D)    # (T, T)

    # Causal mask: токен не видит будущего
    mask = np.triu(np.ones((T, T)), k=1) * -1e9
    scores = scores + mask

    weights = softmax(scores)        # (T, T) — веса внимания
    out = weights @ V_mat            # (T, D) — взвешенная сумма

    return out, weights

# ══════════════════════════════════════════════════════════
#  5. FORWARD PASS
# ══════════════════════════════════════════════════════════
def forward(token_ids):
    """
    token_ids: список из T индексов
    Возвращает: logits (V,) для следующего токена
    """
    # Эмбеддинги
    X = E[token_ids]                         # (T, D)
    X = X + positional_encoding(T, D)        # + позиции

    # Self-attention
    X, attn_weights = self_attention(X)      # (T, D)

    # Берём только последний токен для предсказания
    last = X[-1]                             # (D,)

    # Проекция в пространство словаря
    logits = last @ Wo                       # (V,)
    return logits, attn_weights

# ══════════════════════════════════════════════════════════
#  6. ГЕНЕРАЦИЯ
# ══════════════════════════════════════════════════════════
def generate(prompt_words, n_new=5, temperature=1.0, seed=0):
    """
    temperature > 1 → более случайный текст
    temperature < 1 → более детерминированный
    """
    rng_gen = np.random.default_rng(seed)
    ids = [w2i[w] for w in prompt_words if w in w2i]

    for _ in range(n_new):
        # Берём последние T токенов как контекст
        ctx = ids[-T:]
        if len(ctx) < T:
            ctx = [0] * (T - len(ctx)) + ctx   # паддинг нулями

        logits, _ = forward(ctx)

        # Temperature sampling
        probs = softmax(logits / temperature)
        next_id = rng_gen.choice(V, p=probs)
        ids.append(next_id)

    return [i2w[i] for i in ids]

# ══════════════════════════════════════════════════════════
#  7. ДЕМОНСТРАЦИЯ ВНИМАНИЯ
# ══════════════════════════════════════════════════════════
def show_attention(prompt_words):
    ids = [w2i.get(w, 0) for w in prompt_words[:T]]
    _, weights = forward(ids)

    print(f"\nВеса внимания для: {prompt_words[:T]}")
    print(f"{'':12}", end="")
    for w in prompt_words[:T]:
        print(f"{w:>10}", end="")
    print()
    for i, w_from in enumerate(prompt_words[:T]):
        print(f"{w_from:12}", end="")
        for j in range(T):
            v = weights[i, j]
            print(f"{v:>10.3f}", end="")
        print()

# ══════════════════════════════════════════════════════════
#  8. ЗАПУСК
# ══════════════════════════════════════════════════════════
print("\n=== Генерация (необученная модель) ===")
for seed in range(4):
    result = generate(["кот", "сидел"], n_new=5, temperature=0.8, seed=seed)
    print(f"  [{seed}] {' '.join(result)}")

print("\n=== Веса внимания ===")
show_attention(["кот", "сидел", "на", "коврике"])

print("\n=== Параметры модели ===")
total = E.size + Wq.size + Wk.size + Wv.size + Wo.size
print(f"  E:  {E.shape}  = {E.size} параметров")
print(f"  Wq: {Wq.shape} = {Wq.size} параметров")
print(f"  Wk: {Wk.shape} = {Wk.size} параметров")
print(f"  Wv: {Wv.shape} = {Wv.size} параметров")
print(f"  Wo: {Wo.shape} = {Wo.size} параметров")
print(f"  Итого: {total} параметров")
