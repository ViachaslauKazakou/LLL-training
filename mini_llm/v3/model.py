"""
model.py — класс MiniLLM

Учебная реализация авторегрессионного трансформера на чистом NumPy.

Архитектура (упрощённый GPT):
  Token Embeddings + Positional Encoding
  → Single-head Masked Self-Attention
  → Linear projection  →  Logits  →  Softmax
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable

import numpy as np


class MiniLLM:
    """
    Миниатюрный авторегрессионный трансформер.

    Параметры
    ---------
    d_model     : размерность эмбеддингов
    context_len : длина входного окна (число токенов)
    seed        : seed для воспроизводимой инициализации весов
    """

    # ------------------------------------------------------------------ #
    #  Инициализация                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, d_model: int = 16, context_len: int = 4, seed: int = 42):
        self.D = d_model
        self.T = context_len
        self._seed = seed

        # Словарь
        self.vocab: list[str] = []
        self.w2i: dict[str, int] = {}
        self.i2w: dict[int, str] = {}
        self.V: int = 0

        # Веса модели
        self.E:  np.ndarray | None = None   # Embedding table   (V, D)
        self.Wq: np.ndarray | None = None   # Query weights     (D, D)
        self.Wk: np.ndarray | None = None   # Key weights       (D, D)
        self.Wv: np.ndarray | None = None   # Value weights     (D, D)
        self.Wo: np.ndarray | None = None   # Output projection (D, V)

        # Кэшированное позиционное кодирование — пересчитывается при смене T/D
        self._PE: np.ndarray | None = None

        self.is_trained: bool = False
        self.train_history: list[float] = []

    # ------------------------------------------------------------------ #
    #  Загрузка данных                                                     #
    # ------------------------------------------------------------------ #

    def load_corpus(self, path: str | pathlib.Path) -> None:
        """Строит словарь из JSON-файла корпуса и инициализирует веса."""
        path = pathlib.Path(path)
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)

        words = " ".join(raw["sentences"]).split()

        self.vocab = sorted(set(words))
        self.w2i   = {w: i for i, w in enumerate(self.vocab)}
        self.i2w   = {i: w for w, i in self.w2i.items()}
        self.V     = len(self.vocab)
        self._data = [self.w2i[w] for w in words]

        self._init_weights()
        print(f"[corpus]  {len(words)} токенов · словарь: {self.V} слов")
        print(f"[vocab]   {self.vocab}")

    # ------------------------------------------------------------------ #
    #  Инициализация весов                                                 #
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        rng   = np.random.default_rng(self._seed)
        scale = 0.01
        D, V  = self.D, self.V

        self.E  = rng.normal(0, scale, (V, D))
        self.Wq = rng.normal(0, scale, (D, D))
        self.Wk = rng.normal(0, scale, (D, D))
        self.Wv = rng.normal(0, scale, (D, D))
        self.Wo = rng.normal(0, scale, (D, V))

        self._PE = self._build_pos_enc()

        self.is_trained   = False
        self.train_history = []

    # ------------------------------------------------------------------ #
    #  Вспомогательные функции                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically-stable softmax."""
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def _build_pos_enc(self) -> np.ndarray:
        """
        Синусоидальное позиционное кодирование (Vaswani et al., 2017).

        pe[pos, 2i]   = sin(pos / 10000^(2i/D))
        pe[pos, 2i+1] = cos(pos / 10000^(2i/D))
        """
        T, D = self.T, self.D
        pe  = np.zeros((T, D))
        pos = np.arange(T)[:, None]
        div = np.exp(np.arange(0, D, 2) * (-np.log(10000.0) / D))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return pe

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def _forward(self, token_ids: list[int]) -> tuple[np.ndarray, tuple]:
        """
        Прямой проход модели.

        token_ids : T целых чисел (индексы в словаре)
        Возвращает:
            logits (V,)  — оценки для каждого слова словаря
            cache        — всё нужное для backward pass
        """
        # 1. Эмбеддинги + позиционное кодирование
        X  = self.E[token_ids] + self._PE        # (T, D)

        # 2. Проекции Q / K / V
        Q  = X @ self.Wq                         # (T, D)
        K  = X @ self.Wk                         # (T, D)
        Vm = X @ self.Wv                         # (T, D)

        # 3. Скоры внимания со стабилизатором √D
        scores = Q @ K.T / np.sqrt(self.D)       # (T, T)

        # 4. Causal mask — токен не видит будущее
        mask   = np.triu(np.ones((self.T, self.T)), k=1) * -1e9
        A      = self._softmax(scores + mask)    # (T, T) — веса внимания

        # 5. Взвешенная сумма Value-векторов
        ctx    = A @ Vm                          # (T, D)

        # 6. Берём только последний токен и проецируем в словарь
        last   = ctx[-1]                         # (D,)
        logits = last @ self.Wo                  # (V,)

        cache = (token_ids, X, Q, K, Vm, A, ctx, last)
        return logits, cache

    # ------------------------------------------------------------------ #
    #  Backward pass                                                       #
    # ------------------------------------------------------------------ #

    def _loss_and_grads(
        self, token_ids: list[int], target_id: int
    ) -> tuple[float, dict[str, np.ndarray]]:
        """
        Cross-entropy loss + градиенты по всем параметрам.

        Градиент течёт:
          dlogits → dWo, dlast → dctx → dA, dVm → dWq, dWk, dWv → dE
        """
        logits, cache = self._forward(token_ids)
        ids_, X, Q, K, Vm, A, ctx, last = cache

        # ── Loss ──────────────────────────────────────────────────────── #
        probs = self._softmax(logits)
        loss  = -np.log(probs[target_id] + 1e-9)

        # ── dL / d(logits)  (cross-entropy + softmax → probs - 1_target) #
        dlogits            = probs.copy()
        dlogits[target_id] -= 1.0                              # (V,)

        # ── Output layer: last @ Wo = logits ─────────────────────────── #
        dWo   = last[:, None] @ dlogits[None, :]               # (D, V)
        dlast = self.Wo @ dlogits                              # (D,)

        # ── Gradient через последний токен контекста ──────────────────── #
        dctx      = np.zeros_like(ctx)
        dctx[-1]  = dlast                                      # (T, D)

        # ── Attention: ctx = A @ Vm ───────────────────────────────────── #
        dA  = dctx @ Vm.T                                      # (T, T)
        dVm = A.T  @ dctx                                      # (T, D)

        dWv       = X.T @ dVm                                  # (D, D)
        dX_from_v = dVm @ self.Wv.T                            # (T, D)

        # ── Softmax backward для A ────────────────────────────────────── #
        dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(self.D)                    # (T, T)

        # ── Scores = Q @ K.T ─────────────────────────────────────────── #
        dQ = dscores   @ K                                     # (T, D)
        dK = dscores.T @ Q                                     # (T, D)

        dWq = X.T @ dQ                                         # (D, D)
        dWk = X.T @ dK                                         # (D, D)

        # ── Суммарный градиент по входу X ────────────────────────────── #
        dX = dQ @ self.Wq.T + dK @ self.Wk.T + dX_from_v     # (T, D)

        # ── Эмбеддинги — только затронутые строки ────────────────────── #
        dE = np.zeros_like(self.E)
        for t, idx in enumerate(ids_):
            dE[idx] += dX[t]

        grads = dict(E=dE, Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo)
        return loss, grads

    # ------------------------------------------------------------------ #
    #  Обучение (SGD + gradient clipping)                                 #
    # ------------------------------------------------------------------ #

    def train(
        self,
        n_epochs: int = 300,
        lr: float = 0.01,
        clip: float = 1.0,
        on_epoch: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        """
        Обучает модель методом SGD с gradient clipping.

        Параметры
        ---------
        n_epochs  : число эпох
        lr        : learning rate
        clip      : порог gradient clipping
        on_epoch  : колбэк(epoch, avg_loss) — вызывается после каждой эпохи
        """
        # Строим пары (контекст T токенов → следующий токен)
        pairs = [
            (self._data[i : i + self.T], self._data[i + self.T])
            for i in range(len(self._data) - self.T)
        ]

        # Ссылки на параметры модели для удобного цикла обновления
        param_refs = [
            ("E",  self.E),
            ("Wq", self.Wq),
            ("Wk", self.Wk),
            ("Wv", self.Wv),
            ("Wo", self.Wo),
        ]

        for epoch in range(n_epochs):
            total_loss = 0.0

            for idx in np.random.permutation(len(pairs)):
                ctx, target = pairs[idx]
                loss, grads = self._loss_and_grads(ctx, target)
                total_loss += loss

                # SGD шаг с gradient clipping
                for name, param in param_refs:
                    g    = grads[name]
                    norm = np.linalg.norm(g)
                    if norm > clip:
                        g = g / norm        # нормируем, если слишком большой
                    param -= lr * g         # in-place обновление

            avg = total_loss / len(pairs)
            self.train_history.append(avg)

            if on_epoch:
                on_epoch(epoch, avg)

        self.is_trained = True
        return self.train_history

    # ------------------------------------------------------------------ #
    #  Генерация текста                                                    #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        n_new: int = 6,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        seed: int = 1,
    ) -> str:
        """
        Генерирует текст, продолжая prompt.

        temperature > 1  →  более случайный текст
        temperature < 1  →  более детерминированный
        top_p       nucleus sampling — выбирает из топ-p% вероятностной массы
        repetition_penalty  штраф за повторение недавних токенов (1.0 = нет штрафа)
        """
        rng = np.random.default_rng(seed)
        ids = [self.w2i[w] for w in prompt.split() if w in self.w2i]

        if not ids:
            return "(нет слов из словаря в prompt)"

        for _ in range(n_new):
            ctx = ids[-self.T :]
            if len(ctx) < self.T:
                ctx = [0] * (self.T - len(ctx)) + ctx

            logits, _ = self._forward(ctx)
            
            # Repetition penalty — штрафуем токены которые уже встречались
            if repetition_penalty != 1.0:
                for token_id in set(ids):
                    logits[token_id] /= repetition_penalty
            
            probs = self._softmax(logits / max(temperature, 1e-6))

            # Nucleus sampling (top-p) — отсекаем длинный хвост
            sorted_idx  = np.argsort(probs)[::-1]
            sorted_prob = probs[sorted_idx]
            cumsum      = np.cumsum(sorted_prob)
            cutoff      = np.searchsorted(cumsum, top_p) + 1
            top_idx     = sorted_idx[:cutoff]
            top_probs   = probs[top_idx]
            top_probs   = top_probs / top_probs.sum()  # ренормализуем

            next_id = int(rng.choice(top_idx, p=top_probs))
            ids.append(next_id)

        return " ".join(self.i2w[i] for i in ids)

    # ------------------------------------------------------------------ #
    #  Предсказание следующего слова                                       #
    # ------------------------------------------------------------------ #

    def predict_next(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        """
        Возвращает топ-k вероятных следующих слов для заданного контекста.

        context : список слов (последние T будут использованы)
        k       : сколько вариантов вернуть
        """
        ids = [self.w2i.get(w, 0) for w in context[-self.T :]]
        if len(ids) < self.T:
            ids = [0] * (self.T - len(ids)) + ids

        logits, _ = self._forward(ids)
        probs     = self._softmax(logits)
        top_k     = np.argsort(probs)[::-1][:k]
        return [(self.i2w[i], float(probs[i])) for i in top_k]

    # ------------------------------------------------------------------ #
    #  Матрица внимания                                                    #
    # ------------------------------------------------------------------ #

    def attention_weights(self, context: list[str]) -> tuple[np.ndarray, list[str]]:
        """
        Возвращает матрицу весов внимания (T, T) для заданного контекста.

        context : список слов (ровно T слов; лишние обрезаются, нехватка — паддинг)
        """
        tokens = context[-self.T :]
        ids    = [self.w2i.get(w, 0) for w in tokens]
        if len(ids) < self.T:
            pad    = ["<pad>"] * (self.T - len(ids))
            tokens = pad + tokens
            ids    = [0]      * (self.T - len(ids)) + ids

        _, cache = self._forward(ids)
        _, _, _, _, _, A, _, _ = cache
        return A, tokens

    # ------------------------------------------------------------------ #
    #  Сохранение / Загрузка                                               #
    # ------------------------------------------------------------------ #

    def save(self, path: str | pathlib.Path) -> None:
        """Сохраняет веса и метаданные в .npz файл."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            E=self.E,
            Wq=self.Wq,
            Wk=self.Wk,
            Wv=self.Wv,
            Wo=self.Wo,
            vocab=np.array(self.vocab),
            D=np.int32(self.D),
            T=np.int32(self.T),
            history=np.array(self.train_history),
        )
        print(f"[save]  Модель сохранена → {path}")

    def load_weights(self, path: str | pathlib.Path) -> None:
        """Загружает веса и метаданные из .npz файла."""
        path = pathlib.Path(path)
        data = np.load(path, allow_pickle=True)

        self.E   = data["E"]
        self.Wq  = data["Wq"]
        self.Wk  = data["Wk"]
        self.Wv  = data["Wv"]
        self.Wo  = data["Wo"]

        self.vocab = list(data["vocab"])
        self.D     = int(data["D"])
        self.T     = int(data["T"])

        self.w2i  = {w: i for i, w in enumerate(self.vocab)}
        self.i2w  = {i: w for w, i in self.w2i.items()}
        self.V    = len(self.vocab)
        self._PE  = self._build_pos_enc()

        self.train_history = list(data["history"])
        self.is_trained    = True

        print(f"[load]  Модель загружена ← {path}")

    # ------------------------------------------------------------------ #
    #  Информация о модели                                                 #
    # ------------------------------------------------------------------ #

    def info(self) -> dict:
        """Возвращает словарь с параметрами и состоянием модели."""
        weights = [self.E, self.Wq, self.Wk, self.Wv, self.Wo]
        total   = sum(p.size for p in weights if p is not None)
        return {
            "vocab_size":    self.V,
            "d_model":       self.D,
            "context_len":   self.T,
            "total_params":  total,
            "is_trained":    self.is_trained,
            "epochs_trained": len(self.train_history),
            "last_loss":     self.train_history[-1] if self.train_history else None,
        }
