"""
tokenizer.py — Character-level и tiktoken токенизаторы

Поддерживает:
- CharTokenizer: символьный уровень (legacy, для совместимости)
- TikTokenizer: BPE токенизация через tiktoken (быстро, современно)
"""

from typing import List, Dict, Optional
from pathlib import Path
import json
import pickle
import re
from collections import Counter
import tiktoken

from logger import data_logger


SUBSCRIPT_TRANSLATION = str.maketrans({
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
})

SUPERSCRIPT_TRANSLATION = str.maketrans({
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁺': '+', '⁻': '-',
})


def normalize_chemistry_text(text: str) -> str:
    """
    Нормализует химическую запись перед токенизацией.

    Цели:
    - Снизить шум словаря (CO₂ -> CO2)
    - Привести редкие unicode-символы к стабильной ASCII-форме
    - Упростить обучение на смешанных источниках
    """
    normalized = text.replace('\u00A0', ' ')
    normalized = normalized.translate(SUBSCRIPT_TRANSLATION)
    normalized = normalized.translate(SUPERSCRIPT_TRANSLATION)

    # Унифицируем стрелки и минусы
    normalized = normalized.replace('→', '->').replace('⟶', '->').replace('⇒', '=>')
    normalized = normalized.replace('−', '-').replace('–', '-')

    return normalized


class CharTokenizer:
    """
    Character-level токенизатор (legacy).
    
    Простейший подход: каждый символ = отдельный токен.
    Хорошо работает для коротких текстов на одном языке,
    но плохо обобщает на новые слова и domain-specific термины.
    """
    
    def __init__(self, vocab: List[str]):
        """
        vocab: список всех уникальных символов в данных
        """
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        self.pad_id = self.char_to_idx.get(self.pad_token, 0)
        self.unk_id = self.char_to_idx.get(self.unk_token, 1)
        self.bos_id = self.char_to_idx.get(self.bos_token, 2)
        self.eos_id = self.char_to_idx.get(self.eos_token, 3)
    
    def encode(self, text: str) -> List[int]:
        """Преобразует текст в список индексов токенов"""
        return [
            self.char_to_idx.get(char, self.unk_id) 
            for char in text
        ]
    
    def decode(self, indices: List[int]) -> str:
        """Преобразует список индексов обратно в текст"""
        chars = []
        for idx in indices:
            if idx == self.eos_id:
                break
            if idx in [self.pad_id, self.bos_id]:
                continue
            chars.append(self.idx_to_char.get(idx, self.unk_token))
        return ''.join(chars)
    
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    @staticmethod
    def from_text(text: str) -> 'CharTokenizer':
        """Создаёт токенизатор из текста (извлекает уникальные символы)"""
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        unique_chars = sorted(set(text))
        vocab = special_tokens + unique_chars
        return CharTokenizer(vocab)
    
    def to_dict(self) -> Dict:
        """Сериализация для сохранения"""
        return {
            'type': 'char',
            'vocab': self.vocab
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'CharTokenizer':
        """Десериализация при загрузке"""
        return CharTokenizer(data['vocab'])


class TikTokenizer:
    """
    TikToken токенизатор (BPE от OpenAI).
    
    Преимущества:
    - ⚡ В 3-10 раз быстрее HuggingFace tokenizers (Rust-based)
    - 🎨 Простой API
    - 🌍 Готовые токенизаторы: cl100k_base (GPT-4), o200k_base (GPT-4o), p50k_base (Codex)
    - 🔧 Можно обучить свой (требует tiktoken_ext)
    
    Для русского + химия отлично подходит cl100k_base.
    """
    
    def __init__(
        self, 
        encoding_name: str = "cl100k_base",
        special_tokens: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            encoding_name: имя энкодера tiktoken (cl100k_base, o200k_base, p50k_base)
            special_tokens: дополнительные special tokens {token_str: token_id}
        """
        # Загружаем базовый энкодер
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
        
        # Special tokens (используем зарезервированные ID в конце словаря)
        base_vocab_size = self.encoding.n_vocab
        self.special_tokens = special_tokens or {
            '<|pad|>': base_vocab_size,
            '<|unk|>': base_vocab_size + 1,
            '<|bos|>': base_vocab_size + 2,
            '<|eos|>': base_vocab_size + 3,
        }
        
        # Создаём энкодер с special tokens
        self.encoding = tiktoken.Encoding(
            name=f"{encoding_name}_with_special",
            pat_str=self.encoding._pat_str,
            mergeable_ranks=self.encoding._mergeable_ranks,
            special_tokens=self.special_tokens
        )
        
        # Сохраняем ID special tokens
        self.pad_token = '<|pad|>'
        self.unk_token = '<|unk|>'
        self.bos_token = '<|bos|>'
        self.eos_token = '<|eos|>'
        
        self.pad_id = self.special_tokens[self.pad_token]
        self.unk_id = self.special_tokens[self.unk_token]
        self.bos_id = self.special_tokens[self.bos_token]
        self.eos_id = self.special_tokens[self.eos_token]
    
    def encode(
        self, 
        text: str, 
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[int]:
        """
        Преобразует текст в список индексов токенов.
        
        Args:
            text: входной текст
            add_bos: добавить <|bos|> в начало
            add_eos: добавить <|eos|> в конец
        
        Returns:
            Список токен IDs
        """
        # Кодируем текст, разрешая все special tokens
        tokens = self.encoding.encode(text, allowed_special="all")
        
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        
        return tokens
    
    def decode(
        self, 
        tokens: List[int], 
        skip_special_tokens: bool = True
    ) -> str:
        """
        Преобразует список токенов обратно в текст.
        
        Args:
            tokens: список токен IDs
            skip_special_tokens: убрать special tokens из вывода
        
        Returns:
            Декодированный текст
        """
        if skip_special_tokens:
            # Фильтруем special tokens
            special_ids = set(self.special_tokens.values())
            tokens = [t for t in tokens if t not in special_ids]
        
        return self.encoding.decode(tokens)
    
    def vocab_size(self) -> int:
        """Полный размер словаря (базовый + special tokens)"""
        return self.encoding.n_vocab
    
    def base_vocab_size(self) -> int:
        """Размер базового словаря (без special tokens)"""
        return self.encoding.n_vocab - len(self.special_tokens)
    
    def to_dict(self) -> Dict:
        """Сериализация для сохранения в checkpoint"""
        return {
            'type': 'tiktoken',
            'encoding_name': self.encoding_name,
            'special_tokens': self.special_tokens
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'TikTokenizer':
        """Десериализация при загрузке из checkpoint"""
        return TikTokenizer(
            encoding_name=data['encoding_name'],
            special_tokens=data['special_tokens']
        )
    
    def save(self, path: Path) -> None:
        """Сохраняет конфигурацию токенизатора"""
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        data_logger.info(f"✓ TikTokenizer сохранён: {path}")
    
    @staticmethod
    def load(path: Path) -> 'TikTokenizer':
        """Загружает токенизатор из файла"""
        data = json.loads(path.read_text())
        data_logger.info(f"✓ TikTokenizer загружен: {path}")
        return TikTokenizer.from_dict(data)


class HybridChemTokenizer:
    """
    Гибридный токенизатор для учебных chemistry-корпусов.

    Стратегия:
    1) Частотные доменные токены (формулы, слова, числа)
    2) Fallback на char-level для OOV
    """

    TOKEN_PATTERN = re.compile(
        r"[A-ZА-Я][a-zа-я]?\d*(?:[A-ZА-Я][a-zа-я]?\d*)+"
        r"|[A-Za-zА-Яа-яЁё]+"
        r"|\d+(?:[.,]\d+)?"
        r"|\s+"
        r"|.",
        re.UNICODE,
    )

    def __init__(self, vocab: List[str], min_token_freq: int = 2, max_domain_tokens: int = 2000):
        self.vocab = vocab
        self.token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        self.min_token_freq = min_token_freq
        self.max_domain_tokens = max_domain_tokens

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'

        self.pad_id = self.token_to_idx.get(self.pad_token, 0)
        self.unk_id = self.token_to_idx.get(self.unk_token, 1)
        self.bos_id = self.token_to_idx.get(self.bos_token, 2)
        self.eos_id = self.token_to_idx.get(self.eos_token, 3)

    @classmethod
    def from_text(
        cls,
        text: str,
        min_token_freq: int = 2,
        max_domain_tokens: int = 2000,
    ) -> 'HybridChemTokenizer':
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        tokens = cls.TOKEN_PATTERN.findall(text)
        counts = Counter(tokens)

        domain_tokens = [
            token for token, freq in counts.most_common(max_domain_tokens * 2)
            if len(token) > 1 and not token.isspace() and freq >= min_token_freq
        ][:max_domain_tokens]

        chars = sorted(set(text))
        vocab = special_tokens + domain_tokens
        vocab_set = set(vocab)

        for ch in chars:
            if ch not in vocab_set:
                vocab.append(ch)
                vocab_set.add(ch)

        return cls(vocab=vocab, min_token_freq=min_token_freq, max_domain_tokens=max_domain_tokens)

    def _tokenize(self, text: str) -> List[str]:
        return self.TOKEN_PATTERN.findall(text)

    def encode(self, text: str) -> List[int]:
        result: List[int] = []
        for token in self._tokenize(text):
            token_id = self.token_to_idx.get(token)
            if token_id is not None:
                result.append(token_id)
                continue

            # fallback на char-level внутри OOV токена
            for ch in token:
                result.append(self.token_to_idx.get(ch, self.unk_id))

        return result

    def decode(self, indices: List[int]) -> str:
        pieces: List[str] = []
        for idx in indices:
            if idx == self.eos_id:
                break
            if idx in [self.pad_id, self.bos_id]:
                continue
            pieces.append(self.idx_to_token.get(idx, self.unk_token))
        return ''.join(pieces)

    def vocab_size(self) -> int:
        return len(self.vocab)

    def to_dict(self) -> Dict:
        return {
            'type': 'hybrid',
            'vocab': self.vocab,
            'min_token_freq': self.min_token_freq,
            'max_domain_tokens': self.max_domain_tokens,
        }

    @staticmethod
    def from_dict(data: Dict) -> 'HybridChemTokenizer':
        return HybridChemTokenizer(
            vocab=data['vocab'],
            min_token_freq=data.get('min_token_freq', 2),
            max_domain_tokens=data.get('max_domain_tokens', 2000),
        )


def create_tokenizer(
    text: str, 
    tokenizer_type: str = 'tiktoken',
    encoding_name: str = 'cl100k_base'
):
    """
    Фабричный метод: создаёт токенизатор нужного типа.
    
    Args:
        text: обучающий текст (используется только для char)
        tokenizer_type: 'char', 'tiktoken' или 'hybrid'
        encoding_name: для tiktoken - имя энкодера (cl100k_base, o200k_base, p50k_base)
    
    Returns:
        CharTokenizer или TikTokenizer
    """
    if tokenizer_type == 'char':
        return CharTokenizer.from_text(text)
    elif tokenizer_type == 'tiktoken':
        return TikTokenizer(encoding_name=encoding_name)
    elif tokenizer_type == 'hybrid':
        return HybridChemTokenizer.from_text(text)
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}. Use 'char', 'tiktoken' or 'hybrid'")


if __name__ == '__main__':
    # Тестирование
    sample_text = """
    Углекислый газ (CO₂) — химическое соединение углерода и кислорода.
    При нормальных условиях это бесцветный газ, без запаха и вкуса.
    Молекула CO₂ линейна и имеет формулу O=C=O.
    
    Вопрос: Что такое химическая реакция?
    Ответ: Химическая реакция — процесс превращения одних веществ в другие.
    """
    
    data_logger.info("=== CharTokenizer ===")
    char_tok = CharTokenizer.from_text(sample_text)
    data_logger.info(f"Vocab size: {char_tok.vocab_size()}")
    test_text = "CO₂ — газ"
    encoded = char_tok.encode(test_text)
    data_logger.info(f"Text: '{test_text}'")
    data_logger.info(f"Encoded ({len(encoded)} tokens): {encoded[:20]}...")
    data_logger.info(f"Decoded: {char_tok.decode(encoded)}")
    
    data_logger.info("\n=== TikTokenizer (cl100k_base) ===")
    tik_tok = TikTokenizer(encoding_name='cl100k_base')
    data_logger.info(f"Vocab size: {tik_tok.vocab_size()}")
    data_logger.info(f"Base vocab: {tik_tok.base_vocab_size()}")
    encoded_tik = tik_tok.encode(test_text)
    data_logger.info(f"Text: '{test_text}'")
    data_logger.info(f"Encoded ({len(encoded_tik)} tokens): {encoded_tik}")
    data_logger.info(f"Decoded: {tik_tok.decode(encoded_tik)}")
    
    # Сравнение длины последовательности
    long_text = "Углекислый газ используется в химической промышленности для производства различных веществ"
    data_logger.info(f"\n=== Сравнение эффективности ===")
    data_logger.info(f"Text: '{long_text}'")
    char_tokens = char_tok.encode(long_text)
    tik_tokens = tik_tok.encode(long_text)
    data_logger.info(f"Char:     {len(char_tokens):3d} токенов")
    data_logger.info(f"TikToken: {len(tik_tokens):3d} токенов")
    data_logger.info(f"Сжатие:   {len(char_tokens) / len(tik_tokens):.1f}x")
    
    # Тест special tokens
    data_logger.info(f"\n=== Special Tokens ===")
    data_logger.info(f"<|pad|> = {tik_tok.pad_id}")
    data_logger.info(f"<|bos|> = {tik_tok.bos_id}")
    data_logger.info(f"<|eos|> = {tik_tok.eos_id}")
    
    with_special = tik_tok.encode("Привет", add_bos=True, add_eos=True)
    data_logger.info(f"\nWith BOS/EOS: {with_special}")
    data_logger.info(f"Decoded (with special): {tik_tok.decode(with_special, skip_special_tokens=False)}")
    data_logger.info(f"Decoded (skip special): {tik_tok.decode(with_special, skip_special_tokens=True)}")
