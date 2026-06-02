"""
model.py — архитектура GPT-подобного трансформера

Стек из N идентичных блоков:
    LayerNorm → Multi-Head Attention → Residual
    LayerNorm → Feed-Forward → Residual
"""

import torch
import torch.nn as nn

from attention import MultiHeadAttention, create_causal_mask
from config import ModelConfig
from logger import data_logger


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) — 2-слойный MLP.
    
    Архитектура:
        Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model) → Dropout
    
    Обычно d_ff = 4 × d_model для достаточной экспрессивности.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Один блок трансформера.
    
    Структура:
        x → LayerNorm → MultiHeadAttention → Dropout → Residual → 
        → LayerNorm → FeedForward → Dropout → Residual → выход
    
    Residual connections помогают градиентам течь через глубокую сеть.
    Dropout после каждого sub-layer предотвращает overfitting.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # LayerNorm до attention (Pre-LN архитектура — более стабильная)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(
            config.d_model, 
            config.n_heads, 
            config.dropout
        )
        self.dropout1 = nn.Dropout(config.dropout)  # После attention
        
        # LayerNorm до feed-forward
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)  # После feed-forward
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        mask: (batch, 1, seq_len, seq_len)
        """
        # Attention → Dropout → Residual
        x = x + self.dropout1(self.attn(self.ln1(x), mask))
        
        # Feed-forward → Dropout → Residual  
        x = x + self.dropout2(self.ff(self.ln2(x)))
        
        return x


class GPTModel(nn.Module):
    """
    Авторегрессионный трансформер (GPT-подобная архитектура).
    
    Архитектура:
        Token Embedding + Positional Embedding
        ↓
        N × TransformerBlock
        ↓
        LayerNorm
        ↓
        Linear (d_model → vocab_size)
        ↓
        Softmax (в loss function)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Эмбеддинги
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Стек трансформерных блоков
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Финальная layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Output projection (без bias для экономии памяти)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying — эмбеддинги и output слой делят веса (экономия параметров)
        self.lm_head.weight = self.token_embedding.weight
        
        # Инициализация весов
        self.apply(self._init_weights)
        
        data_logger.info(f"Модель создана: {self.count_parameters():,} параметров")
    
    def _init_weights(self, module: nn.Module):
        """Инициализация весов по GPT-2."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        input_ids: (batch, seq_len) — индексы токенов
        targets: (batch, seq_len) — target токены для loss (опционально)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar или None
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Token embeddings  →  (batch, seq_len, d_model)
        tok_emb = self.token_embedding(input_ids)
        
        # 2. Positional embeddings  →  (seq_len, d_model)
        positions = torch.arange(0, seq_len, device=device)
        pos_emb = self.position_embedding(positions)
        
        # 3. Суммируем + dropout
        x = self.dropout(tok_emb + pos_emb)  # (batch, seq_len, d_model)
        
        # 4. Causal mask — токен не видит будущее
        mask = create_causal_mask(seq_len, device)
        
        # 5. Пропускаем через все блоки трансформера
        for block in self.blocks:
            x = block(x, mask)
        
        # 6. Финальная layer norm
        x = self.ln_f(x)
        
        # 7. Проекция в словарь  →  (batch, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        # 8. Считаем loss если есть targets
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_token_id
            )
        
        return logits, loss
    
    def count_parameters(self) -> int:
        """Подсчёт всех обучаемых параметров."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None
    ) -> torch.Tensor:
        """
        Авторегрессионная генерация.
        
        input_ids: (batch, seq_len) — начальная последовательность
        max_new_tokens: сколько новых токенов сгенерировать
        temperature: > 1 = более случайно, < 1 = более детерминированно
        top_k: если задано, выбирать только из top-k токенов
        
        Returns: (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Обрезаем если длина превышает context_len
            input_ids_cond = input_ids if input_ids.size(1) <= self.config.context_len \
                             else input_ids[:, -self.config.context_len:]
            
            # Forward pass
            logits, _ = self(input_ids_cond)
            
            # Берём logits последнего токена  →  (batch, vocab_size)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Softmax → вероятности
            probs = nn.functional.softmax(logits, dim=-1)
            
            # Сэмплируем следующий токен
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Добавляем к последовательности
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
