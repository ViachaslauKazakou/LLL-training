"""
attention.py — Multi-Head Self-Attention

Ядро трансформера. Каждая "голова" attention учится смотреть 
на разные аспекты контекста.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    
    Параметры
    ---------
    d_model : размерность входа
    n_heads : количество голов attention
    dropout : dropout rate
    
    Формула:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
        
    где d_k = d_model // n_heads
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model должен делиться на n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # размерность каждой головы
        
        # Проекции для Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Финальная проекция после конкатенации голов
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        mask: (batch, 1, seq_len, seq_len) — causal mask
        
        Returns: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Проекции Q, K, V  →  (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. Разбиваем на n_heads  →  (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (batch, n_heads, seq_len, seq_len)
        
        # 4. Применяем causal mask (если есть)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax  →  веса attention
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # 6. Взвешенная сумма Value-векторов
        out = attn_weights @ V  # (batch, n_heads, seq_len, d_k)
        
        # 7. Конкатенируем головы  →  (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 8. Финальная проекция
        out = self.W_o(out)
        
        return out


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Создаёт causal mask — нижнетреугольная матрица.
    Токен может смотреть только на себя и предыдущие токены.
    
    Returns: (1, 1, seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = (mask == 0).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    return mask
