#!/usr/bin/env python3
"""Инспектор checkpoint файла chemistry_model_best.pt"""
import torch
from pathlib import Path

checkpoint_path = "checkpoints/chemistry_model_best.pt"

print("="*70)
print("🔍 ПАРАМЕТРЫ CHECKPOINT")
print("="*70)

# Загружаем checkpoint
cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

file_size = Path(checkpoint_path).stat().st_size / 1024 / 1024
print(f"\n📦 Файл: {checkpoint_path}")
print(f"💾 Размер: {file_size:.2f} MB")

print("\n📋 Содержимое checkpoint:")
print("-" * 70)

for key in cp.keys():
    value = cp[key]
    if isinstance(value, dict):
        print(f"  {key}: dict с {len(value)} элементами")
        if key == "model_state_dict":
            total_params = sum(p.numel() for p in value.values())
            print(f"    └─ Всего параметров: {total_params:,}")
    elif hasattr(value, "__class__"):
        print(f"  {key}: {value.__class__.__name__}")
    else:
        print(f"  {key}: {value}")

print("\n" + "="*70)
print("⚙️ КОНФИГУРАЦИЯ МОДЕЛИ")
print("="*70)

config = cp.get("config", {})
if hasattr(config, "__dict__"):
    config_dict = vars(config)
elif isinstance(config, dict):
    config_dict = config
else:
    config_dict = {}

for k, v in config_dict.items():
    print(f"  {k}: {v}")

print("\n" + "="*70)
print("📊 МЕТРИКИ ОБУЧЕНИЯ")
print("="*70)

print(f"  epoch: {cp.get('epoch', 'N/A')}")
print(f"  train_loss: {cp.get('train_loss', 'N/A'):.4f}" if 'train_loss' in cp else "  train_loss: N/A")
print(f"  val_loss: {cp.get('val_loss', 'N/A'):.4f}" if 'val_loss' in cp else "  val_loss: N/A")
print(f"  best_val_loss: {cp.get('best_val_loss', 'N/A'):.4f}" if 'best_val_loss' in cp else "  best_val_loss: N/A")

if "tokenizer_config" in cp:
    print("\n" + "="*70)
    print("🔤 ТОКЕНИЗАТОР")
    print("="*70)
    tok_config = cp["tokenizer_config"]
    print(f"  type: {tok_config.get('type', 'N/A')}")
    if "vocab_size" in tok_config:
        print(f"  vocab_size: {tok_config['vocab_size']}")
    elif "vocab" in tok_config:
        if isinstance(tok_config["vocab"], dict):
            print(f"  vocab_size: {len(tok_config['vocab'])}")
        elif isinstance(tok_config["vocab"], list):
            print(f"  vocab_size: {len(tok_config['vocab'])}")

print("\n" + "="*70)
print("🧠 АРХИТЕКТУРА (параметры модели)")
print("="*70)

state_dict = cp.get("model_state_dict", {})
if state_dict:
    # Группируем слои
    embeddings = sum(p.numel() for k, p in state_dict.items() if "embed" in k or "wte" in k or "wpe" in k)
    attention = sum(p.numel() for k, p in state_dict.items() if "attn" in k)
    ffn = sum(p.numel() for k, p in state_dict.items() if "mlp" in k or "ffn" in k)
    layernorm = sum(p.numel() for k, p in state_dict.items() if "ln" in k or "norm" in k)
    lm_head = sum(p.numel() for k, p in state_dict.items() if "lm_head" in k)
    
    total = sum(p.numel() for p in state_dict.values())
    
    print(f"  📐 Всего параметров: {total:,}")
    print(f"  ├─ Embeddings: {embeddings:,} ({embeddings/total*100:.1f}%)")
    print(f"  ├─ Attention: {attention:,} ({attention/total*100:.1f}%)")
    print(f"  ├─ Feed-Forward: {ffn:,} ({ffn/total*100:.1f}%)")
    print(f"  ├─ LayerNorm: {layernorm:,} ({layernorm/total*100:.1f}%)")
    print(f"  └─ LM Head: {lm_head:,} ({lm_head/total*100:.1f}%)")

print("\n" + "="*70)
