"""
training.py — цикл обучения и оценки

Включает:
- Training loop с gradient accumulation
- Validation
- Checkpointing
- Мониторинг (loss, perplexity)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from logger import training_logger, log_session_start, log_session_end

from model import GPTModel
from config import ModelConfig, TrainingConfig, get_memory_stats


class Trainer:
    """
    Управляет процессом обучения модели.
    """
    
    def __init__(
        self,
        model: GPTModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: str = "cuda",
        tokenizer = None,
        model_name: str = "model"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.model_name = model_name.strip().replace(" ", "_")  # Очистка имени
        
        # Optimizer (AdamW с weight decay)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model.config.learning_rate,
            weight_decay=model.config.weight_decay,
            betas=(0.9, 0.95),  # как в GPT-2/3
            eps=1e-8
        )
        
        # Scheduler (cosine annealing)
        total_steps = len(train_loader) * config.n_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=model.config.learning_rate * 0.1
        )
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.steps_without_improvement = 0  # Early stopping counter
        
        # Checkpoint директория
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        """Одна эпоха обучения."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.n_epochs}")
        
        for batch_idx, (input_ids, targets) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (предотвращает exploding gradients)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.model.config.grad_clip
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                lr = self.scheduler.get_last_lr()[0]
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.2f}',
                    'lr': f'{lr:.2e}'
                })
            
            # Evaluation
            if self.global_step % self.config.eval_every == 0:
                val_loss = self.evaluate()
                print(f"\nStep {self.global_step}: val_loss={val_loss:.4f}")
                
                # Статистика памяти
                mem_stats = get_memory_stats(self.device)
                if mem_stats and not mem_stats.get('error'):
                    device_emoji = "🍎" if self.device == "mps" else ("🟢" if self.device == "cuda" else "💻")
                    print(f"{device_emoji} Memory: {mem_stats['reserved_gb']:.2f} GB / {mem_stats['total_gb']:.1f} GB ({mem_stats['usage_percent']:.1f}%)")
                
                # Сохраняем лучший checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.steps_without_improvement = 0  # Сброс счетчика!
                    best_checkpoint_name = f"{self.model_name}_best.pt"
                    self.save_checkpoint(best_checkpoint_name)
                    print(f"✓ Сохранён лучший checkpoint (val_loss={val_loss:.4f}) → {best_checkpoint_name}")
                    training_logger.info(f"Сохранён лучший checkpoint: val_loss={val_loss:.4f} (step {self.global_step}) → {best_checkpoint_name}")
                else:
                    self.steps_without_improvement += 1
                    print(f"⚠️  Нет улучшения ({self.steps_without_improvement}/{self.config.patience})")
                    training_logger.info(f"Нет улучшения: {self.steps_without_improvement}/{self.config.patience}")
                    
                    # Early stopping
                    if self.steps_without_improvement >= self.config.patience:
                        print(f"\n⏹️  Early stopping! Модель не улучшается {self.config.patience} проверок подряд.")
                        training_logger.info(f"Early stopping на шаге {self.global_step}")
                        return total_loss / max(len(self.train_loader), 1), True  # Флаг остановки
                
                self.model.train()
            
            # Periodic checkpoint
            if self.global_step % self.config.save_every == 0:
                checkpoint_name = f"{self.model_name}_step_{self.global_step}.pt"
                self.save_checkpoint(checkpoint_name)
                training_logger.info(f"Сохранён periodic checkpoint: {checkpoint_name}")
        
        return total_loss / len(self.train_loader), False  # Нет early stopping
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Оценка на validation set."""
        self.model.eval()
        total_loss = 0.0
        
        for input_ids, targets in self.val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            logits, loss = self.model(input_ids, targets)
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Полный цикл обучения."""
        log_session_start(training_logger, "Обучение модели")
        training_logger.info(f"Устройство: {self.device}")
        training_logger.info(f"Параметров модели: {self.model.count_parameters():,}")
        training_logger.info(f"Эпох: {self.config.n_epochs}")
        training_logger.info(f"Batch size: {self.train_loader.batch_size}")
        training_logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        
        print(f"\n{'='*60}")
        print(f"Начинаем обучение на {self.device}")
        print(f"Модель: {self.model.count_parameters():,} параметров")
        print(f"Эпох: {self.config.n_epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.n_epochs):
            train_loss, early_stop = self.train_epoch(epoch)
            
            # Проверка early stopping
            if early_stop:
                print(f"\n🏁 Обучение остановлено досрочно на эпохе {epoch+1}")
                training_logger.info(f"Обучение остановлено early stopping на эпохе {epoch+1}")
                break
            
            val_loss = self.evaluate()
            
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            
            training_logger.info(f"Epoch {epoch+1}/{self.config.n_epochs} завершена")
            training_logger.info(f"  Train: loss={train_loss:.4f}, perplexity={train_ppl:.2f}")
            training_logger.info(f"  Val:   loss={val_loss:.4f}, perplexity={val_ppl:.2f}")
            
            print(f"\nEpoch {epoch+1} завершена:")
            print(f"  Train: loss={train_loss:.4f}, perplexity={train_ppl:.2f}")
            print(f"  Val:   loss={val_loss:.4f}, perplexity={val_ppl:.2f}\n")
            
            # Сохраняем checkpoint эпохи
            self.save_checkpoint(f"{self.model_name}_epoch_{epoch+1}.pt")
        
        training_logger.info("Обучение завершено!")
        training_logger.info(f"Лучший val_loss: {self.best_val_loss:.4f}")
        log_session_end(training_logger, "Обучение модели")
        
        print("\n✓ Обучение завершено!")
        print(f"Лучший val_loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Сохранение checkpoint."""
        from datetime import datetime
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.model.config,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            # Metadata для версионирования
            'model_name': self.model_name,
            'dataset_path': str(self.config.data_path),
            'training_date': datetime.now().isoformat(),
        }
        
        # Сохраняем vocab токенизатора, если есть
        if self.tokenizer is not None:
            checkpoint['vocab'] = self.tokenizer.vocab
        
        path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, path)
    
    @staticmethod
    def load_checkpoint(model: GPTModel, checkpoint_path: str, device: str = "cuda"):
        """Загрузка checkpoint для дообучения."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Загружен checkpoint: {checkpoint_path}")
        print(f"  Global step: {checkpoint['global_step']}")
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
        return checkpoint


def continue_training(
    checkpoint_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    additional_epochs: int = 5,
    device: str = "cuda",
    tokenizer = None,
    model_name: str = None
):
    """
    Продолжить обучение с checkpoint.
    
    Использование:
        continue_training(
            "checkpoints/best_model.pt",
            train_loader,
            val_loader,
            config,
            additional_epochs=10,
            tokenizer=tokenizer,
            model_name="my_model_v2"
        )
    """
    # Загружаем checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['config']
    
    # Если model_name не указан, берем из checkpoint или используем дефолтное
    if model_name is None:
        model_name = checkpoint.get('model_name', 'model_continued')
    
    # Создаём модель и загружаем веса
    model = GPTModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✓ Checkpoint загружен: {checkpoint_path}")
    print(f"  Продолжаем с шага {checkpoint['global_step']}")
    print(f"  Дообучаем ещё {additional_epochs} эпох\n")
    
    # Создаём trainer
    config.n_epochs = additional_epochs
    trainer = Trainer(model, train_loader, val_loader, config, device, tokenizer=tokenizer, model_name=model_name)
    
    # Восстанавливаем optimizer и scheduler (опционально)
    if 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    trainer.global_step = checkpoint['global_step']
    trainer.best_val_loss = checkpoint['best_val_loss']
    
    # Запускаем обучение
    trainer.train()
