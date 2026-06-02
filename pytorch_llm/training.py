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
import math
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
        
        # Scheduler: linear warmup + cosine decay по optimizer steps.
        self.accumulation_steps = max(1, config.gradient_accumulation_steps)
        self.total_update_steps = max(
            1,
            math.ceil(len(train_loader) / self.accumulation_steps) * config.n_epochs
        )

        def lr_lambda(current_step: int) -> float:
            warmup_steps = min(config.warmup_steps, max(self.total_update_steps - 1, 0))

            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)

            decay_steps = max(1, self.total_update_steps - warmup_steps)
            decay_progress = min(1.0, max(0.0, (current_step - warmup_steps) / decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            min_lr_ratio = 0.1
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda
        )
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.steps_without_improvement = 0  # Early stopping counter
        
        # Профилирование времени
        import time
        self.training_start_time = time.time()
        
        # Checkpoint директория
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        """Одна эпоха обучения."""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.n_epochs}")
        
        for batch_idx, (input_ids, targets) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, targets)
            loss_value = loss.item()
            total_loss += loss_value
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()

            should_step = (
                (batch_idx + 1) % self.accumulation_steps == 0
                or batch_idx == len(self.train_loader) - 1
            )

            if should_step:
                # Gradient clipping (предотвращает exploding gradients)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.model.config.grad_clip
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Logging
            if self.global_step > 0 and self.global_step % self.config.log_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                lr = self.scheduler.get_last_lr()[0]
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{perplexity:.2f}',
                    'lr': f'{lr:.2e}'
                })
            
            # Evaluation
            if should_step and self.global_step > 0 and self.global_step % self.config.eval_every == 0:
                val_loss = self.evaluate()
                
                # Вычисляем elapsed time
                import time
                elapsed = time.time() - self.training_start_time
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                
                # Текущий train loss
                current_train_loss = total_loss / (batch_idx + 1)
                
                training_logger.info(f"\n⏱️  Time: {time_str} | Step {self.global_step}: train_loss={current_train_loss:.4f}, val_loss={val_loss:.4f}")
                
                # Статистика памяти
                mem_stats = get_memory_stats(self.device)
                if mem_stats and not mem_stats.get('error'):
                    device_emoji = "🍎" if self.device == "mps" else ("🟢" if self.device == "cuda" else "💻")
                    training_logger.info(f"{device_emoji} Memory: {mem_stats['reserved_gb']:.2f} GB / {mem_stats['total_gb']:.1f} GB ({mem_stats['usage_percent']:.1f}%)")
                
                # Сохраняем лучший checkpoint
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.steps_without_improvement = 0  # Сброс счетчика!
                    best_checkpoint_name = f"{self.model_name}_best.pt"
                    self.save_checkpoint(best_checkpoint_name)
                    training_logger.info(f"✓ Сохранён лучший checkpoint (val_loss={val_loss:.4f}) → {best_checkpoint_name}")
                    training_logger.info(f"Сохранён лучший checkpoint: val_loss={val_loss:.4f} (step {self.global_step}) → {best_checkpoint_name}")
                else:
                    self.steps_without_improvement += 1
                    training_logger.warning(f"⚠️  Нет улучшения ({self.steps_without_improvement}/{self.config.patience})")
                    training_logger.info(f"Нет улучшения: {self.steps_without_improvement}/{self.config.patience}")
                    
                    # Early stopping (но только после min_epochs эпох)
                    if self.steps_without_improvement >= self.config.patience:
                        # Проверяем, прошли ли минимальное количество эпох
                        if epoch + 1 >= self.config.min_epochs:
                            training_logger.warning(f"\n⏹️  Early stopping! Модель не улучшается {self.config.patience} проверок подряд.")
                            training_logger.info(f"Early stopping на шаге {self.global_step}")
                            return total_loss / max(len(self.train_loader), 1), True  # Флаг остановки
                        else:
                            training_logger.info(f"ℹ️  Early stopping отложен: еще не завершено {self.config.min_epochs} эпох (сейчас: {epoch+1})")
                            training_logger.info(f"Early stopping отложен до эпохи {self.config.min_epochs} (сейчас: {epoch+1})")
                
                self.model.train()
            
            # Periodic checkpoint по шагам
            if should_step and self.global_step > 0 and self.global_step % self.config.save_every == 0:
                checkpoint_name = f"{self.model_name}_step_{self.global_step}.pt"
                self.save_checkpoint(checkpoint_name)
                training_logger.info(f"Сохранён periodic checkpoint: {checkpoint_name}")
        
        return total_loss / len(self.train_loader), False  # Нет early stopping
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Оценка на validation set."""
        self.model.eval()
        total_loss = 0.0
        
        for batch_idx, (input_ids, targets) in enumerate(self.val_loader):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            logits, loss = self.model(input_ids, targets)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        training_logger.info(f"✓ Evaluate завершен: avg_loss={avg_loss:.4f}")
        return avg_loss
    
    def train(self):
        """Полный цикл обучения."""
        log_session_start(training_logger, "Обучение модели")
        
        # ═══════════════════════════════════════════════════════════════
        # КОНФИГУРАЦИЯ ОБУЧЕНИЯ
        # ═══════════════════════════════════════════════════════════════
        
        # Модель
        training_logger.info(f"\n{'='*70}")
        training_logger.info(f"📦 КОНФИГУРАЦИЯ МОДЕЛИ")
        training_logger.info(f"{'='*70}")
        training_logger.info(f"Название модели:     {self.model_name}")
        training_logger.info(f"Параметров:          {self.model.count_parameters():,}")
        training_logger.info(f"Устройство:          {self.device.upper()}")
        training_logger.info("")
        
        training_logger.info(f"Название модели: {self.model_name}")
        training_logger.info(f"Устройство: {self.device}")
        training_logger.info(f"Параметров модели: {self.model.count_parameters():,}")
        
        # Архитектура модели
        cfg = self.model.config
        training_logger.info(f"🏗️  АРХИТЕКТУРА")
        training_logger.info(f"  d_model:           {cfg.d_model}")
        training_logger.info(f"  n_layers:          {cfg.n_layers}")
        training_logger.info(f"  n_heads:           {cfg.n_heads}")
        training_logger.info(f"  d_ff:              {cfg.d_ff}")
        training_logger.info(f"  context_len:       {cfg.context_len}")
        training_logger.info(f"  vocab_size:        {cfg.vocab_size}")
        training_logger.info(f"  dropout:           {cfg.dropout}")
        training_logger.info("")
        
        training_logger.info(f"Архитектура: d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}")
        training_logger.info(f"  d_ff={cfg.d_ff}, context_len={cfg.context_len}, dropout={cfg.dropout}")
        training_logger.info(f"  vocab_size={cfg.vocab_size}")
        
        # Датасет
        train_windows = len(self.train_loader.dataset)
        val_windows = len(self.val_loader.dataset)
        total_windows = train_windows + val_windows
        
        # Размер исходных данных
        try:
            data_file = Path(self.config.data_path)
            if data_file.exists():
                file_size = data_file.stat().st_size
                if data_file.suffix == '.txt':
                    text_content = data_file.read_text(encoding='utf-8')
                    text_size = len(text_content)
                    file_format = "text"
                elif data_file.suffix == '.json':
                    import json
                    with open(data_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Определяем тип JSON
                    if 'tasks' in data:
                        file_format = "json (tasks)"
                    elif 'items' in data:
                        file_format = "json (interview)"
                    elif 'User' in data and 'messages' in data:
                        file_format = "json (forum)"
                    else:
                        file_format = "json"
                    text_size = file_size  # Для JSON показываем размер файла
                else:
                    file_format = data_file.suffix
                    text_size = file_size
            else:
                file_format = "unknown"
                text_size = 0
        except:
            file_format = "unknown"
            text_size = 0
        
        training_logger.info(f"📊 ДАТАСЕТ")
        if text_size > 0:
            if text_size < 1024:
                size_str = f"{text_size} байт"
            elif text_size < 1024*1024:
                size_str = f"{text_size/1024:.1f} KB"
            else:
                size_str = f"{text_size/(1024*1024):.1f} MB"
            training_logger.info(f"  Источник:          {Path(self.config.data_path).name}")
            training_logger.info(f"  Формат:            {file_format}")
            training_logger.info(f"  Размер файла:      {size_str}")
        training_logger.info(f"  Токенизатор:       {cfg.vocab_size} уникальных токенов")
        training_logger.info(f"  Train окон:        {train_windows:,}")
        training_logger.info(f"  Val окон:          {val_windows:,}")
        training_logger.info(f"  Всего окон:        {total_windows:,}")
        training_logger.info(f"  Train/Val split:   {train_windows/total_windows*100:.1f}% / {val_windows/total_windows*100:.1f}%")
        training_logger.info("")
        
        training_logger.info(f"Датасет: train={train_windows:,} окон, val={val_windows:,} окон")
        if text_size > 0:
            training_logger.info(f"  Источник: {Path(self.config.data_path).name}, формат: {file_format}")
            if text_size < 1024*1024:
                training_logger.info(f"  Размер: {text_size/1024:.1f} KB")
            else:
                training_logger.info(f"  Размер: {text_size/(1024*1024):.1f} MB")
        training_logger.info(f"  Токенизатор: {cfg.vocab_size} токенов")
        
        # Параметры обучения
        training_logger.info(f"⚙️  ПАРАМЕТРЫ ОБУЧЕНИЯ")
        training_logger.info(f"  Эпох:              {self.config.n_epochs}")
        training_logger.info(f"  Min epochs:        {self.config.min_epochs}")
        training_logger.info(f"  Batch size:        {self.train_loader.batch_size}")
        training_logger.info(f"  Learning rate:     {self.optimizer.param_groups[0]['lr']:.2e}")
        training_logger.info(f"  Weight decay:      {cfg.weight_decay}")
        training_logger.info(f"  Grad clip:         {cfg.grad_clip}")
        training_logger.info(f"  Grad accum steps:  {self.accumulation_steps}")
        training_logger.info(f"  Warmup steps:      {self.config.warmup_steps}")
        training_logger.info(f"  Total upd steps:   {self.total_update_steps}")
        training_logger.info(f"  Eval every:        {self.config.eval_every} steps")
        training_logger.info(f"  Save every:        {self.config.save_every} steps")
        training_logger.info(f"  Early stop patience: {self.config.patience}")
        training_logger.info(f"  Min delta:         {self.config.min_delta}")
        training_logger.info(f"{'='*70}\n")
        
        training_logger.info(f"Параметры обучения: epochs={self.config.n_epochs}, min_epochs={self.config.min_epochs}")
        training_logger.info(f"  batch_size={self.train_loader.batch_size}, lr={self.optimizer.param_groups[0]['lr']:.2e}")
        training_logger.info(f"  dropout={cfg.dropout}, patience={self.config.patience}")
        
        for epoch in range(self.config.n_epochs):
            training_logger.info(f"\n📊 Завершаем эпоху {epoch+1}...")
            train_loss, early_stop = self.train_epoch(epoch)
            training_logger.info(f"✓ train_epoch завершен: loss={train_loss:.4f}")
            
            # Проверка early stopping
            if early_stop:
                training_logger.info(f"\n🏁 Обучение остановлено досрочно на эпохе {epoch+1}")
                training_logger.info(f"Обучение остановлено early stopping на эпохе {epoch+1}")
                break
            
            training_logger.info(f"📈 Вычисляем validation loss...")
            val_loss = self.evaluate()
            training_logger.info(f"✓ Validation завершен: loss={val_loss:.4f}")
            
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            
            training_logger.info(f"Epoch {epoch+1}/{self.config.n_epochs} завершена")
            training_logger.info(f"  Train: loss={train_loss:.4f}, perplexity={train_ppl:.2f}")
            training_logger.info(f"  Val:   loss={val_loss:.4f}, perplexity={val_ppl:.2f}")
            
            training_logger.info(f"\nEpoch {epoch+1} завершена:")
            training_logger.info(f"  Train: loss={train_loss:.4f}, perplexity={train_ppl:.2f}")
            training_logger.info(f"  Val:   loss={val_loss:.4f}, perplexity={val_ppl:.2f}\n")
            
            # Сохраняем checkpoint эпохи
            training_logger.info(f"💾 Сохраняем checkpoint эпохи {epoch+1}...")
            self.save_checkpoint(f"{self.model_name}_epoch_{epoch+1}.pt")
            training_logger.info(f"✓ Checkpoint сохранен")
        
        training_logger.info("Обучение завершено!")
        training_logger.info(f"Лучший val_loss: {self.best_val_loss:.4f}")
        log_session_end(training_logger, "Обучение модели")
        
        training_logger.info("\n✓ Обучение завершено!")
        training_logger.info(f"Лучший val_loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Сохранение checkpoint."""
        from datetime import datetime
        
        training_logger.info(f"💾 save_checkpoint: подготовка данных...")
        
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
        
        # Сохраняем конфигурацию токенизатора (CharTokenizer или TikTokenizer)
        if self.tokenizer is not None:
            checkpoint['tokenizer_config'] = self.tokenizer.to_dict()
        
        path = Path(self.config.checkpoint_dir) / filename
        training_logger.info(f"💾 save_checkpoint: сохранение в {path}...")
        torch.save(checkpoint, path)
        training_logger.info(f"✓ save_checkpoint: файл сохранен ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    @staticmethod
    def load_checkpoint(model: GPTModel, checkpoint_path: str, device: str = "cuda"):
        """Загрузка checkpoint для дообучения."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        training_logger.info(f"✓ Загружен checkpoint: {checkpoint_path}")
        training_logger.info(f"  Global step: {checkpoint['global_step']}")
        training_logger.info(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
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
    
    training_logger.info(f"\n✓ Checkpoint загружен: {checkpoint_path}")
    training_logger.info(f"  Продолжаем с шага {checkpoint['global_step']}")
    training_logger.info(f"  Дообучаем ещё {additional_epochs} эпох\n")
    
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
