"""
trainer.py — запуск MLX LoRA fine-tuning через subprocess.

Ответственность: только управление процессом обучения.
"""

import logging
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Используем training_logger если он уже настроен в logger.py, иначе стандартный
_logger = logging.getLogger("pytorch_llm.training")


@dataclass
class MLXTrainingConfig:
    model_id: str
    adapter_path: str           # куда сохранять адаптеры
    data_dir: str               # директория с train.jsonl + valid.jsonl
    # LoRA
    lora_layers: int = 8
    lora_rank: int = 8
    lora_scale: float = 20.0
    lora_dropout: float = 0.0
    # Training
    learning_rate: float = 2e-5
    batch_size: int = 4
    iters: int = 500
    max_seq_length: int = 1024
    val_batches: int = 25
    steps_per_eval: int = 100
    save_every: int = 100
    grad_checkpoint: bool = False
    resume_adapter_file: str | None = None  # путь к .safetensors для продолжения


# Regex паттерны для парсинга stdout mlx-lm
_RE_TRAIN = re.compile(
    r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\d.e+-]+),\s+It/sec\s+([\d.]+)"
)
_RE_VAL = re.compile(
    r"Val loss\s+([\d.]+)"
)
_RE_SAVED = re.compile(
    r"Saved adapter weights to (.+)"
)


def _parse_mlx_line(line: str) -> dict | None:
    """Парсит строку stdout mlx-lm и возвращает dict с метриками или None."""
    m = _RE_TRAIN.search(line)
    if m:
        return {
            "type": "train",
            "step": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "learning_rate": float(m.group(3)),
            "it_per_sec": float(m.group(4)),
        }

    m = _RE_VAL.search(line)
    if m:
        return {
            "type": "val",
            "val_loss": float(m.group(1)),
        }

    m = _RE_SAVED.search(line)
    if m:
        return {
            "type": "saved",
            "path": m.group(1).strip(),
        }

    return None


def run_mlx_training(config: MLXTrainingConfig, training_state) -> None:
    """
    Запускает MLX LoRA fine-tuning в фоновом потоке.

    Запускает mlx_lm.lora как subprocess, парсит stdout для прогресса,
    обновляет training_state.logs и training_state.metrics.
    Устанавливает training_state.active = False по завершению.
    """
    adapter_path = Path(config.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Write LoRA params to a temp config YAML (--lora-parameters is not a valid CLI arg)
    lora_config = {
        "lora_parameters": {
            "rank": config.lora_rank,
            "scale": config.lora_scale,
            "dropout": config.lora_dropout,
        }
    }
    config_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=adapter_path
    )
    yaml.dump(lora_config, config_file)
    config_file.flush()
    config_file.close()

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", config.model_id,
        "--train",
        "--data", config.data_dir,
        "--adapter-path", config.adapter_path,
        "--num-layers", str(config.lora_layers),
        "-c", config_file.name,
        "--learning-rate", str(config.learning_rate),
        "--batch-size", str(config.batch_size),
        "--iters", str(config.iters),
        "--max-seq-length", str(config.max_seq_length),
        "--val-batches", str(config.val_batches),
        "--steps-per-eval", str(config.steps_per_eval),
        "--save-every", str(config.save_every),
    ]

    if config.grad_checkpoint:
        cmd.append("--grad-checkpoint")

    if config.resume_adapter_file:
        cmd += ["--resume-adapter-file", config.resume_adapter_file]

    def _log(msg: str):
        training_state.logs.append(msg)
        _logger.info(msg)

    def _run():
        training_state.active = True
        _log("Запуск MLX LoRA fine-tuning...")
        _log(f"Модель: {config.model_id}")
        _log(f"Адаптер: {config.adapter_path}")
        _log(f"Данные: {config.data_dir}")
        _log(f"Итераций: {config.iters}, batch={config.batch_size}, lr={config.learning_rate}")
        _log("─" * 50)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(config.adapter_path).parent.parent),
            )

            # Сохраняем PID для возможной отмены
            if hasattr(training_state, "process"):
                training_state.process = process

            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue

                training_state.logs.append(line)
                _logger.info(line)

                # Парсим метрики
                parsed = _parse_mlx_line(line)
                if parsed:
                    if parsed["type"] == "train":
                        training_state.metrics["step"] = parsed["step"]
                        training_state.metrics["train_loss"] = parsed["train_loss"]
                        training_state.metrics["it_per_sec"] = parsed.get("it_per_sec", 0)
                        # Прогресс в процентах
                        if config.iters > 0:
                            training_state.metrics["progress"] = parsed["step"] / config.iters
                    elif parsed["type"] == "val":
                        training_state.metrics["val_loss"] = parsed["val_loss"]
                    elif parsed["type"] == "saved":
                        training_state.metrics["last_saved"] = parsed["path"]

            process.wait()
            retcode = process.returncode

            if retcode == 0:
                _log("─" * 50)
                _log("✅ Обучение завершено успешно!")
                _log(f"Адаптеры сохранены в: {config.adapter_path}")
                training_state.metrics["finished"] = True
                training_state.metrics["progress"] = 1.0
            else:
                _log(f"❌ Процесс завершился с кодом {retcode}")
                training_state.metrics["error"] = f"Exit code: {retcode}"

        except Exception as e:
            _log(f"❌ Ошибка запуска: {e}")
            training_state.metrics["error"] = str(e)
        finally:
            training_state.active = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def fuse_adapter(model_id: str, adapter_path: str, output_path: str) -> tuple[bool, str]:
    """
    Сливает LoRA адаптер с базовой моделью через mlx_lm.fuse.

    Returns:
        (success, message)
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", model_id,
        "--adapter-path", adapter_path,
        "--save-path", output_path,
        "--dequantize",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            return True, f"Модель сохранена в: {output_path}"
        else:
            error_msg = result.stderr or result.stdout or f"Exit code: {result.returncode}"
            return False, f"Ошибка слияния: {error_msg}"

    except subprocess.TimeoutExpired:
        return False, "Превышено время ожидания (5 минут)"
    except Exception as e:
        return False, f"Ошибка: {e}"
