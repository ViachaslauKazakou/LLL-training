"""
trainer.py — запуск MLX LoRA fine-tuning через subprocess.

Ответственность: только управление процессом обучения.
"""

import logging
import os
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
_MAX_LOG_LINES = 4000
_CRITICAL_MEMORY_STREAK_LIMIT = 20


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


def _append_log(training_state, msg: str) -> None:
    """Добавляет строку в буфер логов с ограничением роста памяти."""
    training_state.logs.append(msg)
    if len(training_state.logs) > _MAX_LOG_LINES:
        del training_state.logs[: len(training_state.logs) - _MAX_LOG_LINES]


def _detect_mlx_backend() -> dict:
    """Пытается определить backend MLX и факт запуска на Metal/GPU."""
    env_use_gpu = os.getenv("MLX_USE_GPU")
    if env_use_gpu == "0":
        return {
            "label": "MLX_USE_GPU=0 (CPU)",
            "using_metal_gpu": False,
        }

    try:
        import mlx.core as mx

        raw_device = str(mx.default_device())
        lowered = raw_device.lower()

        if "cpu" in lowered:
            using_metal_gpu = False
        elif "gpu" in lowered or "metal" in lowered:
            using_metal_gpu = True
        else:
            using_metal_gpu = None

        return {
            "label": f"default_device={raw_device}",
            "using_metal_gpu": using_metal_gpu,
        }
    except Exception as e:
        return {
            "label": f"backend unknown ({e})",
            "using_metal_gpu": None,
        }


def _format_iter_memory_snapshot(process: subprocess.Popen | None = None) -> tuple[str, dict]:
    """Возвращает строку и структурированные системные метрики памяти для логов Iter."""
    parts: list[str] = []
    stats: dict = {}
    child_pid = process.pid if process and process.pid else None
    uses_child_process = child_pid is not None and child_pid != os.getpid()

    try:
        import psutil

        vm = psutil.virtual_memory()
        used_gb = (vm.total - vm.available) / (1024**3)
        total_gb = vm.total / (1024**3)
        swap = psutil.swap_memory()

        # Совместимые с PyTorch-логикой ключи
        stats["reserved_gb"] = used_gb
        stats["total_gb"] = total_gb
        stats["usage_percent"] = vm.percent

        stats["ram_used_gb"] = used_gb
        stats["ram_total_gb"] = total_gb
        stats["ram_percent"] = vm.percent
        stats["swap_used_gb"] = swap.used / (1024**3)
        stats["swap_percent"] = swap.percent

        parts.append(f"Memory: {used_gb:.2f} GB / {total_gb:.2f} GB ({vm.percent:.1f}%)")
        parts.append(f"SWAP {swap.used / (1024**3):.2f} GB ({swap.percent:.1f}%)")

        if process and process.pid:
            proc = psutil.Process(process.pid)
            mem_info = proc.memory_info()
            rss = mem_info.rss / (1024**3)
            vms = mem_info.vms / (1024**3)
            stats["process_pid"] = process.pid
            stats["process_rss_gb"] = rss
            stats["process_vms_gb"] = vms
            parts.append(f"MLX pid={process.pid}")
            parts.append(f"RSS {rss:.2f} GB")
            parts.append(f"VMS {vms:.2f} GB")

            # cpu_percent без interval не блокирует; первое значение может быть 0.0.
            cpu_pct = proc.cpu_percent(interval=None)
            stats["process_cpu_percent"] = cpu_pct
            parts.append(f"CPU {cpu_pct:.1f}%")
    except Exception:
        pass

    if not uses_child_process:
        try:
            import mlx.core as mx

            # Только новые API MLX (без legacy mx.metal fallback).
            active_gb = float(mx.get_active_memory()) / (1024**3)
            peak_gb = float(mx.get_peak_memory()) / (1024**3)
            cache_gb = float(mx.get_cache_memory()) / (1024**3)
            parts.append(f"Metal active {active_gb:.2f} GB")
            parts.append(f"peak {peak_gb:.2f} GB")
            parts.append(f"cache {cache_gb:.2f} GB")
        except Exception:
            pass

    return (" | ".join(parts) if parts else "memory stats unavailable", stats)


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


def _is_metal_oom(line: str) -> bool:
    lower = line.lower()
    return "insufficient memory" in lower or "kioGPUcommandbuffercallbackerroroutofmemory".lower() in lower


def _oom_recovery_hint(config: MLXTrainingConfig) -> str:
    suggested_batch = max(1, config.batch_size // 2)
    # Keep sequence length aligned to common multiples and cap to a safer value.
    suggested_seq = max(1024, min(2048, (config.max_seq_length // 256) * 256))
    if suggested_seq >= config.max_seq_length:
        suggested_seq = max(1024, config.max_seq_length - 512)

    suggested_val_batches = max(5, min(10, config.val_batches // 2 if config.val_batches > 1 else 5))

    return (
        "Обнаружен GPU OOM (Metal). Рекомендуется снизить нагрузку и перезапустить: "
        f"batch_size {config.batch_size} -> {suggested_batch}, "
        f"max_seq_length {config.max_seq_length} -> {suggested_seq}, "
        f"val_batches {config.val_batches} -> {suggested_val_batches}, "
        f"grad_checkpoint -> {True}."
    )


def _safety_guard_error(config: MLXTrainingConfig) -> str | None:
    """Блокирует заведомо рискованные конфигурации до запуска процесса."""
    mem_pressure = int(config.batch_size) * int(config.max_seq_length)
    if mem_pressure >= 8192:
        suggested_batch = max(1, config.batch_size // 2)
        suggested_seq = max(512, config.max_seq_length - 512)
        return (
            "Конфигурация слишком тяжёлая для стабильного MLX fine-tuning "
            f"(batch_size × max_seq_length = {mem_pressure}). "
            f"Уменьшите batch_size до ~{suggested_batch} или max_seq_length до ~{suggested_seq}."
        )

    if mem_pressure >= 4096 and not config.grad_checkpoint:
        return (
            "Высокая нагрузка на память при выключенном gradient checkpointing "
            f"(batch_size × max_seq_length = {mem_pressure}). "
            "Включите Gradient checkpointing или уменьшите batch_size/max_seq_length."
        )

    return None


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


def _is_critical_memory_pressure(mem_stats: dict) -> bool:
    """Проверяет критическое давление памяти по тем же порогам, что и в UI."""
    ram_percent = float(mem_stats.get("ram_percent", 0.0) or 0.0)
    swap_used_gb = float(mem_stats.get("swap_used_gb", 0.0) or 0.0)
    swap_percent = float(mem_stats.get("swap_percent", 0.0) or 0.0)
    return (ram_percent >= 92.0) or (swap_used_gb >= 2.0) or (swap_percent >= 25.0)


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
        _append_log(training_state, msg)
        _logger.info(msg)

    def _run():
        training_state.active = True
        saw_metal_oom = False
        critical_memory_streak = 0
        guard_error = _safety_guard_error(config)
        if guard_error:
            _log(f"⛔ {guard_error}")
            training_state.metrics["error"] = f"SAFETY_GUARD: {guard_error}"
            training_state.active = False
            return

        backend = _detect_mlx_backend()
        training_state.metrics["backend"] = backend["label"]
        training_state.metrics["using_metal_gpu"] = backend["using_metal_gpu"]

        _log(f"🧭 MLX backend: {backend['label']}")
        if backend["using_metal_gpu"] is False:
            msg = "MLX работает не на Metal/GPU. Остановлено, чтобы избежать перегрузки CPU."
            _log(f"❌ {msg}")
            training_state.metrics["error"] = f"NO_METAL_GPU: {msg}"
            training_state.active = False
            return

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

                _append_log(training_state, line)
                _logger.info(line)

                if _is_metal_oom(line):
                    saw_metal_oom = True

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

                        mem_line, mem_stats = _format_iter_memory_snapshot(process)
                        training_state.metrics.update(mem_stats)
                        if _is_critical_memory_pressure(mem_stats):
                            critical_memory_streak += 1
                        else:
                            critical_memory_streak = 0

                        training_state.metrics["critical_memory_streak"] = critical_memory_streak
                        _log(f"🧠 Iter {parsed['step']}: {mem_line}")

                        if critical_memory_streak >= _CRITICAL_MEMORY_STREAK_LIMIT:
                            reason = (
                                "CRITICAL_MEMORY_PRESSURE: критическое давление памяти "
                                f"держится {critical_memory_streak} итераций подряд. "
                                "Обучение остановлено автоматически для защиты системы."
                            )
                            _log(f"🚨 {reason}")
                            training_state.metrics["error"] = reason
                            try:
                                process.terminate()
                            except Exception:
                                pass
                            break
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
                if saw_metal_oom:
                    hint = _oom_recovery_hint(config)
                    _log(f"⚠️ {hint}")
                    training_state.metrics["error"] = f"METAL_OOM: Exit code {retcode}. {hint}"
                else:
                    training_state.metrics["error"] = f"Exit code: {retcode}"

        except Exception as e:
            _log(f"❌ Ошибка запуска: {e}")
            training_state.metrics["error"] = str(e)
        finally:
            training_state.active = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def fuse_adapter(
    model_id: str,
    adapter_path: str,
    output_path: str,
    dequantize: bool = False,
) -> tuple[bool, str]:
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
    ]

    if dequantize:
        cmd.append("--dequantize")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            mode = "dequantized" if dequantize else "quantized"
            return True, f"Модель сохранена в: {output_path} ({mode})"
        else:
            error_msg = result.stderr or result.stdout or f"Exit code: {result.returncode}"
            return False, f"Ошибка слияния: {error_msg}"

    except subprocess.TimeoutExpired:
        return False, "Превышено время ожидания (5 минут)"
    except Exception as e:
        return False, f"Ошибка: {e}"
