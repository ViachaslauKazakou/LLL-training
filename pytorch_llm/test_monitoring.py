"""
Тест функций мониторинга device
"""
import torch
from config import get_device, get_memory_stats, benchmark_device

print('🔍 Проверка Device Detection:')
device = get_device()
print(f'   Detected: {device}')
print(f'   MPS available: {torch.backends.mps.is_available()}')
print(f'   CUDA available: {torch.cuda.is_available()}')
print()

print('💾 Проверка Memory Stats:')
mem_stats = get_memory_stats(device)
if mem_stats:
    if 'error' in mem_stats:
        print(f'   ⚠️ Error: {mem_stats["error"]}')
    else:
        print(f'   Reserved: {mem_stats["reserved_gb"]:.2f} GB')
        print(f'   Total: {mem_stats["total_gb"]:.1f} GB')
        print(f'   Usage: {mem_stats["usage_percent"]:.1f}%')
else:
    print('   ℹ️ Memory stats not available for CPU')
print()

print('⚡ Проверка Benchmark:')
bench = benchmark_device(device, size=256)
if bench['success']:
    print(f'   Device: {bench["device"]}')
    print(f'   Time: {bench["time_ms"]:.2f} ms')
    print(f'   Speed: {bench["gflops"]:.1f} GFLOPS')
else:
    print(f'   ❌ Error: {bench["error"]}')
print()

# Если не CPU, проверяем ускорение vs CPU
if device != "cpu":
    print('🚀 Сравнение с CPU:')
    cpu_bench = benchmark_device("cpu", size=256)
    if cpu_bench['success'] and bench['success']:
        speedup = cpu_bench['time_ms'] / bench['time_ms']
        print(f'   CPU: {cpu_bench["time_ms"]:.2f} ms')
        print(f'   {device.upper()}: {bench["time_ms"]:.2f} ms')
        print(f'   Ускорение: {speedup:.1f}x')
    print()

print('✅ Все функции работают!')
