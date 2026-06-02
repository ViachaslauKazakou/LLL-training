#!/usr/bin/env python3
"""
Конвертер PDF в текст для подготовки датасетов

Использование:
    python pdf_to_text.py input.pdf output.txt
    python pdf_to_text.py --dir pdf_folder/ --output-dir text_folder/
"""

import argparse
from pathlib import Path
from typing import Optional, Union, Dict
import sys

try:
    from pypdf import PdfReader
except ImportError:
    data_logger.error("❌ Установите pypdf: poetry add pypdf")
    sys.exit(1)

# Импорт логгера
try:
    from logger import data_logger
except ImportError:
    # Fallback если запускается из utils/
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from logger import data_logger


def convert_pdf_to_text(pdf_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, 
                        clean: bool = True) -> Dict[str, any]:
    """
    Конвертирует PDF файл в текст.
    
    Args:
        pdf_path: Путь к PDF файлу (строка или Path)
        output_path: Путь для сохранения текста (опционально, строка или Path)
        clean: Очистка текста от лишних символов
        
    Returns:
        Dict с информацией: pages, chars, words, size_kb
    """
    # Конвертируем в Path объекты
    pdf_path = Path(pdf_path)
    if output_path:
        output_path = Path(output_path)
    
    data_logger.info(f"📄 Обработка PDF: {pdf_path.name}")
    
    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        
        text_parts = []
        for i, page in enumerate(reader.pages, 1):
            # Прогресс без логирования каждой страницы
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        text = "\n\n".join(text_parts)
        
        if clean:
            # Очистка текста
            # Убираем множественные пробелы
            text = ' '.join(text.split())
            # Восстанавливаем параграфы
            text = text.replace('. ', '.\n')
            # Убираем лишние переносы
            text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        
        # Подсчитываем статистику
        words_count = len(text.split())
        size_kb = len(text.encode('utf-8')) / 1024
        
        data_logger.info(f"✅ Извлечено {len(text):,} символов из {total_pages} страниц ({size_kb:.1f} KB)")
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding='utf-8')
            data_logger.info(f"💾 Сохранено в: {output_path}")
        
        # Возвращаем метаданные и текст
        return {
            'text': text,
            'pages': total_pages,
            'chars': len(text),
            'words': words_count,
            'size_kb': size_kb
        }
        
    except Exception as e:
        data_logger.error(f"❌ Ошибка конвертации PDF {pdf_path.name}: {e}")
        return {
            'text': '',
            'pages': 0,
            'chars': 0,
            'words': 0,
            'size_kb': 0
        }


def convert_pdfs_in_directory(input_dir: Union[str, Path], output_dir: Union[str, Path], 
                               pattern: str = "*.pdf", clean: bool = True):
    """
    Конвертирует все PDF файлы в директории.
    
    Args:
        input_dir: Директория с PDF файлами (строка или Path)
        output_dir: Директория для сохранения текстовых файлов (строка или Path)
        pattern: Паттерн для поиска PDF (например, "*.pdf" или "chemistry_*.pdf")
        clean: Очистка текста
    """
    # Конвертируем в Path объекты
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    pdf_files = list(input_dir.glob(pattern))
    
    if not pdf_files:
        data_logger.warning(f"❌ PDF файлы не найдены в {input_dir} по паттерну {pattern}")
        return
    
    data_logger.info(f"📚 Найдено PDF файлов: {len(pdf_files)}")
    data_logger.info(f"📂 Выходная директория: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_chars = 0
    successful = 0
    
    for pdf_path in pdf_files:
        output_path = output_dir / f"{pdf_path.stem}.txt"
        result = convert_pdf_to_text(pdf_path, output_path, clean=clean)
        
        if result['chars'] > 0:
            total_chars += result['chars']
            successful += 1
    
    data_logger.info(f"✅ Конвертировано: {successful}/{len(pdf_files)} файлов")
    data_logger.info(f"📊 Всего символов: {total_chars:,} ({total_chars / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Конвертер PDF в текст для датасетов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Один файл
  python pdf_to_text.py book.pdf output.txt
  
  # Вся директория
  python pdf_to_text.py --dir pdfs/ --output-dir texts/
  
  # С паттерном
  python pdf_to_text.py --dir pdfs/ --output-dir texts/ --pattern "chemistry_*.pdf"
  
  # Без очистки
  python pdf_to_text.py book.pdf output.txt --no-clean
        """
    )
    
    parser.add_argument('input', nargs='?', type=str,
                        help='Входной PDF файл')
    parser.add_argument('output', nargs='?', type=str,
                        help='Выходной текстовый файл')
    parser.add_argument('--dir', type=str,
                        help='Директория с PDF файлами')
    parser.add_argument('--output-dir', type=str,
                        help='Директория для текстовых файлов')
    parser.add_argument('--pattern', type=str, default='*.pdf',
                        help='Паттерн для поиска PDF (default: *.pdf)')
    parser.add_argument('--no-clean', action='store_true',
                        help='Не очищать текст')
    
    args = parser.parse_args()
    
    # Режим директории
    if args.dir and args.output_dir:
        input_dir = Path(args.dir)
        output_dir = Path(args.output_dir)
        
        if not input_dir.exists():
            data_logger.error(f"❌ Директория не существует: {input_dir}")
            return
        
        convert_pdfs_in_directory(
            input_dir, 
            output_dir, 
            pattern=args.pattern,
            clean=not args.no_clean
        )
    
    # Режим одного файла
    elif args.input and args.output:
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if not input_path.exists():
            data_logger.error(f"❌ Файл не существует: {input_path}")
            return
        
        convert_pdf_to_text(input_path, output_path, clean=not args.no_clean)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
