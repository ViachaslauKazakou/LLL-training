"""
from logger import app_logger
Быстрый способ получить адекватную генерацию — использовать pretrained модель.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка русской GPT-2 модели (обучена на 300 GB текста!)
model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Генерация
prompt = "Python: что такое декоратор?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
app_logger.info(text)

# Результат будет НАМНОГО лучше вашей модели!
