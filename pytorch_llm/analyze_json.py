"""Analyze forum JSON data structure."""
import json
from pathlib import Path
from logger import data_logger

p = Path('data/parser_2039898_20260521T071908.json')
data_logger.info(f'File size: {p.stat().st_size / 1024:.1f} KB')

with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)

data_logger.info(f'\nUser: {data["User"]}')
data_logger.info(f'Topics: {len(data["messages"])}')

total_messages = sum(len(msgs) for msgs in data['messages'].values())
data_logger.info(f'Total messages: {total_messages}')

# Show topics
data_logger.info('\nSample topics:')
for i, (topic, msgs) in enumerate(list(data['messages'].items())[:3]):
    data_logger.info(f'  {i+1}. "{topic[:70]}..." ({len(msgs)} msgs)')
    if msgs:
        data_logger.info(f'     First message: "{msgs[0][:100]}..."')

# Calculate total text
total_chars = sum(len(msg) for msgs in data['messages'].values() for msg in msgs)
data_logger.info(f'\nTotal characters: {total_chars:,} (~{total_chars / 1024:.1f} KB)')
