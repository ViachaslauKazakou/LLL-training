"""Analyze forum JSON data structure."""
import json
from pathlib import Path

p = Path('data/parser_2039898_20260521T071908.json')
print(f'File size: {p.stat().st_size / 1024:.1f} KB')

with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'\nUser: {data["User"]}')
print(f'Topics: {len(data["messages"])}')

total_messages = sum(len(msgs) for msgs in data['messages'].values())
print(f'Total messages: {total_messages}')

# Show topics
print('\nSample topics:')
for i, (topic, msgs) in enumerate(list(data['messages'].items())[:3]):
    print(f'  {i+1}. "{topic[:70]}..." ({len(msgs)} msgs)')
    if msgs:
        print(f'     First message: "{msgs[0][:100]}..."')

# Calculate total text
total_chars = sum(len(msg) for msgs in data['messages'].values() for msg in msgs)
print(f'\nTotal characters: {total_chars:,} (~{total_chars / 1024:.1f} KB)')
