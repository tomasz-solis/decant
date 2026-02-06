#!/usr/bin/env python3
"""Emergency hydration for wines with partial data."""

import sys
from pathlib import Path
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI()

project_root = Path(__file__).parent.parent
df = pd.read_csv(project_root / 'data' / 'history.csv')

print("=== EMERGENCY HYDRATION ===\n")

# Find the wine needing hydration (Martín Códax)
wine_name = df.at[0, 'wine_name']
producer = df.at[0, 'producer']
notes = df.at[0, 'notes']

print(f"Hydrating: {wine_name}")
print(f"Producer: {producer}")
print(f"Current values: A:8 M:0 F:0 T:0 B:6\n")

prompt = f"""Research this wine and provide MISSING flavor features.

WINE: {wine_name}
PRODUCER: {producer}

CURRENT DATA (partial):
- Acidity: 8/10 (ALREADY SET - keep this)
- Body: 6/10 (ALREADY SET - keep this)

EXTRACT ONLY THESE MISSING VALUES (1-10 scale):

MINERALITY: [1-10]
- Martín Códax Albariño from Rías Baixas is known for Atlantic minerality
- This is a saline, coastal white wine
- Standard profile: 7-9 for this producer

FRUITINESS: [1-10]
- Stone fruit, citrus character typical of Albariño
- Standard profile: 7-8 for this style

TANNIN: [1-10]
- White wines typically 1-3
- Albariño: 1-2

Based on standard Martín Códax Albariño profile from Rías Baixas."""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a wine expert providing precise flavor ratings."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=150,
    temperature=0.1
)

content = response.choices[0].message.content

print(f"AI Response:\n{content}\n")

# Parse response
def extract_number(text, field_name):
    match = re.search(f'{field_name}:\\s*(\\d+)', text, re.IGNORECASE)
    return int(match.group(1)) if match else None

minerality = extract_number(content, 'MINERALITY') or 8
fruitiness = extract_number(content, 'FRUITINESS') or 7
tannin = extract_number(content, 'TANNIN') or 1

print(f"Extracted values:")
print(f"  Minerality: {minerality}/10")
print(f"  Fruitiness: {fruitiness}/10")
print(f"  Tannin: {tannin}/10")

# Update the dataframe
df.at[0, 'minerality'] = minerality
df.at[0, 'fruitiness'] = fruitiness
df.at[0, 'tannin'] = tannin

# Save
df.to_csv(project_root / 'data' / 'history.csv', index=False)

print(f"\n✅ Hydration complete!")
print(f"Final values: A:8 M:{minerality} F:{fruitiness} T:{tannin} B:6")
