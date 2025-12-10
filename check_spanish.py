import pandas as pd

df = pd.read_csv('synthetic_phishing_dolphin.csv')

SPANISH_INDICATORS = [
    'estimado', 'estimada', 'atentamente', 'saludos', 'cordiales',
    'por favor', 'gracias', 'adjunto', 'factura', 'cuenta',
    'empresa', 'urgente', 'importante', 'señor', 'señora'
]

def count_spanish(text):
    return sum(1 for ind in SPANISH_INDICATORS if ind in text.lower())

df['spanish_count'] = df['email_text'].apply(count_spanish)
low_spanish = df[df['spanish_count'] < 2]

print(f'Low Spanish indicator count (<2): {len(low_spanish)}')
print('\nExamples of "not Spanish" emails:')
for i, row in low_spanish.head(5).iterrows():
    print(f'\n[{row["spanish_count"]} indicators]')
    print(row['email_text'][:200] + '...')

