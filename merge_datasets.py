import pandas as pd

# Load cleaned phishing from dolphin
phishing = pd.read_csv('synthetic_phishing_clean.csv')

# Load legitimate from original cleaned dataset  
original = pd.read_csv('synthetic_emails_clean.csv')
legit = original[original['label'] == 0]

# Combine
combined = pd.concat([phishing, legit], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Save
combined.to_csv('dataset_final.csv', index=False)

print('=== FINAL DATASET ===')
print(f'Total rows: {len(combined)}')
print(f'Phishing (1): {len(combined[combined["label"]==1])}')
print(f'Legitimate (0): {len(combined[combined["label"]==0])}')
ratio = len(combined[combined["label"]==1]) / len(combined[combined["label"]==0])
print(f'Ratio (phishing/legit): {ratio:.2f}')
print(f'\nSaved to: dataset_final.csv')

