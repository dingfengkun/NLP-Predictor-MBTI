import time
import re
import html
import pandas as pd
import nltk
from nltk.corpus import stopwords

# --- Setup ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

mbti_types = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP'
]
mbti_pattern = r'\b(' + '|'.join(mbti_types) + r')\b'

# --- Text preprocessing function ---
def clean_text(text):
    text = html.unescape(str(text))
    text = text.replace('\n', ' ').replace('\\n', ' ')
    text = text.lower()
    text = text.replace('_', ' ')
    text = re.sub(r'(.)\1{2,}', r'\1', text)        
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
    

# --- Start full timer ---
total_start = time.time()

# --- Step 1: Load dataset ---
start = time.time()
df = pd.read_csv('mbti_full_pull.csv')
print(f"✅ Loaded dataset: {len(df)} rows in {time.time() - start:.2f}s")

# --- Step 2: Extract MBTI ---
start = time.time()
def extract_mbti(text):
    match = re.findall(mbti_pattern, str(text).upper())
    return match[0] if match else None

df['MBTI'] = df['author_flair_text'].apply(extract_mbti)
before = len(df)
df = df.dropna(subset=['MBTI'])
print(f"✅ Extracted MBTI. Dropped {before - len(df)} rows in {time.time() - start:.2f}s")

# --- Step 3: Remove posts mentioning MBTI in raw body ---
start = time.time()
df['body'] = df['body'].astype(str)
before = len(df)
df = df[~df['body'].str.upper().str.contains(mbti_pattern)]
print(f"✅ Dropped {before - len(df)} posts mentioning MBTI in {time.time() - start:.2f}s")

# --- Step 4: Remove empty posts ---
start = time.time()
before = len(df)
df = df[df['body'].str.strip() != '']
print(f"✅ Removed {before - len(df)} empty body posts in {time.time() - start:.2f}s")

# --- Step 5: Remove duplicate raw body posts ---
start = time.time()
before = len(df)
df = df.drop_duplicates(subset='body')
print(f"✅ Removed {before - len(df)} duplicate body posts in {time.time() - start:.2f}s")

# --- Step 6: Clean and normalize text ---
start = time.time()
df['POST'] = df['body'].apply(clean_text)
print(f"✅ Cleaned and normalized text in {time.time() - start:.2f}s")

# --- Step 7: Filter by word count ---
start = time.time()
df['clean_word_count'] = df['POST'].apply(lambda x: len(x.split()))
before = len(df)
df = df[(df['clean_word_count'] >= 10) & (df['clean_word_count'] <= 1000)]
print(f"✅ Removed {before - len(df)} posts outside [10, 1000] words in {time.time() - start:.2f}s")

# --- Step 8: Drop duplicate cleaned POSTs ---
start = time.time()
before = len(df)
df = df.sort_values(by='POST')
df = df.drop_duplicates(subset='POST')
print(f"✅ Removed {before - len(df)} duplicate cleaned POSTs in {time.time() - start:.2f}s")

# --- Final output ---
df_cleaned = df[['MBTI', 'POST']].reset_index(drop=True)
#df_cleaned.to_csv("cleaned_mbti_basic.csv", index=False)
print(f"\n📁 Saved to cleaned_mbti_basic.csv")
print(f"📊 Final cleaned dataset has {len(df_cleaned)} rows")
print(f"⏱️ Total preprocessing time: {time.time() - total_start:.2f}s")
