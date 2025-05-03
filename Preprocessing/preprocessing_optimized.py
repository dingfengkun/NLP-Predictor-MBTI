import pandas as pd
import re
import time
import multiprocessing as mp

# --- Global Constants ---
stop_words = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','any','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now'
}

mbti_types = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP'
]
mbti_pattern = r'\b(' + '|'.join(mbti_types) + r')\b'

# --- Cleaning Function ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# --- Multiprocessing Wrapper ---
def parallel_apply(series, func, num_processes=None):
    with mp.Pool(num_processes or mp.cpu_count()) as pool:
        return pool.map(func, series)

# --- MBTI Extractor ---
def extract_mbti(text):
    match = re.findall(mbti_pattern, str(text).upper())
    return match[0] if match else None

# --- Main Entry ---
if __name__ == "__main__":
    total_start = time.time()

    # Step 1: Load
    start = time.time()
    df = pd.read_csv("mbti_full_pull.csv")
    print(f"✅ Loaded dataset: {len(df)} rows in {time.time() - start:.2f}s")

    # Step 2: Extract MBTI
    start = time.time()
    df['MBTI'] = df['author_flair_text'].apply(extract_mbti)
    before = len(df)
    df = df.dropna(subset=['MBTI'])
    print(f"✅ Extracted MBTI. Dropped {before - len(df)} rows in {time.time() - start:.2f}s")

    # Step 3: Remove MBTI mentions in body
    start = time.time()
    df['body'] = df['body'].astype(str)
    before = len(df)
    df = df[~df['body'].str.upper().str.contains(mbti_pattern)]
    print(f"✅ Dropped {before - len(df)} posts mentioning MBTI in {time.time() - start:.2f}s")

    # Step 4: Remove empty body
    start = time.time()
    before = len(df)
    df = df[df['body'].str.strip() != '']
    print(f"✅ Removed {before - len(df)} empty posts in {time.time() - start:.2f}s")

    # Step 5: Drop raw duplicates
    start = time.time()
    before = len(df)
    df = df.drop_duplicates(subset='body')
    print(f"✅ Removed {before - len(df)} duplicate body posts in {time.time() - start:.2f}s")

    # Step 6: Clean & normalize in parallel
    start = time.time()
    print("🧼 Cleaning text with multiprocessing...")
    df['POST'] = parallel_apply(df['body'], clean_text)
    print(f"✅ Cleaned and normalized text in {time.time() - start:.2f}s")

    # Step 7: Filter by word count
    start = time.time()
    df['clean_word_count'] = df['POST'].apply(lambda x: len(x.split()))
    before = len(df)
    df = df[(df['clean_word_count'] >= 10) & (df['clean_word_count'] <= 1000)]
    print(f"✅ Removed {before - len(df)} posts outside [10, 1000] words in {time.time() - start:.2f}s")

    # Step 8: Drop cleaned duplicates
    start = time.time()
    before = len(df)
    df = df.sort_values(by='POST').drop_duplicates(subset='POST')
    print(f"✅ Removed {before - len(df)} duplicate cleaned posts in {time.time() - start:.2f}s")

    # Step 9: Save final cleaned data
    df_cleaned = df[['MBTI', 'POST']].reset_index(drop=True)
    df_cleaned.to_csv("cleaned_mbti_parallel.csv", index=False)
    print(f"\n📁 Saved to cleaned_mbti_parallel.csv")
    print(f"📊 Final cleaned dataset has {len(df_cleaned)} rows")
    print(f"⏱️ Total preprocessing time: {time.time() - total_start:.2f}s")