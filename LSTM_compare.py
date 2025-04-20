import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import time
from tqdm import tqdm

# Dataset 定义
class MBTIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        sequence = self.tokenizer.texts_to_sequences([str(self.texts[idx])])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')[0]
        return {
            'text': torch.tensor(padded, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

# Baseline LSTM 模型
class MBTIBaselineLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))

# 参数设置
MAX_WORDS = 10000
MAX_LEN = 500
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
OUTPUT_DIM = 16
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3

# 数据加载与预处理
print("Start!")
df = pd.read_csv("MBTI 500.csv")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df["type"].values)
texts = df["posts"].values

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# 保存 LabelEncoder 和 Tokenizer（可选）
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# 数据划分
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_dataset = MBTIDataset(X_train, y_train, tokenizer, MAX_LEN)
val_dataset = MBTIDataset(X_val, y_val, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = MBTIBaselineLSTM(MAX_WORDS, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# 训练过程
print("🔁 开始训练...")
start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0, 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x = batch["text"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (logits.argmax(1) == y).sum().item()

    acc = train_correct / len(train_dataset)
    print(f"📈 Epoch {epoch+1} - Loss: {train_loss:.4f}, Train Acc: {acc*100:.2f}%")

torch.save(model.state_dict(), "baseline_lstm.pth")
print(f"✅ 模型训练完成，耗时 {time.time() - start_time:.2f} 秒")

# 模型评估
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        x = batch["text"].to(device)
        y = batch["label"].to(device)
        out = model(x)
        preds = out.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# 结果评估
print("\n📊 分类报告:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0))
print("📌 混淆矩阵:")
print(confusion_matrix(all_labels, all_preds))
print(f"Macro F1 Score: {f1_score(all_labels, all_preds, average='macro'):.4f}")
print(f"Weighted F1 Score: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
