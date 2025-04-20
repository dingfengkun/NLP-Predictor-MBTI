import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import time

print("start!")

# 超参数设置（与 NumPy 版本保持一致）
MAX_LEN = 50
VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 16
LR = 0.05
EPOCHS = 5
BATCH_SIZE = 32

# 数据加载与处理
df = pd.read_csv("MBTI 500.csv")
texts = df["posts"].values
labels = LabelEncoder().fit_transform(df["type"].values)

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = labels

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset 和 Dataloader
class MBTIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

train_loader = DataLoader(MBTIDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MBTIDataset(X_val, y_val), batch_size=BATCH_SIZE)

# PyTorch LSTM 模型定义（结构与 NumPy 手写一致）
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)              # [batch, seq, embed_dim]
        _, (hidden, _) = self.lstm(x)      # hidden: [1, batch, hidden_dim]
        out = self.fc(hidden.squeeze(0))   # [batch, output_dim]
        return out

# 初始化模型
device = torch.device("cpu")
model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# 模型训练
print("🚀 开始训练 PyTorch LSTM ...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y_batch).sum().item()

    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}, Train Acc: {acc*100:.2f}%")

train_time = time.time() - start_time
print(f"✅ 总训练时间：{train_time:.2f} 秒")

# 测试
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        logits = model(x_batch)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"✅ 测试准确率: {acc*100:.2f}%")
