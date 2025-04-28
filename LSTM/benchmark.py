import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import pandas as pd
import torch.optim.adam
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Global configuration
class Config:
    VOCAB_SIZE = 1000      # Vocabulary size
    EMBEDDING_DIM = 16     # Word embedding dimension
    HIDDEN_DIM = 32        # LSTM hidden state dimension
    OUTPUT_DIM = 16        # Number of output classes
    LR = 1e-2              # Learning rate
    EPOCHS = 3             # Number of training epochs
    BATCH_SIZE = 64        # Mini-batch size
    MAX_LEN = 50           # Maximum sequence length
    SEED = 42              # Random seed for reproducibility
    csv_path = "MBTI 500.csv"

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

def load_and_preprocess_data(csv_path: str, test_size: float = 0.2, random_state: int = 42,  MAX_WORDS: int = 10000,
MAX_LEN:int = 500, BATCH_SIZE: int = 64 ) -> tuple:
    """
    Complete data loading and preprocessing pipeline for MBTI classification task.
    
    Args:
        csv_path (str): Path to the CSV file containing the dataset
        test_size (float): Proportion of dataset to include in validation split (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (train_loader, val_loader, tokenizer, label_encoder)
            - train_loader: PyTorch DataLoader for training data
            - val_loader: PyTorch DataLoader for validation data
            - tokenizer: Trained Keras Tokenizer instance
            - label_encoder: Trained LabelEncoder instance
    """
    # Load raw data from CSV
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["type"].values)
    texts = df["posts"].values
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=test_size, random_state=random_state)
    train_dataset = MBTIDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = MBTIDataset(X_val, y_val, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Print final statistics
    print("\n Preprocessing complete!")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Vocabulary size: {len(tokenizer.word_index):,}")
    print(f"  Maximum sequence length: {MAX_LEN}")
    
    return train_loader, val_loader, tokenizer, label_encoder

# PyTorch LSTM Implementation
class TorchLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through the network
        x = self.embedding(x)               # Convert token IDs to embeddings
        _, (hidden, _) = self.lstm(x)       # Process sequence with LSTM
        return self.fc(hidden.squeeze(0))   # Final classification layer

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on validation data
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def run_torch(device='cpu'):
    # Set reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # Prepare data
    train_loader, val_loader, tokenizer, label_encoder = load_and_preprocess_data(
        csv_path=Config.csv_path,
        MAX_WORDS=Config.VOCAB_SIZE,
        MAX_LEN=Config.MAX_LEN,
        BATCH_SIZE=Config.BATCH_SIZE
    )
    
    # Initialize model and training components
    model = TorchLSTM(
        vocab_size=Config.VOCAB_SIZE, 
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    
    # Training loop
    start_time = time.time()
    best_val_acc = 0
    
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Prepare batch
            inputs = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Calculate batch statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch statistics
        train_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Calculate training time
    training_time = time.time() - start_time
    return training_time, train_acc, best_val_acc

class NumpyLSTM:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        # Initialize all LSTM parameters with Xavier initialization
        # Input gate parameters
        self.W_xi = np.random.randn(embedding_dim, hidden_dim) * np.sqrt(2/(embedding_dim+hidden_dim))
        self.W_hi = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2/(hidden_dim*2))
        self.b_i = np.zeros(hidden_dim)
        
        # Forget gate parameters
        self.W_xf = np.random.randn(embedding_dim, hidden_dim) * np.sqrt(2/(embedding_dim+hidden_dim))
        self.W_hf = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2/(hidden_dim*2))
        self.b_f = np.zeros(hidden_dim)
        
        # Cell state parameters
        self.W_xc = np.random.randn(embedding_dim, hidden_dim) * np.sqrt(2/(embedding_dim+hidden_dim))
        self.W_hc = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2/(hidden_dim*2))
        self.b_c = np.zeros(hidden_dim)
        
        # Output gate parameters
        self.W_xo = np.random.randn(embedding_dim, hidden_dim) * np.sqrt(2/(embedding_dim+hidden_dim))
        self.W_ho = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2/(hidden_dim*2))
        self.b_o = np.zeros(hidden_dim)
        
        # Final fully connected layer
        self.W_fc = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/(hidden_dim+output_dim))
        self.b_fc = np.zeros(output_dim)
        
        # Embedding matrix
        self.embedding = np.random.randn(vocab_size, embedding_dim) * np.sqrt(2/embedding_dim)
        
        # Cache for backward pass
        self.cache = []
        self.grads = None
        
        # Store dimensions
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        """
        Forward pass for LSTM
        Args:
            x: Input sequence of shape (seq_len,)
        Returns:
            output: Final output after processing sequence
        """
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        self.cache = []
        
        for t in range(len(x)):
            # Embedding lookup
            x_t = x[t]  # Scalar index
            if x_t >= self.vocab_size:
                x_t = 0  # OOV token
            embed = self.embedding[x_t]
            
            # Gates calculation
            i = self.sigmoid(embed @ self.W_xi + h @ self.W_hi + self.b_i)
            f = self.sigmoid(embed @ self.W_xf + h @ self.W_hf + self.b_f)
            o = self.sigmoid(embed @ self.W_xo + h @ self.W_ho + self.b_o)
            c_hat = np.tanh(embed @ self.W_xc + h @ self.W_hc + self.b_c)
            
            # Cell state update
            c = f * c + i * c_hat
            h = o * np.tanh(c)
            
            # Save intermediate values for backward pass
            self.cache.append((
                embed.copy(), i, f, o, c_hat, c.copy(), h.copy(), x_t
            ))
        
        # Final output
        output = h @ self.W_fc + self.b_fc
        return output

    def backward(self, x, y_true):
        """
        Backward pass through time (BPTT)
        Returns:
            grads: Dictionary containing gradients for all parameters
        """
        grads = {
            'W_xi': np.zeros_like(self.W_xi),
            'W_hi': np.zeros_like(self.W_hi),
            'b_i': np.zeros_like(self.b_i),
            'W_xf': np.zeros_like(self.W_xf),
            'W_hf': np.zeros_like(self.W_hf),
            'b_f': np.zeros_like(self.b_f),
            'W_xc': np.zeros_like(self.W_xc),
            'W_hc': np.zeros_like(self.W_hc),
            'b_c': np.zeros_like(self.b_c),
            'W_xo': np.zeros_like(self.W_xo),
            'W_ho': np.zeros_like(self.W_ho),
            'b_o': np.zeros_like(self.b_o),
            'W_fc': np.zeros_like(self.W_fc),
            'b_fc': np.zeros_like(self.b_fc),
            'embedding': np.zeros_like(self.embedding)
        }
        
        # Initialize gradients
        dh_next = np.zeros(self.hidden_dim)  # dh from next time step
        dc_next = np.zeros(self.hidden_dim)  # dc from next time step
        
        # Backprop through time
        for t in reversed(range(len(self.cache))):
            embed, i, f, o, c_hat, c, h, x_t = self.cache[t]
            
            # Gradients from output layer
            if t == len(self.cache)-1:
                # Get softmax gradients
                scores = h @ self.W_fc + self.b_fc
                probs = self.softmax(scores)
                d_output = probs.copy()
                d_output[y_true] -= 1  # Gradient of cross-entropy
                
                dh = d_output @ self.W_fc.T + dh_next
                grads['W_fc'] += np.outer(h, d_output)
                grads['b_fc'] += d_output
            else:
                dh = dh_next
            
            # Gates gradients
            do = dh * np.tanh(c)
            do_raw = do * o * (1 - o)
            
            dc = dh * o * (1 - np.tanh(c)**2) + dc_next
            dc_hat = dc * i
            dc_hat_raw = dc_hat * (1 - c_hat**2)
            
            di = dc * c_hat
            di_raw = di * i * (1 - i)
            
            df = dc * (self.cache[t-1][5] if t > 0 else np.zeros_like(dc))
            df_raw = df * f * (1 - f)
            
            # Parameter gradients
            grads['W_xo'] += np.outer(embed, do_raw)
            grads['W_ho'] += np.outer((self.cache[t-1][6] if t > 0 else np.zeros_like(h)), do_raw)
            grads['b_o'] += do_raw
            
            grads['W_xi'] += np.outer(embed, di_raw)
            grads['W_hi'] += np.outer((self.cache[t-1][6] if t > 0 else np.zeros_like(h)), di_raw)
            grads['b_i'] += di_raw
            
            grads['W_xf'] += np.outer(embed, df_raw)
            grads['W_hf'] += np.outer((self.cache[t-1][6] if t > 0 else np.zeros_like(h)), df_raw)
            grads['b_f'] += df_raw
            
            grads['W_xc'] += np.outer(embed, dc_hat_raw)
            grads['W_hc'] += np.outer((self.cache[t-1][6] if t > 0 else np.zeros_like(h)), dc_hat_raw)
            grads['b_c'] += dc_hat_raw
            
            # Embedding gradients
            d_embed = (
                do_raw @ (self.W_xo.T) +
                di_raw @ (self.W_xi.T) +
                df_raw @ (self.W_xf.T) +
                dc_hat_raw @ (self.W_xc.T)
            )
            grads['embedding'][x_t] += d_embed
            
            # Backprop to previous time step
            if t > 0:
                dh_prev = (
                    do_raw @ self.W_ho +
                    di_raw @ self.W_hi +
                    df_raw @ self.W_hf +
                    dc_hat_raw @ self.W_hc
                )
                dc_prev = f * dc
                
                dh_next = dh_prev
                dc_next = dc_prev
            
        self.grads = grads
        return grads

    def update_params(self, lr):
        """
        Update parameters using calculated gradients
        Args:
            lr: Learning rate
        """
        # Update all parameters
        self.W_xi -= lr * self.grads['W_xi']
        self.W_hi -= lr * self.grads['W_hi']
        self.b_i -= lr * self.grads['b_i']
        
        self.W_xf -= lr * self.grads['W_xf']
        self.W_hf -= lr * self.grads['W_hf']
        self.b_f -= lr * self.grads['b_f']
        
        self.W_xc -= lr * self.grads['W_xc']
        self.W_hc -= lr * self.grads['W_hc']
        self.b_c -= lr * self.grads['b_c']
        
        self.W_xo -= lr * self.grads['W_xo']
        self.W_ho -= lr * self.grads['W_ho']
        self.b_o -= lr * self.grads['b_o']
        
        self.W_fc -= lr * self.grads['W_fc']
        self.b_fc -= lr * self.grads['b_fc']
        
        self.embedding -= lr * self.grads['embedding']
    
    @staticmethod
    def sigmoid(x):
        # Clip values to avoid overflow
        x = np.clip(x, -30, 30)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

def generate_numpy_dataset(train_loader, tokenizer, label_encoder):
    """
    Convert PyTorch DataLoader to NumPy arrays for the NumpyLSTM implementation
    """
    X_train = []
    y_train = []
    
    for batch in train_loader:
        texts = batch['text'].numpy()
        labels = batch['label'].numpy()
        
        for i in range(len(texts)):
            X_train.append(texts[i])
            y_train.append(labels[i])
    
    return np.array(X_train), np.array(y_train)

def run_numpy():
    # Set random seed for reproducibility
    np.random.seed(Config.SEED)
    
    # Prepare data - use the same pipeline as PyTorch for consistency
    train_loader, val_loader, tokenizer, label_encoder = load_and_preprocess_data(
        csv_path=Config.csv_path,
        MAX_WORDS=Config.VOCAB_SIZE,
        MAX_LEN=Config.MAX_LEN,
        BATCH_SIZE=Config.BATCH_SIZE
    )
    
    # Convert to NumPy format
    X_train, y_train = generate_numpy_dataset(train_loader, tokenizer, label_encoder)
    X_val, y_val = generate_numpy_dataset(val_loader, tokenizer, label_encoder)
    
    # Initialize model
    model = NumpyLSTM(
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM
    )
    
    # Training loop
    start_time = time.time()
    best_val_acc = 0
    
    for epoch in range(Config.EPOCHS):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        train_loss = 0
        train_correct = 0
        
        # Process each batch
        for i in range(0, len(X_train_shuffled), Config.BATCH_SIZE):
            batch_X = X_train_shuffled[i:i+Config.BATCH_SIZE]
            batch_y = y_train_shuffled[i:i+Config.BATCH_SIZE]
            
            batch_loss = 0
            batch_correct = 0
            
            # Process each sample in batch
            for j in range(len(batch_X)):
                x = batch_X[j]
                y_true = batch_y[j]
                
                # Forward pass
                output = model.forward(x)
                
                # Calculate loss (for tracking only)
                probs = model.softmax(output)
                batch_loss -= np.log(probs[y_true] + 1e-10)  # Cross-entropy loss
                
                # Check prediction
                pred = np.argmax(output)
                if pred == y_true:
                    batch_correct += 1
                
                # Backward pass and parameter update
                model.backward(x, y_true)
                model.update_params(Config.LR)
            
            # Accumulate batch statistics
            train_loss += batch_loss / len(batch_X)
            train_correct += batch_correct
        
        # Calculate epoch statistics
        train_loss /= (len(X_train_shuffled) // Config.BATCH_SIZE)
        train_acc = 100 * train_correct / len(X_train_shuffled)
        
        # Validation
        val_preds = []
        val_loss = 0
        
        for i in range(len(X_val)):
            x = X_val[i]
            y_true = y_val[i]
            
            # Forward pass
            output = model.forward(x)
            probs = model.softmax(output)
            val_loss -= np.log(probs[y_true] + 1e-10)
            
            pred = np.argmax(output)
            val_preds.append(pred)
        
        val_loss /= len(X_val)
        val_acc = 100 * accuracy_score(y_val, val_preds)
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        
        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Calculate total training time
    training_time = time.time() - start_time
    
    return training_time, train_acc, best_val_acc

# Benchmark comparison function
def benchmark():
    # Set random seeds for reproducibility
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    
    print("Starting benchmark comparison...")
    
    # Run PyTorch CPU version
    print("\nRunning PyTorch (CPU) implementation...")
    torch_time_cpu, torch_train_acc_cpu, torch_val_acc_cpu = run_torch('cpu')
    
    # Run PyTorch GPU version if available
    torch_time_gpu = float('inf')
    torch_train_acc_gpu = 0
    torch_val_acc_gpu = 0
    
    if torch.cuda.is_available():
        print("\nRunning PyTorch (CUDA) implementation...")
        torch.cuda.synchronize()  # Ensure CUDA initialization
        torch_time_gpu, torch_train_acc_gpu, torch_val_acc_gpu = run_torch('cuda')
    
    # Run NumPy CPU version
    print("\nRunning NumPy (CPU) implementation...")
    numpy_time, numpy_train_acc, numpy_val_acc = run_numpy()
    
    # Print comparison results
    print("\n=========== Benchmark Results ===========")
    print(f"| Implementation   | Training Time | Train Acc | Val Acc |")
    print("|------------------|---------------|-----------|---------|")
    print(f"| PyTorch (CPU)    | {torch_time_cpu:6.2f}s     | {torch_train_acc_cpu:.2f}%   | {torch_val_acc_cpu:.2f}%  |")
    
    if torch.cuda.is_available():
        print(f"| PyTorch (CUDA)   | {torch_time_gpu:6.2f}s     | {torch_train_acc_gpu:.2f}%   | {torch_val_acc_gpu:.2f}%  |")

    
    # Calculate speedup factors
    speedup_cpu = numpy_time / torch_time_cpu
    print(f"\n⚡ Speedup Factors:")
    print(f"- NumPy → PyTorch CPU: {speedup_cpu:.1f}x faster")
    
    if torch.cuda.is_available():
        speedup_gpu = numpy_time / torch_time_gpu
        print(f"- NumPy → PyTorch GPU: {speedup_gpu:.1f}x faster")

if __name__ == "__main__":
    benchmark()