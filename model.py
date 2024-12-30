import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the Shakespeare dataset (you'll need to download this file and set the path)
with open("t8.shakespeare.txt", "r") as f:
    text = f.read()

# Prepare the dataset
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

seq_length = 100
step = 1
sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i:i + seq_length])
    next_chars.append(text[i + seq_length])

X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.float32)
y = np.zeros((len(sequences), vocab_size), dtype=np.float32)

for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# Custom Dataset for DataLoader
class ShakespeareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = ShakespeareDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the LSTM model
class ShakespeareModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(ShakespeareModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Only take the output from the last time step
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

# Hyperparameters
hidden_size = 256
num_layers = 2
learning_rate = 0.001
num_epochs = 20

model = ShakespeareModel(vocab_size, hidden_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    model.train()
    hidden = model.init_hidden(64)
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = tuple([h.data for h in hidden])  # Detach hidden state to avoid backprop through time
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, torch.argmax(targets, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Generating text
def generate_text(model, seed_text, char_to_idx, idx_to_char, length=500):
    model.eval()
    input_seq = torch.zeros((1, seq_length, vocab_size), dtype=torch.float32, device=device)
    for i, char in enumerate(seed_text):
        input_seq[0, i, char_to_idx[char]] = 1

    hidden = model.init_hidden(1)
    generated_text = seed_text

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        probs = torch.softmax(output, dim=1).data
        char_idx = torch.multinomial(probs, 1).item()
        next_char = idx_to_char[char_idx]
        generated_text += next_char

        input_seq = torch.zeros((1, seq_length, vocab_size), dtype=torch.float32, device=device)
        for i, char in enumerate(generated_text[-seq_length:]):
            input_seq[0, i, char_to_idx[char]] = 1

    return generated_text

# Generate some Shakespearean-style text
seed_text = "To be or not to be, that is the question: "
generated_text = generate_text(model, seed_text, char_to_idx, idx_to_char)
print(generated_text)
