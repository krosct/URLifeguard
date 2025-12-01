import torch
import torch.nn as nn

class CharCNN(nn.Module):
    """
    Arquitetura 1: Rede Convolucional 1D (Character-level CNN)
    Focada em detectar padrões locais (ex: 'http', '.exe', 'admin').
    """
    def __init__(self, vocab_size, embed_dim=32, num_classes=2, max_len=150):
        super(CharCNN, self).__init__()
        
        # 1. Camada de Embedding: Transforma índices inteiros em vetores densos
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. Camadas Convolucionais
        # Conv1d espera entrada (Batch, Channels, Length), por isso precisaremos permutar depois
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Reduz o tamanho pela metade
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # 3. Camadas Fully Connected (Densas)
        # Precisamos calcular o tamanho da entrada linear dinamicamente ou fixar baseado no max_len.
        # Após 2 pools de tamanho 2, o comprimento 150 vira aprox 37.
        linear_input_size = 256 * (max_len // 4) 
        
        self.fc1 = nn.Linear(linear_input_size, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Para evitar overfitting
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (Batch, Max_Len)
        
        # Embedding
        x = self.embedding(x) # -> (Batch, Max_Len, Embed_Dim)
        
        # Ajustar dimensões para Conv1d: (Batch, Embed_Dim, Max_Len)
        x = x.permute(0, 2, 1)
        
        # Bloco Conv 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Bloco Conv 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten (Aplanar) para entrar na camada densa
        x = x.flatten(1)
        
        # Classificação
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CharLSTM(nn.Module):
    """
    Arquitetura 2: LSTM (Long Short-Term Memory)
    Focada em aprender a sequência e dependências de longo prazo.
    """
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_classes=2):
        super(CharLSTM, self).__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM Layer
        # batch_first=True garante que entrada/saída seja (Batch, Seq, Features)
        self.lstm = nn.LSTM(input_size=embed_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=2, 
                            batch_first=True, 
                            dropout=0.2)
        
        # Camada de Classificação
        # Pegamos apenas o último estado oculto da sequência
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, Max_Len)
        
        x = self.embedding(x) # -> (Batch, Max_Len, Embed_Dim)
        
        # LSTM retorna: output, (hidden_state, cell_state)
        # output shape: (Batch, Max_Len, Hidden_Dim)
        # hidden_state shape: (Num_Layers, Batch, Hidden_Dim)
        lstm_out, (ht, ct) = self.lstm(x)
        
        # Usamos o último estado oculto da última camada LSTM para classificar
        # ht[-1] pega a última camada
        last_hidden_state = ht[-1]
        
        out = self.fc(last_hidden_state)
        
        return out