import torch
import torch.nn as nn

class CharCNN(nn.Module):
    """
    Arquitetura 1: Rede Convolucional 1D (Character-level CNN)
    Focada em detectar padrões locais (ex: 'http', '.exe', 'admin').
    """
    def __init__(self, vocab_size, embed_dim, n_filters, kernel_sizes, fc_dim, dropout, num_classes, max_len=150):
        super(CharCNN, self).__init__()
        
        # 1. Camada de Embedding: Transforma índices inteiros em vetores densos
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. Camadas Convolucionais
        # Conv1d espera entrada (Batch, Channels, Length), por isso precisaremos permutar depois

        self.conv_layers = nn.ModuleList()
        in_channels = embed_dim

        # Se kernel_sizes for um int, replica para todas as camadas
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(n_filters)
            
        for out_channels, k_size in zip(n_filters, kernel_sizes):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k_size, padding=k_size//2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            ))
            in_channels = out_channels # Atualiza entrada da proxima camada
            
        # --- Cálculo Automático do Tamanho para a Camada Linear ---
        # "Passamos" um dado falso (dummy) pela rede só para ver qual tamanho sai
        dummy_input = torch.zeros(1, embed_dim, max_len)
        output_size = self._get_conv_output(dummy_input)
        
        # --- Camadas Densas ---
        self.fc1 = nn.Linear(output_size, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def _get_conv_output(self, x):
        # Função auxiliar para simular o forward pass
        with torch.no_grad():
            for layer in self.conv_layers:
                x = layer(x)
        return x.numel() # Retorna o número total de features (flatten)

    def forward(self, x):
        # x shape: (Batch, Max_Len)
        
        # Embedding
        x = self.embedding(x) # -> (Batch, Max_Len, Embed_Dim)
        
        # Ajustar dimensões para Conv1d: (Batch, Embed_Dim, Max_Len)
        x = x.permute(0, 2, 1)
        
        for layer in self.conv_layers:
            x = layer(x)
            
        x = x.flatten(1)           # Flatten dinâmico
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
class CharLSTM(nn.Module):
    """
    Arquitetura 2: LSTM (Long Short-Term Memory)
    Focada em aprender a sequência e dependências de longo prazo.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout, num_classes):
        super(CharLSTM, self).__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM Layer
        # batch_first=True garante que entrada/saída seja (Batch, Seq, Features)
        # Nota: Dropout no LSTM só funciona se n_layers > 1
        lstm_dropout = dropout if n_layers > 1 else 0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=lstm_dropout
        )
        
        # Camada de Classificação
        # Pegamos apenas o último estado oculto da sequência
        self.dropout_layer = nn.Dropout(dropout) # Dropout extra antes da FC
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
        
        out = self.dropout_layer(last_hidden_state)
        out = self.fc(out)
        
        return out