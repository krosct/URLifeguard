import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import sys

# Adiciona o diretório atual ao path para garantir que o Python encontre os módulos
sys.path.append(os.getcwd())

# Importando das subpastas (pacotes)
from utils.utils import url_to_tensor, VOCAB_SIZE
from models.models import CharCNN, CharLSTM

# Configuração do Dispositivo (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# --- 1. Dataset Customizado ---
class URLDataset(Dataset):
    def __init__(self, urls, labels, max_len=150):
        self.urls = urls
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        label = self.labels[idx]
        
        # Converter string para tensor numérico
        url_tensor = url_to_tensor(url, self.max_len)
        
        return url_tensor, torch.tensor(label, dtype=torch.long)

# --- 2. Função de Treinamento ---
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Treinando"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calcular acurácia
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# --- 3. Função de Avaliação ---
def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Avaliando"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# --- 4. Main ---
if __name__ == "__main__":
    # Configurações
    DATASET_PATH = "data/malicious_phish.csv" # Caminho relativo para a pasta data
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    MODEL_TYPE = "LSTM" # Opções: "CNN" ou "LSTM"
    
    # Verificar se o dataset existe
    if not os.path.exists(DATASET_PATH):
        print(f"ERRO: Dataset não encontrado em {DATASET_PATH}")
        print("Por favor, baixe o dataset e salve na pasta 'data'.")
        FileNotFoundError("Dataset não encontrado!")
        exit()

    # Carregar Dados
    print("Carregando dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Tratamento básico de labels
    classes = df['type'].unique()
    class_map = {c: i for i, c in enumerate(classes)}
    print(f"Classes mapeadas: {class_map}")

    X = df['url'].values
    y = df['type'].map(class_map).values
    
    # Divisão Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = URLDataset(X_train, y_train)
    test_dataset = URLDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Inicializar Modelo
    print(f"Inicializando modelo {MODEL_TYPE}...")
    num_classes = len(df['type'].unique())
    
    if MODEL_TYPE == "CNN":
        model = CharCNN(vocab_size=VOCAB_SIZE, num_classes=num_classes).to(device)
    elif MODEL_TYPE == "LSTM":
        model = CharLSTM(vocab_size=VOCAB_SIZE, num_classes=num_classes).to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loop de Treinamento
    for epoch in range(EPOCHS):
        print(f"\nÉpoca {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        
        print(f"Treino - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Teste  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
    # Vamos criar uma pasta 'checkpoints' para não misturar com o código
    os.makedirs("checkpoints", exist_ok=True)
    for i in range(0, 10000):
        save_path = f"checkpoints/urlifeguard_{MODEL_TYPE.lower()}_{i}.pth"
        if not os.path.exists(save_path):
            torch.save(model.state_dict(), save_path)
            print(f"\nModelo salvo em {save_path}")
            break
    else:
        print(f"\nNão foi possível salvar o modelo pois já existem 10000 modelos salvos.")
