import optuna
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import sys

# ! Tem ajustar o documento dos modelos.

# Adiciona o diretório atual ao path para garantir que o Python encontre os módulos
sys.path.append(os.getcwd())

# Importando das subpastas (pacotes)
from utils.utils import url_to_tensor, VOCAB_SIZE
from models.models_optuna import CharCNN, CharLSTM

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
    
    for inputs, labels in tqdm(train_loader, desc="Treinando"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(train_loader)

# --- 3. Função de Avaliação ---
def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Avaliando"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calcular F1-Score (Weighted é bom para classes desbalanceadas)
    return running_loss / len(test_loader), f1_score(all_labels, all_preds, average='weighted')

def get_data(DATASET_PATH=None, sample_size=None):
    """
    Carrega os dados uma única vez.
    """

    # Verificar se o dataset existe
    if not os.path.exists(DATASET_PATH):
        print(f"ERRO: Dataset não encontrado em {DATASET_PATH}")
        print("Por favor, baixe o dataset e salve na pasta 'data'.")
        raise FileNotFoundError("Dataset não encontrado!")
        exit()
    
    # Carregar Dados
    print("Carregando dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Tratamento básico de labels
    classes = df['type'].unique()
    class_map = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Classes mapeadas ({num_classes}): {class_map}")

    # Amostragem para o Optuna ser mais rápido (Recomendado usar ~20% a 50% dos dados no tuning)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    X = df['url'].values
    y = df['type'].map(class_map).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = URLDataset(X_train, y_train)
    test_dataset = URLDataset(X_test, y_test)
    
    return train_dataset, test_dataset, num_classes

def objective(trial, MODEL_TYPE, data_tuple, best_f1_global):
    train_dataset, test_dataset, num_classes = data_tuple
    best_f1_local = 0.0

    # 1. Sugerir Hiperparâmetros
    learning_rate = trial.suggest_float("lr", 1e-7, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    epochs = trial.suggest_int("epochs", 1, 101, step=10)
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.6)
    
    # 2. Preparar DataLoaders com o batch_size sugerido
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    print(f"Parâmetros do Modelo {MODEL_TYPE}:")

    # 3. Inicializar Modelo
    if MODEL_TYPE == "CNN":
        n_conv_layers = trial.suggest_int("n_conv_layers", 1, 6)
        kernel_sizes = trial.suggest_int("kernel_sizes", 2, 5)
        filters_list = []

        # Para cada camada, o Optuna escolhe o tamanho do filtro
        for i in range(n_conv_layers):
            filters_list.append(trial.suggest_categorical(f"n_filter_l{i}", [16, 32, 64, 128, 256]))
            
        fc_dim = trial.suggest_int("fc_dim", 32, 512)
        
        model = CharCNN(
            vocab_size=VOCAB_SIZE,
            embed_dim=embed_dim,
            n_filters=filters_list,  # Passamos a lista gerada [64, 128...]
            kernel_sizes=kernel_sizes,          # Pode fixar ou variar também
            fc_dim=fc_dim,
            dropout=dropout,
            num_classes=num_classes
        ).to(device)

        print(f"learning_rate: {learning_rate}, batch_size: {batch_size}, optimizer_name: {optimizer_name}, epochs: {epochs}, embed_dim: {embed_dim}, dropout: {dropout}, n_conv_layers: {n_conv_layers}, kernel_sizes: {kernel_sizes}")
        
    elif MODEL_TYPE == "LSTM":
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
        n_layers = trial.suggest_int("n_layers", 1, 4) # Varia de 1 a 4 camadas LSTM empilhadas
        
        model = CharLSTM(
            vocab_size=VOCAB_SIZE,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            num_classes=num_classes
        ).to(device)

        print(f"learning_rate: {learning_rate}, batch_size: {batch_size}, optimizer_name: {optimizer_name}, epochs: {epochs}, embed_dim: {embed_dim}, dropout: {dropout}, hidden_dim: {hidden_dim}, n_layers: {n_layers}")
        
    else:
        model = None
        raise ValueError("Valor de MODEL_TYPE = '{MODEL_TYPE}' é inválido !")
        exit()
    
    # 4. Definir Otimizador
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        
    criterion = nn.CrossEntropyLoss()
    
    # 5. Loop de Treino
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        test_loss, val_f1  = evaluate_model(model, test_loader, criterion)
        print(f"Treino - Loss: {train_loss:.2f}")
        print(f"Teste  - Loss: {test_loss:.2f}, F1-Score: {val_f1*100:.2f}%")
        
        # Reportar métrica intermediária para o Optuna (para Pruning)
        trial.report(val_f1, epoch)
        
        # Se o resultado for muito ruim, o Optuna para este teste aqui mesmo (Pruning)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if val_f1 > best_f1_local:
            best_f1_local = val_f1

        # if best_f1_local > best_f1_global['score'] and best_f1_global['score'] != 0:
        #     os.makedirs("checkpoints", exist_ok=True)
        #     for i in range(0, 10000):
        #         save_path = f"checkpoints/urlifeguard_{MODEL_TYPE.lower()}_{i}.pth"
        #         if not os.path.exists(save_path):
        #             torch.save(model.state_dict(), save_path)
        #             print(f"\nModelo salvo em {save_path}")
        #             break
        #     else:
        #         print(f"\nNão foi possível salvar o modelo pois já existem 10000 modelos salvos.")
            
    return best_f1_local # O objetivo é MAXIMIZAR o F1_SCORE

def main(n_trials, MODEL_TYPE, DATASET_PATH):

    best_f1_global = {'score': 0.0}
    # Criar o estudo
    study = optuna.create_study(
        study_name="estudo_url_lifeguard",
        storage="sqlite:///db.sqlite",     # Cria um arquivo 'db.sqlite3' na pasta
        load_if_exists=True,               # Se o arquivo já existir, ele CARREGA e continua
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner()
    )
    # study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    print("Iniciando otimização...")
    data_tuple = get_data(DATASET_PATH)
    study.optimize(lambda trial: objective(trial, MODEL_TYPE, data_tuple, best_f1_global), n_trials=n_trials)
    
    print("\n--- Melhores Resultados ---")
    print(f"Melhor F1-Score: {study.best_value:.2f}%")
    print("Melhores Parâmetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    os.makedirs("results", exist_ok=True)
    for i in range(0, 10000):
        save_path = f"results/optuna_results_{MODEL_TYPE.lower()}_{i}.csv"
        if not os.path.exists(save_path):
            df_results = study.trials_dataframe()
            df_results.to_csv(save_path)
            print("Resultados detalhados salvos em '{save_path}'.")
            break
    else:
        print(f"\nNão foi possível salvar o csv pois já existem 10000 csv salvos.")

if __name__ == "__main__":
    DATASET_PATH = "data/malicious_phish.csv"
    MODEL_TYPE = "CNN" # Opções: "CNN" ou "LSTM"
    n_trials = 999
    main(n_trials, MODEL_TYPE, DATASET_PATH)

# Comando para ver o progresso do treinamento
# optuna-dashboard sqlite:///db.sqlite3
