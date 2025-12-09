import shutil
from typing import Any
import optuna
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
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# Importando das subpastas (pacotes)
from utils.utils import url_to_tensor, VOCAB_SIZE
from models.models_optuna import CharCNN, CharLSTM

# Adiciona o diretório atual ao path para garantir que o Python encontre os módulos
sys.path.append(os.getcwd())

# Configuração do Dispositivo (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class InfoConfig():
    def __init__(self, *, N_TRIALS, MODEL_VERSION, MODEL_TYPES, DATASET_PATH, LEARNING_RATE, BATCH_SIZE, \
                 OPTIMIZER_NAME, EPOCHS, EMBED_DIM, DROPOUT, PATIENCE, WEIGHT_DECAY, N_CONV_LAYERS_CNN, KERNEL_SIZES_CNN, \
                 FC_DIM_CNN, HIDDEN_DIM_LSTM, N_LAYERS_LSTM, FILTER_SIZES_CNN, NR_SAMPLE, TEST_SIZE, DB_NAME, SCREEN_WIDTH):
        self.N_TRIALS = N_TRIALS
        self.MODEL_VERSION = MODEL_VERSION
        self.MODEL_TYPES = MODEL_TYPES
        self.DATASET_PATH = DATASET_PATH
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.OPTIMIZER_NAME = OPTIMIZER_NAME
        self.EPOCHS = EPOCHS
        self.EMBED_DIM = EMBED_DIM
        self.DROPOUT = DROPOUT
        self.PATIENCE = PATIENCE
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.N_CONV_LAYERS_CNN = N_CONV_LAYERS_CNN
        self.KERNEL_SIZES_CNN = KERNEL_SIZES_CNN
        self.FC_DIM_CNN = FC_DIM_CNN
        self.HIDDEN_DIM_LSTM = HIDDEN_DIM_LSTM
        self.N_LAYERS_LSTM = N_LAYERS_LSTM
        self.FILTER_SIZES_CNN = FILTER_SIZES_CNN
        self.NR_SAMPLE = NR_SAMPLE
        self.TEST_SIZE = TEST_SIZE
        self.DB_NAME = DB_NAME
        self.SCREEN_WIDTH = SCREEN_WIDTH

        self.best_params: dict[str, dict[str, Any]] = {}
        self.filter_list_cnn = []

    def load_data(self):
        if hasattr(self, 'DATA_TUPLE'):
            print(f"\n[AVISO] DATA_TUPLE já existe. Abortando load_data!")
            return
        
        self.DATA_TUPLE = get_data(self)
        self.TRAIN_DATASET, self.TEST_DATASET, self.NUM_CLASSES, self.CLASS_WEIGHTS = self.DATA_TUPLE
        self.CLASS_WEIGHTS = self.CLASS_WEIGHTS.to(device)
    
    def print_g(self, model_type: str='', best: bool=False):
        if best:
            ans = f"[{model_type}] LEARNING_RATE: {self.best_params[model_type]['learning_rate']}, \n[{model_type}] BATCH_SIZE: {self.best_params[model_type]['batch_size']}, \n[{model_type}] OPTIMIZER_NAME: {self.best_params[model_type]['optimizer_name']}, \n[{model_type}] EMBED_DIM: {self.best_params[model_type]['embed_dim']}, \n[{model_type}] DROPOUT: {self.best_params[model_type]['dropout']}"
        else:
            ans = f"[{model_type}] LEARNING_RATE: {self.LEARNING_RATE}, \n[{model_type}] BATCH_SIZE: {self.BATCH_SIZE}, \n[{model_type}] OPTIMIZER_NAME: {self.OPTIMIZER_NAME}, \n[{model_type}] EMBED_DIM: {self.EMBED_DIM}, \n[{model_type}] DROPOUT: {self.DROPOUT}"
        return ans
    
    def print_c(self, model_type: str='', best: bool=False):
        if best:
            ans = f"[{model_type}] FILTERS_LIST: {self.best_params[model_type]['filters_list']}, \n[{model_type}] KERNEL_SIZES: {self.best_params[model_type]['kernel_sizes']}, \n[{model_type}] FC_DIM: {self.best_params[model_type]['fc_dim']}"
        else:
            ans = f"[{model_type}] FILTERS_LIST: {self.filter_list_cnn}, \n[{model_type}] KERNEL_SIZES: {self.KERNEL_SIZES_CNN}, \n[{model_type}] FC_DIM: {self.FC_DIM_CNN}"
        return ans
    
    def print_l(self, model_type: str='', best: bool=False):
        if best:
            ans = f"[{model_type}] HIDDEN_DIM: {self.best_params[model_type]['hidden_dim']}, \n[{model_type}] N_LAYERS: {self.best_params[model_type]['n_layers']}"
        else:
            ans = f"[{model_type}] HIDDEN_DIM: {self.HIDDEN_DIM_LSTM}, \n[{model_type}] N_LAYERS: {self.N_LAYERS_LSTM}"
        return ans
    
    def print_a(self, model_type: str=''):
        campos = [
            "N_TRIALS", "MODEL_VERSION", "MODEL_TYPES", "DATASET_PATH", 
            "LEARNING_RATE", "BATCH_SIZE", "OPTIMIZER_NAME", "EPOCHS", 
            "EMBED_DIM", "DROPOUT", "PATIENCE", "WEIGHT_DECAY", 
            "N_CONV_LAYERS_CNN", "KERNEL_SIZES_CNN", "FC_DIM_CNN", 
            "HIDDEN_DIM_LSTM", "N_LAYERS_LSTM", "FILTER_SIZES_CNN", 
            "NR_SAMPLE", "TEST_SIZE", "DB_NAME"
        ]

        linhas = []
        for i, campo in enumerate(campos, 1):
            valor = getattr(self, campo) 
            linhas.append(f"{i}. {campo}: {valor}")

        return "\n".join(linhas)

class URLDataset(Dataset):
    def __init__(self, urls, labels, max_len=150):
        print("Pré-processando dados para a RAM (Isso pode levar alguns segundos)...")
        
        self.X = []
        for url in tqdm(urls, desc="Tokenizando URLs"):
            self.X.append(url_to_tensor(url, max_len))

        self.X = torch.stack(self.X) 
        self.y = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Função de Treinamento ---
def train_model(model, train_loader, criterion, optimizer, model_type):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(train_loader, desc=f"[{model_type}] Treinando"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(train_loader)

# --- Função de Avaliação ---
def evaluate_model(model, test_loader, criterion, model_type):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"[{model_type}] Avaliando"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calcular F1-Score (Weighted é bom para classes desbalanceadas)
    return running_loss / len(test_loader), f1_score(all_labels, all_preds, average='weighted')

def get_data(info: InfoConfig):
    """
    Carrega os dados uma única vez.
    """

    # Verificar se o dataset existe
    if not os.path.exists(info.DATASET_PATH):
        print(f"ERRO: Dataset não encontrado em {info.DATASET_PATH}\nPor favor, baixe o dataset e salve na pasta 'data'.")
        raise FileNotFoundError("Dataset não encontrado!")
    
    # Carregar Dados
    print_separator_title(info, "Carregando dataset", "=")
    df = pd.read_csv(info.DATASET_PATH)

    if info.NR_SAMPLE:
        df = df.sample(n=info.NR_SAMPLE, random_state=42)

    # Removendo duplicatas
    df.drop_duplicates(subset=['url'], inplace=True)
    
    # Tratamento básico de labels
    classes = df['type'].unique()
    class_map = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Classes mapeadas ({num_classes}): {class_map}")

    X = df['url'].values
    y = df['type'].map(class_map).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=info.TEST_SIZE, random_state=42)

    # Calcular pesos (inverso da frequência)
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    # Converter para tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    train_dataset = URLDataset(X_train, y_train)
    test_dataset = URLDataset(X_test, y_test)

    print_separator_end(info, "=")

    return train_dataset, test_dataset, num_classes, class_weights_tensor

def objective(trial, info: InfoConfig, model_type: str):
    # Sugerir Hiperparâmetros
    learning_rate = trial.suggest_float("learning_rate", info.LEARNING_RATE[0], info.LEARNING_RATE[-1], log=True)
    batch_size = trial.suggest_categorical("batch_size", info.BATCH_SIZE)
    optimizer_name = trial.suggest_categorical("optimizer_name", info.OPTIMIZER_NAME)
    embed_dim = trial.suggest_categorical("embed_dim", info.EMBED_DIM)
    dropout = trial.suggest_float("dropout", info.DROPOUT[0], info.DROPOUT[-1])
    weight_decay = trial.suggest_float("weight_decay", info.WEIGHT_DECAY[0], info.WEIGHT_DECAY[-1], log=True)
    
    print(f"\n[{model_type}] Parâmetros do Modelo:")

    # Inicializar Modelo
    if model_type == "CNN":
        n_conv_layers = trial.suggest_int("n_conv_layers", info.N_CONV_LAYERS_CNN[0], info.N_CONV_LAYERS_CNN[-1])
        kernel_sizes = trial.suggest_int("kernel_sizes", info.KERNEL_SIZES_CNN[0], info.KERNEL_SIZES_CNN[-1])
        filters_list = []

        # Para cada camada, o Optuna escolhe o tamanho do filtro
        for i in range(n_conv_layers):
            filters_list.append(trial.suggest_categorical(f"n_filter_l{i}", info.FILTER_SIZES_CNN))
            
        fc_dim = trial.suggest_int("fc_dim", info.FC_DIM_CNN[0], info.FC_DIM_CNN[-1])
        
        model = CharCNN(
            vocab_size=VOCAB_SIZE,
            embed_dim=embed_dim,
            n_filters=filters_list,
            kernel_sizes=kernel_sizes,
            fc_dim=fc_dim,
            dropout=dropout,
            num_classes=info.NUM_CLASSES
        ).to(device)

        print(f"[{model_type}] learning_rate: {learning_rate}, \n[{model_type}] batch_size: {batch_size}, \n[{model_type}] optimizer_name: {optimizer_name}, \n[{model_type}] embed_dim: {embed_dim}, \n[{model_type}] dropout: {dropout}, \n[{model_type}] n_conv_layers: {n_conv_layers}, \n[{model_type}] filters_list: {filters_list}, \n[{model_type}] kernel_sizes: {kernel_sizes}, \n[{model_type}] fc_dim: {fc_dim}\n")
        
    elif model_type == "LSTM":
        hidden_dim = trial.suggest_int("hidden_dim", info.HIDDEN_DIM_LSTM[0], info.HIDDEN_DIM_LSTM[-1])
        n_layers = trial.suggest_int("n_layers", info.N_LAYERS_LSTM[0], info.N_LAYERS_LSTM[-1])
        
        model = CharLSTM(
            vocab_size=VOCAB_SIZE,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            num_classes=info.NUM_CLASSES
        ).to(device)

        print(f"[{model_type}] learning_rate: {learning_rate}, \n[{model_type}] batch_size: {batch_size}, \n[{model_type}] optimizer_name: {optimizer_name}, \n[{model_type}] embed_dim: {embed_dim}, \n[{model_type}] dropout: {dropout}, \n[{model_type}] n_layers: {n_layers}, \n[{model_type}] hidden_dim: {hidden_dim}\n")
        
    else:
        model = None
        raise ValueError(f"Valor de MODEL_TYPE = '{model_type}' é inválido !")
    
    # Definir Otimizador
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    criterion = nn.CrossEntropyLoss(weight=info.CLASS_WEIGHTS, label_smoothing=0.1)
    
    # Configurações do Early Stopping
    patience = info.PATIENCE
    trigger_times = 0       # Contador de quantas vezes falhou em melhorar
    best_f1_local = 0.0
    
    # Preparar DataLoaders
    train_loader = DataLoader(info.TRAIN_DATASET, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(info.TEST_DATASET, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Loop de Treino
    for epoch in range(info.EPOCHS):
        _ = train_model(model, train_loader, criterion, optimizer, model_type)
        _, val_f1  = evaluate_model(model, test_loader, criterion, model_type)
        print(f"[{model_type}] Epoch [{epoch+1}/{info.EPOCHS}] | F1: {val_f1*100:.4f}%")
        
        # Reportar métrica intermediária para o Optuna (para Pruning)
        trial.report(val_f1, epoch)
        if trial.should_prune():
            print()
            raise optuna.exceptions.TrialPruned()
        
        if val_f1 > best_f1_local:
            best_f1_local = float(val_f1)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"[{model_type}] Early stopping na época {epoch+1}!\n")
                break
            
    return best_f1_local # O objetivo é MAXIMIZAR o F1_SCORE

def study(info: InfoConfig, model_type: str):
    # Criar o estudo
    study = optuna.create_study(
        study_name=f"estudo_urlifeguard_{model_type.lower()}_{info.MODEL_VERSION}",
        storage=f"sqlite:///{info.DB_NAME}",     # Cria um arquivo 'db.sqlite' na pasta
        load_if_exists=True,               # Se o arquivo já existir, ele CARREGA e continua
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Conta quantos trials já existem no banco para esse modelo
    trials_existentes = len(study.trials)
    
    print(f"\n[{model_type}] Trials já realizados: {trials_existentes}")
    print(f"[{model_type}] Meta total: {info.N_TRIALS}")

    if trials_existentes < info.N_TRIALS:
        trials_restantes = info.N_TRIALS - trials_existentes
        print(f"[{model_type}] Iniciando mais {trials_restantes} trials para atingir a meta...")
        print(f"[{model_type}] Iniciando treinamento...")
        
        study.optimize(lambda trial: objective(trial, info, model_type), n_trials=trials_restantes)

        os.makedirs("results", exist_ok=True)
        save_path = f"results/optuna_results_{model_type.lower()}_{info.MODEL_VERSION}.csv"
        df_results = study.trials_dataframe()
        df_results.to_csv(save_path)
        print(f"[{model_type}] Resultados detalhados salvos em '{save_path}'.")
            
    else:
        print(f"[{model_type}] Meta já atingida! Pulando treinamento.")

    print_separator_title(info, "Melhores Resultados", "=")
    try:
        print(f"Melhor F1-Score: {study.best_value:.4f}%")
        print("Melhores Parâmetros:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
    except:
        print("Nenhum trial completado com sucesso ainda.")
    print_separator_end(info, "=")

    info.best_params[model_type] = study.best_params

def create_model(info: InfoConfig, model_type: str):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    model_filename = f"urlifeguard_FINAL_{model_type.lower()}_{info.MODEL_VERSION}"
    save_path_model = f"checkpoints/{model_filename}.pth"
    save_path_csv = f"results/{model_filename}_history.csv"

    if os.path.exists(save_path_model) and os.path.exists(save_path_csv):
        print(f"\n[{model_type}] Modelo e histórico já existem! A criação de modelo foi abortada.")
        return

    # Inicializar Modelo
    print_separator_title(info, "Iniciando Treinamento Final", "=")

    if model_type == "CNN":
        filter_list_cnn = []
        for i in range(info.best_params["CNN"]["n_conv_layers"]):
            # Busca a chave f"n_filter_l{i}" (ex: n_filter_l0, n_filter_l1...)
            info.filter_list_cnn.append(info.best_params["CNN"][f"n_filter_l{i}"])

        info.best_params["CNN"]['filters_list'] = info.filter_list_cnn
            
        
        model = CharCNN(
            vocab_size=VOCAB_SIZE,
            embed_dim=info.best_params["CNN"]["embed_dim"],
            n_filters=info.filter_list_cnn,
            kernel_sizes=info.best_params["CNN"]["kernel_sizes"],
            fc_dim=info.best_params["CNN"]["fc_dim"],
            dropout=info.best_params["CNN"]["dropout"],
            num_classes=info.NUM_CLASSES
        ).to(device)

    elif model_type == "LSTM":
        model = CharLSTM(
            vocab_size=VOCAB_SIZE,
            embed_dim=info.best_params["LSTM"]["embed_dim"],
            hidden_dim=info.best_params["LSTM"]["hidden_dim"],
            n_layers=info.best_params["LSTM"]["n_layers"],
            dropout=info.best_params["LSTM"]["dropout"],
            num_classes=info.NUM_CLASSES
        ).to(device)

    else:
        model = None
        raise ValueError(f"Valor de MODEL_TYPE = '{model_type}' é inválido !")
        

    # Definir Otimizador
    if info.best_params[model_type]["optimizer_name"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=info.best_params[model_type]["learning_rate"], weight_decay=info.best_params[model_type]["weight_decay"])
    elif info.best_params[model_type]["optimizer_name"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=info.best_params[model_type]["learning_rate"], weight_decay=info.best_params[model_type]["weight_decay"])
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=info.best_params[model_type]["learning_rate"], weight_decay=info.best_params[model_type]["weight_decay"])

    criterion = nn.CrossEntropyLoss(weight=info.CLASS_WEIGHTS.to(device), label_smoothing=0.1)

    print(f"{info.print_g(model_type, True)}")
    if model_type == "CNN":
        print(f"{info.print_c(model_type, True)}")
    elif model_type == "LSTM":
        print(f"{info.print_l(model_type, True)}")

    # Configurações do Early Stopping
    patience = info.PATIENCE
    trigger_times = 0
    best_f1_local = 0.0
    best_epoch = 0

    train_loader = DataLoader(info.TRAIN_DATASET, batch_size=info.best_params[model_type]["batch_size"], shuffle=True, pin_memory=True)
    test_loader = DataLoader(info.TEST_DATASET, batch_size=info.best_params[model_type]["batch_size"], shuffle=False, pin_memory=True)
    
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_f1': []}

    # Loop de Treinamento
    for epoch in tqdm(range(info.EPOCHS), desc="Progresso"):
        train_loss = train_model(model, train_loader, criterion, optimizer, model_type)
        val_loss, val_f1 = evaluate_model(model, test_loader, criterion, model_type)
        print(f"[{model_type}] Epoch [{epoch+1}/{info.EPOCHS}] | F1: {val_f1*100:.4f}%")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        if val_f1 > best_f1_local:
            best_f1_local = val_f1
            best_epoch = epoch+1
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"[{model_type}] Early stopping na época {epoch+1}!\n")
                break
    
    print_separator_title(info, "FIM DO TREINAMENTO FINAL", "=")
    print(f"[{model_type}] Melhor época: {best_epoch}! Melhor F1: {best_f1_local:.4f}")

    # A. Salvar Modelo
    torch.save(model.state_dict(), save_path_model)
    print(f"\n[{model_type}] Modelo: {save_path_model}")
    
    # B. Salvar Histórico CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv(save_path_csv, index=False)
    print(f"[{model_type}] Histórico: {save_path_csv}")
    print_separator_end(info, "=")

def print_separator_title(info: InfoConfig, title: str, separator: str):
    l = len(title)
    diff = info.SCREEN_WIDTH - l
    
    if (diff < 0) or len(separator) > 1:
        return 
    
    print(f"\n")
    print(f"{separator}" * (int(diff/2)-1), end='')
    print(f" {title} ", end='')
    print(f"{separator}" * (int(diff/2)-1))
    print()

def print_separator_end(info: InfoConfig, separator: str):
    if len(separator) > 1:
        return 
    
    print(f"\n")
    print(f"{separator}" * info.SCREEN_WIDTH)
    print(f"\n" * 5)
    
def clear_screen():
    # Se for Windows usa 'cls', se for Linux/Mac usa 'clear'
    os.system('cls' if os.name == 'nt' else 'clear')

info = InfoConfig(
    N_TRIALS=50, 
    MODEL_VERSION=100, 
    MODEL_TYPES=["CNN", "LSTM"], 
    DATASET_PATH="data/malicious_phish.csv", 
    LEARNING_RATE=[1e-6, 1e-3], 
    BATCH_SIZE=[32, 64, 128, 256], 
    OPTIMIZER_NAME=["Adam", "SGD", "RMSprop"], 
    EPOCHS=50, 
    EMBED_DIM=[16, 32], 
    DROPOUT=[0.4, 0.6], 
    PATIENCE=10, 
    WEIGHT_DECAY=[1e-4, 1e-3],
    HIDDEN_DIM_LSTM=[16, 32], 
    N_LAYERS_LSTM=[1], 
    N_CONV_LAYERS_CNN=[1], 
    KERNEL_SIZES_CNN=[2, 3], 
    FC_DIM_CNN=[16, 32], 
    FILTER_SIZES_CNN=[16, 32], 
    NR_SAMPLE=None, 
    TEST_SIZE=0.45,
    DB_NAME='db.sqlite',
    SCREEN_WIDTH=shutil.get_terminal_size(fallback=(80, 24)).columns
    )

if __name__ == "__main__":
        while True:
            print(f"Select a option:\n0. Exit\n1. Study Only\n2. Create Model Only\n3. 1 and 2\n4. Change Config")
            try:
                ans = int(input(f"Enter the option number: "))
            except Exception as e:
                clear_screen()
                print(f"Invalid option, try again.")
                continue
            except KeyboardInterrupt as e:
                ans = 0

            if ans == 0:
                print(f"\nBye!")
                break

            if ans == 1 or ans == 3:
                info.load_data()
                for m in info.MODEL_TYPES:
                    study(info, m)
                    # print("Study Here")
                    pass

            if ans == 2 or ans == 3:
                info.load_data()
                for m in info.MODEL_TYPES:
                    create_model(info, m)
                    # print("Create Model Here")
                    pass

            if ans == 4:
                print_separator_title(info, "Config List", "*")
                print(f"{info.print_a()}")
                config_ans = input(f"Write the config name you want to change: ")
                try:
                    value_type = type(getattr(info, config_ans))
                except Exception as e:
                    print(f"Invalid config name, try again.")
                    continue

                new_value_ans = input(f"Enter new value: ")
                try:
                    new_value_ans = value_type(new_value_ans)
                    previous_value = getattr(info, config_ans)
                    setattr(info, config_ans, new_value_ans)
                    print(f"Config changed successfully! {config_ans} changed from '{previous_value}' to '{new_value_ans}'")
                except Exception as e:
                    print(f"Invalid type for new value, try again.")
                    continue

            if ans < 0 or ans > 4:
                clear_screen()
                


# Comando para ver o progresso do treinamento
# optuna-dashboard sqlite:///{nome_do_banco_de_dados}
