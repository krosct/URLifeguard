import torch
import pandas as pd
import argparse
import os
import sys
from utils.utils import url_to_tensor, VOCAB_SIZE
from models.models_optuna import CharCNN, CharLSTM 

# --- CONFIGURAÇÕES ---
MODEL_VERSION = 100
MODEL_TYPE = "CNN"
BENIGN_IDX = 1 
CLASS_MAP = {0: 'Phishing', 1: 'Benign', 2: 'Defacement', 3: 'Malware'}
CSV_PATH_CNN = f"results/optuna_results_{MODEL_TYPE.lower()}_{MODEL_VERSION}.csv"
MODEL_PATH_CNN = f"checkpoints/urlifeguard_FINAL_{MODEL_TYPE.lower()}_{MODEL_VERSION}.pth"
CSV_PATH_LSTM = f"results/optuna_results_{MODEL_TYPE.lower()}_{MODEL_VERSION}.csv"
MODEL_PATH_LSTM = f"checkpoints/urlifeguard_FINAL_{MODEL_TYPE.lower()}_{MODEL_VERSION}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_type: str):
    model_type = model_type.lower()
    """
    Lê o CSV do Optuna para reconstruir a arquitetura exata do modelo
    e depois carrega os pesos.
    """
    csv_path = CSV_PATH_CNN if model_type == "cnn" else CSV_PATH_LSTM
    model_path = MODEL_PATH_CNN if model_type == "cnn" else MODEL_PATH_LSTM

    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ ERRO: Arquivos não encontrados.\nCSV: {csv_path}\nModel: {model_path}")

    try:
        # 1. Ler CSV e pegar a melhor configuração
        df = pd.read_csv(csv_path)
        best_row = df.loc[df['value'].idxmax()] # Pega a linha com maior F1 Score

        # 2. Instanciar o modelo com os parâmetros do CSV
        if model_type == "cnn":
            n_layers = int(best_row['params_n_conv_layers'].item())
            filters_list = [int(best_row[f'params_n_filter_l{i}'].item()) for i in range(n_layers)]
            
            model = CharCNN(
                vocab_size=VOCAB_SIZE,
                embed_dim=int(best_row['params_embed_dim'].item()),
                n_filters=filters_list,
                kernel_sizes=int(best_row['params_kernel_sizes'].item()),
                fc_dim=int(best_row['params_fc_dim'].item()),
                dropout=float(best_row['params_dropout'].item()),
                num_classes=4 
            )
        else: # LSTM
            model = CharLSTM(
                vocab_size=VOCAB_SIZE,
                embed_dim=int(best_row['params_embed_dim'].item()),
                hidden_dim=int(best_row['params_hidden_dim'].item()),
                n_layers=int(best_row['params_n_layers'].item()),
                dropout=float(best_row['params_dropout'].item()),
                num_classes=4
            )

        # 3. Carregar pesos
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    except Exception as e:
        print(f"❌ Erro ao reconstruir o modelo: {e}")
        print("Verifique se o arquivo CSV corresponde ao modelo .pth que você está tentando carregar.")
        sys.exit(1)

def predict(model, url):
    # Processamento
    tensor = url_to_tensor(url, max_len=150).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Pega a classe predita
        conf, pred_idx_tensor = torch.max(probs, 1)
        pred_idx = int(pred_idx_tensor.item())
        
        # Lógica Binária (Benigno vs O Resto)
        is_malicious = (pred_idx != BENIGN_IDX)
        
        # Se for malicioso, a confiança é a soma das chances de ser ruim
        # Ou simplesmente 1 - chance de ser bom
        if is_malicious:
            confidence = 1.0 - probs[0][BENIGN_IDX].item()
        else:
            confidence = probs[0][BENIGN_IDX].item()

        label_specific = CLASS_MAP.get(pred_idx, "Unknown")
        
        return is_malicious, label_specific, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verificador de URL via Terminal")
    parser.add_argument("--url", type=str, required=True, help="A URL para analisar")
    parser.add_argument("--type", type=str, default="CNN", choices=["CNN", "LSTM"], help="Qual modelo usar")
    
    args = parser.parse_args()
    
    # 1. Carrega
    print(f"🔍 Carregando modelo {args.type}...")
    model = load_model(args.type)
    
    # 2. Prediz
    is_malicious, specific_label, conf = predict(model, args.url)
    
    # 3. Exibe Resultado
    print("-" * 30)
    print(f"URL: {args.url}")
    
    if is_malicious:
        # Vermelho para perigo
        print(f"Resultado: \033[91m☣️  MALICIOSO\033[0m")
        print(f"Tipo Detectado: {specific_label}")
    else:
        # Verde para seguro
        print(f"Resultado: \033[92m🛡️  BENIGNO\033[0m")
        
    print(f"Confiança: {conf*100:.2f}%")
    print("-" * 30)