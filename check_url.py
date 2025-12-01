import torch
import argparse
import os
from utils.utils import url_to_tensor, decode_tensor, VOCAB_SIZE
from models.models import CharCNN, CharLSTM

# Configuração do Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

def load_model(model_path, model_type="CNN"):
    """
    Carrega o modelo treinado (.pth) e prepara para inferência.
    """
    if not os.path.exists(model_path):
        print(f"ERRO: Modelo não encontrado em {model_path}")
        return None

    print(f"Carregando modelo {model_type} de {model_path}...")
    
    # Instanciar a arquitetura correta
    # Nota: num_classes deve ser o mesmo usado no treino (geralmente 2)
    if model_type == "CNN":
        model = CharCNN(vocab_size=VOCAB_SIZE, num_classes=2)
    elif model_type == "LSTM":
        model = CharLSTM(vocab_size=VOCAB_SIZE, num_classes=2)
    else:
        print("Tipo de modelo desconhecido.")
        return None
        
    # Carregar os pesos treinados
    # map_location garante que carregue na CPU se não houver GPU disponível
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Modo de avaliação (desliga dropout, etc)
    
    return model

def predict_url(model, url_string):
    """
    Recebe uma string de URL e retorna a predição (Benigno/Malicioso)
    e a confiança (probabilidade).
    """
    # Pré-processamento: Converter string para tensor
    # Adiciona dimensão do batch (1, 150)
    tensor = url_to_tensor(url_string, max_len=150).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        # Aplicar Softmax para ter probabilidades (0 a 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Pegar a classe com maior probabilidade
        confidence, predicted_class = torch.max(probs, 1)
        
    return predicted_class.item(), confidence.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="URLifeguard: Verificador de URLs")
    parser.add_argument("--url", type=str, help="URL para verificar (ex: www.google.com)")
    parser.add_argument("--model_path", type=str, default="checkpoints/urlifeguard_cnn.pth", help="Caminho para o arquivo .pth do modelo")
    parser.add_argument("--type", type=str, default="CNN", choices=["CNN", "LSTM"], help="Tipo do modelo (CNN ou LSTM)")
    
    args = parser.parse_args()
    
    # Carregar o cérebro
    model = load_model(args.model_path, args.type)
    
    if model:
        # Se o usuário passou uma URL via linha de comando
        if args.url:
            urls_to_check = [args.url]
        else:
            # Modo interativo
            print("\n--- URLifeguard Firewall Simulado ---")
            print("Digite uma URL para verificar (ou 'q' para sair)")
            urls_to_check = []
            while True:
                user_input = input("\nURL > ")
                if user_input.lower() == 'q':
                    break
                
                label, conf = predict_url(model, user_input)
                
                # Mapeamento (0 = Benigno, 1 = Malicioso - ajuste conforme seu treino!)
                status = "🛡️ SEGURA" if label == 0 else "☣️ MALICIOSA"
                color = "\033[92m" if label == 0 else "\033[91m" # Verde ou Vermelho
                reset = "\033[0m"
                
                print(f"Resultado: {color}{status}{reset}")
                print(f"Confiança: {conf*100:.2f}%")

        # Caso tenha passado argumento --url, roda só uma vez
        if args.url:
            label, conf = predict_url(model, args.url)
            status = "🛡️ SEGURA" if label == 0 else "☣️ MALICIOSA"
            color = "\033[92m" if label == 0 else "\033[91m" # Verde ou Vermelho
            reset = "\033[0m"
            
            print(f"Resultado: {color}{status}{reset}")
            print(f"Confiança: {conf*100:.2f}%")