import torch
import sys

print(f"Versão do Python: {sys.version}")
print(f"Versão do PyTorch: {torch.__version__}")
print(f"CUDA disponível? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Versão do CUDA suportada pelo Torch: {torch.version.cuda}")
    print(f"Placa de vídeo detectada: {torch.cuda.get_device_name(0)}")
    print(f"Quantidade de dispositivos: {torch.cuda.device_count()}")
else:
    print("❌ O PyTorch NÃO está usando a GPU. Ele está rodando na CPU.")

### --- TRASH --- ###

# import sys
# import os

# # Obtém o caminho absoluto do diretório atual do notebook
# current_dir = os.getcwd()

# # Obtém o caminho do diretório pai (a pasta URLifeguard)
# project_root = os.path.dirname(current_dir)

# # Adiciona o diretório pai ao sys.path se ele ainda não estiver lá
# if project_root not in sys.path:
#     sys.path.append(project_root)

# # --- Agora você pode importar ---

# # Supondo que seu arquivo seja utils/processamento.py
# from utils import processamento 

# # Ou se você tem funções específicas dentro dele
# from utils.processamento import minha_funcao_tokenizacao

# # Carrega a extensão de autoreload
# %load_ext autoreload

# # Configura para recarregar módulos automaticamente antes de executar o código
# %autoreload 2

# import sys
# import os
# # ... resto do código de importação ...