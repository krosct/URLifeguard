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