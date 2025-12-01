import torch
import string

# Definimos o vocabulário: letras, números e símbolos comuns em URLs
# Isto inclui: abc...XYZ...012...!@#...
ALL_CHARS = string.ascii_letters + string.digits + string.punctuation
VOCAB_SIZE = len(ALL_CHARS) + 1  # +1 para o token de preenchimento (padding/unknown)

# Mapeamento de caractere para índice
char_to_idx = {ch: i + 1 for i, ch in enumerate(ALL_CHARS)}

def url_to_tensor(url, max_len=150):
    """
    Converte uma string de URL numa sequência de índices numéricos (Tensor).
    Aplica padding (preenchimento) ou truncagem para garantir tamanho fixo.
    """
    tensor = torch.zeros(max_len, dtype=torch.long)
    
    # Converter cada char para o seu índice
    indices = [char_to_idx.get(c, 0) for c in url] # 0 se o char não estiver no vocab
    
    # Truncar se for maior que max_len
    if len(indices) > max_len:
        indices = indices[:max_len]
        
    # Preencher o tensor
    for i, idx in enumerate(indices):
        tensor[i] = idx
        
    return tensor

def decode_tensor(tensor):
    """
    Função auxiliar para converter tensor de volta para string (para debug).
    """
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    chars = [idx_to_char.get(idx.item(), '') for idx in tensor if idx.item() > 0]
    return "".join(chars)