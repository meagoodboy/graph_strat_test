import torch

def calculate_num_params(model):
  pytorch_total_params = sum(p.numel() for p in model.parameters())
  return pytorch_total_params

def clear_cache():
    torch.cuda.empty_cache()