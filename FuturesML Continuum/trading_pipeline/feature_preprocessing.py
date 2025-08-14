import numpy as np
import torch

def features_to_numpy(features: dict):
    feature_order = ['log_return', 'sma', 'vwap', 'volatility']
    values = [features.get(f, 0.0) for f in feature_order]
    return np.array(values, dtype=np.float32)

def features_to_torch_tensor(features: dict, device='cpu'):
    np_array = features_to_numpy(features)
    tensor = torch.from_numpy(np_array).float().unsqueeze(0)  # [1, feature_dim]
    return tensor.to(device)

if __name__ == "__main__":
    sample_features = {
        'log_return': 0.00012,
        'sma': 119728.16,
        'vwap': 119728.18,
        'volatility': 4.96e-6
    }

    np_input = features_to_numpy(sample_features)
    print("NumPy input shape:", np_input.shape)
    print(np_input)

    torch_input = features_to_torch_tensor(sample_features)
    print("Torch tensor shape:", torch_input.shape)
    print(torch_input)
