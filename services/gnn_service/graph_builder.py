"""
Graph Builder: строит node features и edge index по произвольному equipment_metadata.json
- Любая топология (adjacency_matrix), любые признаки на компонент
- Выход: node_features, edge_index (PyG), labels
"""
import numpy as np
import torch

def build_node_features(data_df, metadata):
    """Формирует node features [n_nodes, n_features] динамически по sensors"""
    feats = []
    for comp in metadata['components']:
        node_feat = []
        for sensor in comp['sensors']:
            col = f"{comp['id']}_{sensor}"
            vals = data_df[col].values if col in data_df else np.zeros(len(data_df))
            # Добавляем статистики: mean, std, min, max, last
            node_feat.extend([
                np.mean(vals),
                np.std(vals),
                np.min(vals),
                np.max(vals),
                vals[-1] if len(vals) else 0.0,
            ])
        feats.append(node_feat)
    return torch.tensor(feats, dtype=torch.float)

def adjacency_to_edge_index(adjacency_matrix):
    """matrix [n,n] -> edge_index [2, num_edges] для PyG"""
    sources, targets = np.where(np.array(adjacency_matrix) == 1)
    edge_index = np.stack([sources, targets], axis=0)
    return torch.tensor(edge_index, dtype=torch.long)

# Пример использования (при интеграции c inference_service)
# metadata = load_metadata()
# data_df = pd.read_csv(...)
# node_features = build_node_features(data_df, metadata)
# edge_index = adjacency_to_edge_index(metadata['adjacency_matrix'])
