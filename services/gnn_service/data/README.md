# Training Dataset for Universal Temporal GNN

Training data structure and preparation scripts.

## Dataset Format

### Temporal Sequence Format
```python
{
  'x_sequence': torch.Tensor,      # [batch, T=12, n_nodes, n_features]
  'edge_index': torch.LongTensor,  # [2, n_edges]
  'health_target': torch.Tensor,   # [batch, n_nodes] range 0-1
  'degradation_target': torch.Tensor,  # [batch, n_nodes] range -1 to 1
}
```

### TimescaleDB Query Example

```sql
-- Extract 60-minute temporal window aggregated by 5-minute buckets
SELECT 
  time_bucket('5 minutes', timestamp) AS bucket,
  component_id,
  sensor_id,
  AVG(value) AS mean_value,
  STDDEV(value) AS std_value,
  MIN(value) AS min_value,
  MAX(value) AS max_value,
  LAST(value, timestamp) AS last_value
FROM sensor_readings
WHERE 
  system_id = 'excavator_001'
  AND timestamp BETWEEN NOW() - INTERVAL '60 minutes' AND NOW()
GROUP BY bucket, component_id, sensor_id
ORDER BY bucket, component_id;
```

### Label Creation

```python
def create_labels(time_to_failure_minutes: float) -> tuple:
    """
    Create health score and degradation rate labels.
    
    Args:
        time_to_failure_minutes: Minutes until component failure
    
    Returns:
        (health_score, degradation_rate)
    """
    # Health score (0 = failed, 1 = healthy)
    health_score = min(1.0, time_to_failure_minutes / 60.0)
    
    # Degradation rate (estimated derivative)
    # Assuming linear degradation over 5-minute window
    degradation_rate = -health_score / (time_to_failure_minutes / 5.0)
    
    return health_score, degradation_rate
```

## Data Preparation

```bash
# Prepare training data from TimescaleDB
python prepare_dataset.py --output data/train.pt

# Split train/val/test
python split_dataset.py --input data/train.pt --train-ratio 0.7 --val-ratio 0.15
```

## Training

```bash
# Start training
python train_universal.py --config config.yaml

# Resume from checkpoint
python train_universal.py --resume models/universal_temporal_latest.ckpt
```

## Directory Structure

```
data/
├── train.pt              # Training dataset
├── val.pt                # Validation dataset
├── test.pt               # Test dataset
└── metadata.json         # System metadata (components, sensors)
```
