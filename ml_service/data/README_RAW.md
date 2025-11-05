## RAW → Processed (UCI Hydraulic)

- RAW dir: `ml_service/data/raw/uci_hydraulic`
- Builder: `ml_service/make_uci_dataset.py`
- Loader: `ml_service/data/uci_raw_loader.py`
- Outputs:
  - Processed parquet: `ml_service/data/processed/uci_hydraulic.parquet`
  - Tidy CSV for training: `ml_service/data/industrial_iot/Industrial_fault_detection.csv`
  - Report: `ml_service/reports/raw_ingestion_report.json`

Run:

```bash
python ml_service/make_uci_dataset.py
python ml_service/train_real_production_models.py --data ml_service/data/industrial_iot/Industrial_fault_detection.csv
```

Notes:
- Aligns all sensors by minimal length, synthetic Timestamp@10Hz
- Applies labels from profile.txt if present; fallback = 0
- Adds lightweight features (RMS VS1, PS*_std_5s)
