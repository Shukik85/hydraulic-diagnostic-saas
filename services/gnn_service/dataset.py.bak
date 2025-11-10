"""HydraulicGraphDataset for loading time-series graph data from TimescaleDB."""
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Tuple, Optional
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import pandas as pd

from .config import config


class HydraulicGraphDataset(Dataset):
    """Dataset for hydraulic system graphs with temporal data."""
    
    def __init__(
        self,
        root: str,
        equipment_ids: List[str],
        start_date: str,
        end_date: str,
        transform=None,
        pre_transform=None,
    ):
        self.equipment_ids = equipment_ids
        self.start_date = start_date
        self.end_date = end_date
        self.engine = sa.create_engine(config.postgres_uri)
        self.Session = sessionmaker(bind=self.engine)
        
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self) -> List[str]:
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        return [f"data_{i}.pt" for i in range(len(self.equipment_ids))]
    
    def download(self):
        pass
    
    def process(self):
        """Load data from TimescaleDB and convert to PyG Data objects."""
        for idx, equipment_id in enumerate(self.equipment_ids):
            # Load metadata (graph structure)
            metadata = self._load_metadata(equipment_id)
            
            # Load time-series sensor data
            sensor_data = self._load_sensor_data(equipment_id)
            
            # Convert to PyG Data
            data = self._create_graph_data(metadata, sensor_data)
            
            torch.save(data, self.processed_paths[idx])
    
    def len(self) -> int:
        return len(self.equipment_ids)
    
    def get(self, idx: int) -> Data:
        data = torch.load(self.processed_paths[idx])
        return data
    
    def _load_metadata(self, equipment_id: str) -> dict:
        """Load component metadata and topology from database."""
        with self.Session() as session:
            query = sa.text("""
                SELECT 
                    equipment_id,
                    components,
                    adjacency_matrix,
                    duty_cycle
                FROM system_metadata
                WHERE equipment_id = :equipment_id
            """)
            result = session.execute(query, {"equipment_id": equipment_id}).fetchone()
            
            if not result:
                raise ValueError(f"No metadata found for equipment {equipment_id}")
            
            return {
                "equipment_id": result[0],
                "components": result[1],  # JSON
                "adjacency_matrix": np.array(result[2]),  # 2D array
                "duty_cycle": result[3]  # JSON
            }
    
    def _load_sensor_data(self, equipment_id: str) -> pd.DataFrame:
        """Load time-series sensor data from TimescaleDB hypertable."""
        query = f"""
            SELECT 
                time,
                component_id,
                pressure,
                temperature,
                flow_rate,
                vibration,
                label
            FROM sensor_readings
            WHERE equipment_id = '{equipment_id}'
              AND time BETWEEN '{self.start_date}' AND '{self.end_date}'
            ORDER BY time ASC
        """
        
        df = pd.read_sql(query, self.engine)
        return df
    
    def _create_graph_data(
        self, 
        metadata: dict, 
        sensor_data: pd.DataFrame
    ) -> Data:
        """Convert metadata + sensor data to PyG Data object."""
        # Node features: aggregate sensor data per component
        components = metadata["components"]
        num_nodes = len(components)
        
        # Extract features (mean, std, min, max, recent_trend)
        node_features = []
        for comp in components:
            comp_data = sensor_data[sensor_data["component_id"] == comp["id"]]
            
            if len(comp_data) > 0:
                features = self._extract_features(comp_data)
            else:
                features = np.zeros(config.num_node_features)
            
            node_features.append(features)
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Edge index from adjacency matrix
        adj_matrix = metadata["adjacency_matrix"]
        edge_index = torch.tensor(
            np.array(np.where(adj_matrix > 0)), 
            dtype=torch.long
        )
        
        # Edge attributes (connection type: pressure=1, return=2, pilot=3)
        edge_attr = []
        for i, j in edge_index.t().tolist():
            # Extract connection type from metadata
            conn_type = components[i].get("connection_types", {}).get(components[j]["id"], "pressure_line")
            type_encoding = {"pressure_line": 1, "return_line": 2, "pilot_line": 3}
            edge_attr.append([type_encoding.get(conn_type, 1)])
        
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Label: 0 = normal, 1 = anomaly
        # Majority vote from sensor_data labels
        if len(sensor_data) > 0 and "label" in sensor_data.columns:
            label = int(sensor_data["label"].mode()[0])
        else:
            label = 0
        
        y = torch.tensor([label], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    def _extract_features(self, comp_data: pd.DataFrame) -> np.ndarray:
        """Extract statistical features from component sensor data."""
        features = []
        
        for col in ["pressure", "temperature", "flow_rate", "vibration"]:
            if col in comp_data.columns:
                values = comp_data[col].dropna()
                if len(values) > 0:
                    features.extend([
                        values.mean(),
                        values.std(),
                        values.min(),
                        values.max(),
                        values.iloc[-10:].mean() - values.iloc[:10].mean()  # trend
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        # Pad/truncate to num_node_features
        features = np.array(features[:config.num_node_features])
        if len(features) < config.num_node_features:
            features = np.pad(features, (0, config.num_node_features - len(features)))
        
        return features
