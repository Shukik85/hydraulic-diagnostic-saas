"""
Data preparation script for hydraulic diagnostics GNN.
Updated to match actual column names in the dataset.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from config import model_config, physical_norms, training_config
from torch_geometric.data import Data

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BIMDataPreprocessor:
    """Preprocess BIM dataset with actual column names."""

    def __init__(self):
        self.norms = physical_norms
        self.config = model_config

    def _get_actual_column_mapping(self) -> dict[str, list[str]]:
        """Get actual column names based on the dataset structure."""
        return {
            "pump": [
                "pump_pressure_outlet",
                "pump_speed_rpm",
                "pump_temperature",
                "pump_vibration",
                "pump_power",
            ],
            "cylinder_boom": [
                "boom_pressure_extend",
                "boom_pressure_retract",
                "boom_position",
                "boom_velocity",
                "boom_pressure_diff",
            ],
            "cylinder_stick": [
                "stick_pressure_extend",
                "stick_pressure_retract",
                "stick_position",
                "stick_velocity",
                "stick_pressure_diff",
            ],
            "cylinder_bucket": [
                "bucket_pressure_extend",
                "bucket_pressure_retract",
                "bucket_position",
                "bucket_velocity",
                "bucket_pressure_diff",
            ],
            "motor_swing": [
                "swing_speed_rpm",
                "swing_torque",
                "swing_temperature",
                "swing_pressure_inlet",
                "swing_vibration",
            ],
            "motor_left": [
                "left_speed_rpm",
                "left_torque",
                "left_temperature",
                "left_pressure_inlet",
                "left_vibration",
            ],
            "motor_right": [
                "right_speed_rpm",
                "right_torque",
                "right_temperature",
                "right_pressure_inlet",
                "right_vibration",
            ],
        }

    def _get_fault_columns(self) -> dict[str, str]:
        """Get actual fault column names."""
        return {
            "pump": "fault_pump",
            "cylinder_boom": "fault_cylinder_boom",
            "cylinder_stick": "fault_cylinder_stick",
            "cylinder_bucket": "fault_cylinder_bucket",
            "motor_swing": "fault_motor_swing",
            "motor_left": "fault_motor_left",
            "motor_right": "fault_motor_right",
        }

    def _normalize_feature(self, value: float, component: str, feature: str) -> float:
        """Normalize feature using corrected physical norms."""
        try:
            # Map feature names to norm keys
            feature_map = {
                "pressure_outlet": "pressure_outlet",
                "speed_rpm": "speed_rpm",
                "temperature": "temperature",
                "vibration": "vibration",
                "power": "power",
                "pressure_extend": "pressure_extend",
                "pressure_retract": "pressure_retract",
                "position": "position",
                "velocity": "velocity",
                "pressure_diff": "pressure_diff",
                "torque": "torque",
                "pressure_inlet": "pressure_inlet",
            }

            norm_key = feature_map.get(feature, feature)
            norms = getattr(self.norms, component.upper())[norm_key]

            if feature in ["position"]:  # Percentage
                return value / 100.0

            nominal = norms["nominal"]
            min_val = norms.get("min", nominal * 0.7)
            max_val = norms.get("max", nominal * 1.3)

            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
            else:
                normalized = 0.5

            return np.clip(normalized, 0.0, 1.0)

        except (AttributeError, KeyError) as e:
            logger.warning(f"Normalization error for {component}.{feature}: {e}")
            return 0.5

    def _calculate_deviation(self, value: float, component: str, feature: str) -> float:
        """Calculate deviation from nominal value."""
        try:
            feature_map = {
                "pressure_outlet": "pressure_outlet",
                "speed_rpm": "speed_rpm",
                "temperature": "temperature",
                "vibration": "vibration",
                "power": "power",
                "pressure_extend": "pressure_extend",
                "pressure_retract": "pressure_retract",
                "position": "position",
                "velocity": "velocity",
                "pressure_diff": "pressure_diff",
                "torque": "torque",
                "pressure_inlet": "pressure_inlet",
            }

            norm_key = feature_map.get(feature, feature)
            norms = getattr(self.norms, component.upper())[norm_key]
            nominal = norms["nominal"]

            if feature in ["position"]:  # Percentage
                return abs(value - nominal) / 100.0

            deviation = abs(value - nominal) / nominal if nominal > 0 else abs(value)

            return np.clip(deviation, 0.0, 2.0)

        except (AttributeError, KeyError) as e:
            logger.warning(
                f"Deviation calculation error for {component}.{feature}: {e}"
            )
            return 0.0

    def _extract_feature_name(self, column_name: str, component: str) -> str:
        """Extract feature name from column name."""
        # Remove component prefix to get feature name
        prefixes = {
            "pump": "pump_",
            "cylinder_boom": "boom_",
            "cylinder_stick": "stick_",
            "cylinder_bucket": "bucket_",
            "motor_swing": "swing_",
            "motor_left": "left_",
            "motor_right": "right_",
        }

        prefix = prefixes[component]
        if column_name.startswith(prefix):
            return column_name[len(prefix) :]
        return column_name

    def _create_node_features(self, row: pd.Series, component: str) -> list[float]:
        """Create node features for a component."""
        features = []
        column_mapping = self._get_actual_column_mapping()

        for column_name in column_mapping[component]:
            try:
                raw_value = float(row[column_name])
                feature_name = self._extract_feature_name(column_name, component)

                # Raw value
                features.append(raw_value)

                # Normalized value
                norm_value = self._normalize_feature(raw_value, component, feature_name)
                features.append(norm_value)

                # Deviation indicator
                deviation = self._calculate_deviation(
                    raw_value, component, feature_name
                )
                features.append(deviation)

            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error processing {column_name}: {e}")
                # Use zeros for missing values
                features.extend([0.0, 0.5, 0.0])

        return features

    def _create_edge_index(self) -> torch.Tensor:
        """Create graph topology: pump connected to all actuators."""
        edge_list = []

        # Pump (node 0) connected to all other nodes (1-6)
        for target_node in range(1, 7):
            edge_list.append([0, target_node])  # Pump -> Actuator
            edge_list.append([target_node, 0])  # Actuator -> Pump

        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def _create_multi_label_target(self, row: pd.Series) -> torch.Tensor:
        """Create multi-label target vector [7] for 7 components."""
        target = []
        fault_columns = self._get_fault_columns()

        for component in self.config.component_names:
            fault_column = fault_columns[component]
            try:
                fault_value = row[fault_column]
                target.append(int(float(fault_value) > 0))
            except (KeyError, ValueError):
                logger.warning(f"Missing or invalid fault column: {fault_column}")
                target.append(0)  # Assume normal operation

        return torch.tensor(target, dtype=torch.float)

    def load_data(self) -> pd.DataFrame:
        """Load BIM dataset."""
        try:
            df = pd.read_csv(training_config.data_path)
            logger.info(f"Loaded dataset with shape: {df.shape}")

            # Validate that we have the required columns
            all_required_columns = []
            column_mapping = self._get_actual_column_mapping()
            fault_columns = self._get_fault_columns()

            for component_columns in column_mapping.values():
                all_required_columns.extend(component_columns)
            all_required_columns.extend(fault_columns.values())

            missing_columns = set(all_required_columns) - set(df.columns)
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
            else:
                logger.info("All required columns are present")

            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def create_graph_data(self, df: pd.DataFrame) -> list[Data]:
        """Create PyG Data objects from DataFrame."""
        graphs = []
        edge_index = self._create_edge_index()

        for idx, row in df.iterrows():
            try:
                # Node features for all 7 components
                node_features = []
                for component in self.config.component_names:
                    features = self._create_node_features(row, component)
                    node_features.append(features)

                x = torch.tensor(node_features, dtype=torch.float)
                y = self._create_multi_label_target(row)

                graph_data = Data(x=x, edge_index=edge_index, y=y)
                graphs.append(graph_data)

                if idx % 10000 == 0 and idx > 0:
                    logger.info(f"Processed {idx} rows...")

            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue

        logger.info(f"Created {len(graphs)} graph objects")
        return graphs

    def calculate_statistics(self, graphs: list[Data]) -> dict:
        """Calculate statistics for each component."""
        if not graphs:
            return {}

        stats = {}

        for i, component in enumerate(self.config.component_names):
            try:
                component_features = [graph.x[i] for graph in graphs]
                features_tensor = torch.stack(component_features)

                stats[component] = {
                    "mean": features_tensor.mean(dim=0).tolist(),
                    "std": features_tensor.std(dim=0).tolist(),
                    "min": features_tensor.min(dim=0)[0].tolist(),
                    "max": features_tensor.max(dim=0)[0].tolist(),
                }
            except Exception as e:
                logger.warning(f"Error calculating stats for {component}: {e}")

        return stats

    def save_data(self, graphs: list[Data], stats: dict):
        """Save processed data and metadata."""
        Path(training_config.graphs_save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(training_config.metadata_path).parent.mkdir(parents=True, exist_ok=True)

        # Save graphs
        torch.save(graphs, training_config.graphs_save_path)

        # Save metadata
        metadata = {
            "num_graphs": len(graphs),
            "node_features_dim": self.config.num_node_features,
            "num_classes": self.config.num_classes,
            "component_names": self.config.component_names,
            "column_mapping": self._get_actual_column_mapping(),
            "fault_columns": self._get_fault_columns(),
            "physical_norms": {
                component: getattr(self.norms, component.upper())
                for component in self.config.component_names
            },
            "statistics": stats,
        }

        with open(training_config.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved {len(graphs)} graphs to {training_config.graphs_save_path}")
        logger.info(f"Saved metadata to {training_config.metadata_path}")


def main():
    """Main data preparation pipeline."""
    try:
        preprocessor = BIMDataPreprocessor()

        # Load data
        df = preprocessor.load_data()

        # Create graph data
        graphs = preprocessor.create_graph_data(df)

        if not graphs:
            raise ValueError("No graphs were created!")

        # Calculate statistics
        stats = preprocessor.calculate_statistics(graphs)

        # Save data
        preprocessor.save_data(graphs, stats)

        logger.info("Data preparation completed successfully")

    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()
