"""
Data utilities for Enhanced CACD experiments on UCI datasets.
Handles data loading, splitting, and basic model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from pathlib import Path


class UCIDataLoader:
    """Load and preprocess UCI datasets for CACD experiments."""

    def __init__(self, dataset_name='energy_heating', base_path='/ssd_4TB/divake/temporal_uncertainty/cacd/datasets'):
        self.dataset_name = dataset_name
        self.base_path = Path(base_path)
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        """Load UCI dataset based on dataset_name."""
        if 'energy' in self.dataset_name:
            return self._load_energy()
        elif 'power' in self.dataset_name:
            return self._load_power_plant()
        elif 'concrete' in self.dataset_name:
            return self._load_concrete()
        elif 'yacht' in self.dataset_name:
            return self._load_yacht()
        elif 'wine' in self.dataset_name:
            return self._load_wine()
        elif 'naval' in self.dataset_name:
            return self._load_naval()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _load_energy(self):
        """Load Energy Efficiency dataset."""
        if 'heating' in self.dataset_name:
            data_path = self.base_path / 'energy_heating.csv'
        else:
            data_path = self.base_path / 'energy_cooling.csv'

        data = pd.read_csv(data_path)

        # Features: V1-V8
        feature_cols = [col for col in data.columns if col.startswith('V')]
        X = data[feature_cols].values

        # Target: y1 (already filtered by file)
        y = data['y1'].values if 'y1' in data.columns else data.iloc[:, -1].values

        self.X, self.y = X, y
        return X, y

    def _load_power_plant(self):
        """Load Combined Cycle Power Plant dataset."""
        data_path = self.base_path / 'power_plant.csv'
        data = pd.read_csv(data_path)

        # All columns except the last one are features
        X = data.iloc[:, :-1].values
        # Last column is the target
        y = data.iloc[:, -1].values

        self.X, self.y = X, y
        return X, y

    def _load_concrete(self):
        """Load Concrete Compressive Strength dataset."""
        data_path = self.base_path / 'concrete.csv'
        data = pd.read_csv(data_path)

        # All columns except the last one are features
        X = data.iloc[:, :-1].values
        # Last column is the target (compressive strength)
        y = data.iloc[:, -1].values

        self.X, self.y = X, y
        return X, y

    def _load_yacht(self):
        """Load Yacht Hydrodynamics dataset."""
        data_path = self.base_path / 'yacht.csv'
        data = pd.read_csv(data_path)

        # Features: all except last column
        X = data.iloc[:, :-1].values
        # Target: last column (residuary resistance)
        y = data.iloc[:, -1].values

        self.X, self.y = X, y
        return X, y

    def _load_wine(self):
        """Load Wine Quality dataset."""
        data_path = self.base_path / 'wine_quality_red.csv'
        data = pd.read_csv(data_path)

        # Features: all except last column
        X = data.iloc[:, :-1].values
        # Target: last column
        y = data.iloc[:, -1].values

        self.X, self.y = X, y
        return X, y

    def _load_naval(self):
        """Load Naval Propulsion dataset."""
        data_path = self.base_path / 'uci_naval.csv'
        data = pd.read_csv(data_path, delim_whitespace=True, header=None)

        # Features: first 16 columns
        X = data.iloc[:, :16].values
        # Target: GT Turbine decay state coefficient
        y = data.iloc[:, 16].values

        self.X, self.y = X, y
        return X, y

    def get_splits(self, test_size=0.15, cal_size=0.25, random_state=42):
        """
        Split data into train, calibration, and test sets.

        Args:
            test_size: Fraction for test set
            cal_size: Fraction of remaining data for calibration
            random_state: Random seed

        Returns:
            Dictionary with splits and predictions
        """
        if self.X is None:
            self.load_data()

        # First split: train+cal vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # Second split: train vs cal
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_temp, y_temp, test_size=cal_size, random_state=random_state
        )

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_cal': X_cal,
            'y_cal': y_cal,
            'X_test': X_test,
            'y_test': y_test
        }


class ModelTrainer:
    """Train and save base models for UCI datasets."""

    def __init__(self, model_dir='/ssd_4TB/divake/temporal_uncertainty/cacd/enhanced_cacd/models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_or_load_model(self, X_train, y_train, dataset_name, force_retrain=False):
        """
        Train a model or load existing one.

        Args:
            X_train: Training features
            y_train: Training targets
            dataset_name: Name of dataset for saving
            force_retrain: Force retraining even if model exists

        Returns:
            Trained model and scaler
        """
        model_path = self.model_dir / f'{dataset_name}_rf_model.pkl'
        scaler_path = self.model_dir / f'{dataset_name}_scaler.pkl'

        if not force_retrain and model_path.exists() and scaler_path.exists():
            print(f"Loading existing model from {model_path}")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        else:
            print(f"Training new Random Forest model for {dataset_name}")

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Train Random Forest
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            # Save model and scaler
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            print(f"Model saved to {model_path}")

        return model, scaler

    def get_predictions(self, model, scaler, X):
        """Get predictions from model."""
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)


def prepare_dataset(dataset_name, force_retrain=False):
    """
    Convenience function to prepare a complete dataset.

    Args:
        dataset_name: Name of UCI dataset
        force_retrain: Force model retraining

    Returns:
        Dictionary with all data, model, and predictions
    """
    # Load data
    loader = UCIDataLoader(dataset_name)
    loader.load_data()

    # Get splits
    splits = loader.get_splits()

    # Train model
    trainer = ModelTrainer()
    model, scaler = trainer.train_or_load_model(
        splits['X_train'],
        splits['y_train'],
        dataset_name,
        force_retrain
    )

    # Get predictions
    y_pred_train = trainer.get_predictions(model, scaler, splits['X_train'])
    y_pred_cal = trainer.get_predictions(model, scaler, splits['X_cal'])
    y_pred_test = trainer.get_predictions(model, scaler, splits['X_test'])

    # Scale features for distance computations
    X_train_scaled = scaler.transform(splits['X_train'])
    X_cal_scaled = scaler.transform(splits['X_cal'])
    X_test_scaled = scaler.transform(splits['X_test'])

    return {
        # Raw data
        'X_train': splits['X_train'],
        'y_train': splits['y_train'],
        'X_cal': splits['X_cal'],
        'y_cal': splits['y_cal'],
        'X_test': splits['X_test'],
        'y_test': splits['y_test'],
        # Scaled features
        'X_train_scaled': X_train_scaled,
        'X_cal_scaled': X_cal_scaled,
        'X_test_scaled': X_test_scaled,
        # Predictions
        'y_pred_train': y_pred_train,
        'y_pred_cal': y_pred_cal,
        'y_pred_test': y_pred_test,
        # Model and scaler
        'model': model,
        'scaler': scaler,
        # Metadata
        'dataset_name': dataset_name,
        'n_features': splits['X_train'].shape[1]
    }