"""
Base classes for uncertainty estimation with save/load functionality.
Provides common interface for aleatoric and epistemic uncertainty models.
"""

import numpy as np
import joblib
from pathlib import Path
from abc import ABC, abstractmethod
import json


class UncertaintyModel(ABC):
    """Base class for all uncertainty models with save/load capability."""

    def __init__(self, name="uncertainty_model"):
        self.name = name
        self.is_fitted = False
        self.metadata = {}

    @abstractmethod
    def fit(self, X_cal, y_cal, y_pred_cal, **kwargs):
        """Fit the uncertainty model on calibration data."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Predict uncertainty for test samples."""
        pass

    def save(self, filepath):
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        model_data = {
            'model': self,
            'metadata': self.metadata,
            'name': self.name,
            'class': self.__class__.__name__
        }
        joblib.dump(model_data, filepath)
        print(f"Saved {self.name} to {filepath}")

        # Also save metadata as JSON for easy inspection
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'w') as f:
            # Convert numpy types to native Python types for JSON
            clean_metadata = {}
            for key, value in self.metadata.items():
                if isinstance(value, np.ndarray):
                    clean_metadata[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    clean_metadata[key] = float(value)
                else:
                    clean_metadata[key] = value

            json.dump({
                'name': self.name,
                'class': self.__class__.__name__,
                'metadata': clean_metadata
            }, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load a saved model from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)
        model = model_data['model']
        model.metadata = model_data.get('metadata', {})

        return model

    def get_info(self):
        """Get information about the model."""
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'metadata': self.metadata
        }


class CombinedUncertaintyModel(UncertaintyModel):
    """Container for combined aleatoric and epistemic models."""

    def __init__(self, aleatoric_model=None, epistemic_model=None, name="combined_uncertainty"):
        super().__init__(name)
        self.aleatoric_model = aleatoric_model
        self.epistemic_model = epistemic_model

    def fit(self, X_cal, y_cal, y_pred_cal, **kwargs):
        """Fit both uncertainty models."""
        if self.aleatoric_model:
            self.aleatoric_model.fit(X_cal, y_cal, y_pred_cal, **kwargs)
        if self.epistemic_model:
            self.epistemic_model.fit(X_cal, y_cal, y_pred_cal, **kwargs)
        self.is_fitted = True

        # Store calibration info
        self.metadata['n_cal'] = len(y_cal)
        self.metadata['n_features'] = X_cal.shape[1]

    def predict(self, X_test):
        """Predict both uncertainties."""
        results = {}
        if self.aleatoric_model:
            results['aleatoric'] = self.aleatoric_model.predict(X_test)
        if self.epistemic_model:
            results['epistemic'] = self.epistemic_model.predict(X_test)
        return results

    def predict_combined(self, X_test):
        """Predict combined uncertainty (for conformal prediction)."""
        results = self.predict(X_test)

        # Combine using root sum of squares
        aleatoric = results.get('aleatoric', np.zeros(len(X_test)))
        epistemic = results.get('epistemic', np.zeros(len(X_test)))

        combined = np.sqrt(aleatoric**2 + epistemic**2)
        return combined

    def save_both(self, directory):
        """Save both models separately."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.aleatoric_model:
            self.aleatoric_model.save(directory / 'aleatoric_model.pkl')
        if self.epistemic_model:
            self.epistemic_model.save(directory / 'epistemic_model.pkl')

        # Save combined metadata
        self.save(directory / 'combined_model.pkl')

    @classmethod
    def load_both(cls, directory):
        """Load both models from directory."""
        directory = Path(directory)

        combined = cls()

        aleatoric_path = directory / 'aleatoric_model.pkl'
        if aleatoric_path.exists():
            # Import dynamically to avoid circular dependency
            from aleatoric import EnhancedAleatoric
            combined.aleatoric_model = EnhancedAleatoric.load(aleatoric_path)

        epistemic_path = directory / 'epistemic_model.pkl'
        if epistemic_path.exists():
            # Import dynamically
            from epistemic import MultiSourceEpistemic
            combined.epistemic_model = MultiSourceEpistemic.load(epistemic_path)

        combined_path = directory / 'combined_model.pkl'
        if combined_path.exists():
            meta_model = cls.load(combined_path)
            combined.metadata = meta_model.metadata
            combined.name = meta_model.name
            combined.is_fitted = True

        return combined