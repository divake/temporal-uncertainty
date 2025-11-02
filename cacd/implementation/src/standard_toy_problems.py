"""
Standard Toy Problems for Uncertainty Quantification
Based on established benchmarks from the literature

References:
1. "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" - Kendall & Gal (2017)
2. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" - Lakshminarayanan et al. (2017)
3. "Accurate Uncertainties for Deep Learning Using Calibrated Regression" - Kuleshov et al. (2018)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class StandardToyProblems:
    """
    Collection of standard toy problems from the uncertainty quantification literature.
    These are well-established benchmarks that clearly separate aleatoric and epistemic uncertainty.
    """

    @staticmethod
    def generate_cubic_with_gaps(n_train: int = 1000,
                                 n_cal: int = 500,
                                 n_test: int = 500,
                                 noise_std: float = 0.3,
                                 gap_regions: list = None,
                                 seed: int = 42) -> Dict:
        """
        1D Cubic function with gaps - Standard benchmark from multiple papers.

        This is used in:
        - Hernández-Lobato & Adams (2015) "Probabilistic Backpropagation"
        - Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"

        Properties:
        - Aleatoric: Homoscedastic Gaussian noise (constant across x)
        - Epistemic: High in gap regions where no training data exists
        - Clear separation between uncertainty types

        Args:
            n_train: Number of training samples
            n_cal: Number of calibration samples
            n_test: Number of test samples
            noise_std: Standard deviation of Gaussian noise (aleatoric uncertainty)
            gap_regions: List of (start, end) tuples defining gaps
            seed: Random seed

        Returns:
            Dictionary with train, calibration, and test data
        """
        np.random.seed(seed)

        if gap_regions is None:
            # Standard gaps from literature
            gap_regions = [(-2.5, -1.5), (0.5, 1.5)]

        def f(x):
            """Cubic function: y = x³"""
            return x**3

        # Generate x values avoiding gaps for training
        x_min, x_max = -4, 4

        # Training data: avoid gaps
        x_train = []
        while len(x_train) < n_train:
            x = np.random.uniform(x_min, x_max)
            in_gap = any(start <= x <= end for start, end in gap_regions)
            if not in_gap:
                x_train.append(x)
        x_train = np.array(x_train)

        # Calibration data: also avoid gaps (same distribution as training)
        x_cal = []
        while len(x_cal) < n_cal:
            x = np.random.uniform(x_min, x_max)
            in_gap = any(start <= x <= end for start, end in gap_regions)
            if not in_gap:
                x_cal.append(x)
        x_cal = np.array(x_cal)

        # Test data: uniform across entire range (including gaps)
        x_test = np.linspace(x_min, x_max, n_test)

        # Generate y values with noise
        y_train = f(x_train) + np.random.normal(0, noise_std, n_train)
        y_cal = f(x_cal) + np.random.normal(0, noise_std, n_cal)
        y_test = f(x_test) + np.random.normal(0, noise_std, n_test)

        # True values without noise
        y_train_true = f(x_train)
        y_cal_true = f(x_cal)
        y_test_true = f(x_test)

        # Mark which test points are in gaps (for analysis)
        in_gap_test = np.array([
            any(start <= x <= end for start, end in gap_regions)
            for x in x_test
        ])

        return {
            'train_x': x_train.reshape(-1, 1),
            'train_y': y_train,
            'train_y_true': y_train_true,
            'cal_x': x_cal.reshape(-1, 1),
            'cal_y': y_cal,
            'cal_y_true': y_cal_true,
            'test_x': x_test.reshape(-1, 1),
            'test_y': y_test,
            'test_y_true': y_test_true,
            'noise_std': noise_std,
            'gap_regions': gap_regions,
            'in_gap_test': in_gap_test,
            'function': 'cubic',
            'description': 'y = x³ with gaps for epistemic uncertainty'
        }

    @staticmethod
    def generate_heteroscedastic_sine(n_train: int = 1000,
                                     n_cal: int = 500,
                                     n_test: int = 500,
                                     seed: int = 42) -> Dict:
        """
        Sine wave with heteroscedastic noise - Standard from uncertainty papers.

        Used in:
        - Kendall & Gal (2017) "What Uncertainties Do We Need..."
        - Lakshminarayanan et al. (2017) "Deep Ensembles"

        Properties:
        - Aleatoric: Heteroscedastic - noise varies with input
        - Epistemic: Low everywhere if well-covered by training data
        - Tests ability to learn input-dependent noise

        Returns:
            Dictionary with train, calibration, and test data
        """
        np.random.seed(seed)

        def f(x):
            """Sine function"""
            return np.sin(2 * np.pi * x / 4)

        def noise_std(x):
            """Input-dependent noise standard deviation"""
            # High noise in middle, low at edges
            return 0.1 + 0.4 * np.exp(-((x - 0) ** 2) / 2)

        # Generate data uniformly
        x_train = np.random.uniform(-4, 4, n_train)
        x_cal = np.random.uniform(-4, 4, n_cal)
        x_test = np.linspace(-4, 4, n_test)

        # True functions
        y_train_true = f(x_train)
        y_cal_true = f(x_cal)
        y_test_true = f(x_test)

        # Add heteroscedastic noise
        noise_train = noise_std(x_train)
        noise_cal = noise_std(x_cal)
        noise_test = noise_std(x_test)

        y_train = y_train_true + np.random.normal(0, 1, n_train) * noise_train
        y_cal = y_cal_true + np.random.normal(0, 1, n_cal) * noise_cal
        y_test = y_test_true + np.random.normal(0, 1, n_test) * noise_test

        return {
            'train_x': x_train.reshape(-1, 1),
            'train_y': y_train,
            'train_y_true': y_train_true,
            'train_noise_std': noise_train,
            'cal_x': x_cal.reshape(-1, 1),
            'cal_y': y_cal,
            'cal_y_true': y_cal_true,
            'cal_noise_std': noise_cal,
            'test_x': x_test.reshape(-1, 1),
            'test_y': y_test,
            'test_y_true': y_test_true,
            'test_noise_std': noise_test,
            'function': 'sine',
            'description': 'sin(2πx/4) with heteroscedastic noise'
        }

    @staticmethod
    def generate_combined_uncertainty(n_train: int = 1000,
                                     n_cal: int = 500,
                                     n_test: int = 500,
                                     seed: int = 42) -> Dict:
        """
        Combined problem with both heteroscedastic noise AND gaps.
        This is ideal for testing uncertainty decomposition.

        Properties:
        - Aleatoric: Varies with x (heteroscedastic)
        - Epistemic: High in gaps, low in data-rich regions
        - Clear orthogonal uncertainty sources

        Returns:
            Dictionary with train, calibration, and test data
        """
        np.random.seed(seed)

        # Define gaps (epistemic source)
        gap_regions = [(1.0, 2.0), (-3.0, -2.0)]

        def f(x):
            """Base function: combination of sine and linear"""
            return 2 * np.sin(x) + 0.2 * x

        def noise_std(x):
            """Heteroscedastic noise (aleatoric source)"""
            # Pattern: high noise for negative x, low for positive
            return 0.15 + 0.35 * (1 / (1 + np.exp(2 * x)))  # Sigmoid decay

        # Generate training and calibration data (avoiding gaps)
        x_min, x_max = -4, 4

        def generate_non_gap_samples(n_samples):
            samples = []
            while len(samples) < n_samples:
                x = np.random.uniform(x_min, x_max)
                in_gap = any(start <= x <= end for start, end in gap_regions)
                if not in_gap:
                    samples.append(x)
            return np.array(samples)

        x_train = generate_non_gap_samples(n_train)
        x_cal = generate_non_gap_samples(n_cal)
        x_test = np.linspace(x_min, x_max, n_test)

        # Generate true values
        y_train_true = f(x_train)
        y_cal_true = f(x_cal)
        y_test_true = f(x_test)

        # Add heteroscedastic noise
        noise_train = noise_std(x_train)
        noise_cal = noise_std(x_cal)
        noise_test = noise_std(x_test)

        y_train = y_train_true + np.random.normal(0, 1, n_train) * noise_train
        y_cal = y_cal_true + np.random.normal(0, 1, n_cal) * noise_cal
        y_test = y_test_true + np.random.normal(0, 1, n_test) * noise_test

        # Mark gap regions for test data
        in_gap_test = np.array([
            any(start <= x <= end for start, end in gap_regions)
            for x in x_test
        ])

        return {
            'train_x': x_train.reshape(-1, 1),
            'train_y': y_train,
            'train_y_true': y_train_true,
            'train_noise_std': noise_train,
            'cal_x': x_cal.reshape(-1, 1),
            'cal_y': y_cal,
            'cal_y_true': y_cal_true,
            'cal_noise_std': noise_cal,
            'test_x': x_test.reshape(-1, 1),
            'test_y': y_test,
            'test_y_true': y_test_true,
            'test_noise_std': noise_test,
            'gap_regions': gap_regions,
            'in_gap_test': in_gap_test,
            'function': 'combined',
            'description': '2sin(x) + 0.2x with heteroscedastic noise and gaps'
        }

    @staticmethod
    def visualize_problem(data: Dict, save_path: Optional[str] = None):
        """
        Comprehensive visualization of the toy problem.
        Shows data distribution, noise patterns, and uncertainty sources.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Data distribution
        ax = axes[0, 0]
        ax.scatter(data['train_x'], data['train_y'], alpha=0.5, s=10, label='Train')
        ax.scatter(data['cal_x'], data['cal_y'], alpha=0.5, s=10, label='Calibration')
        ax.plot(data['test_x'], data['test_y_true'], 'r-', linewidth=2, label='True function')

        if 'gap_regions' in data:
            for start, end in data['gap_regions']:
                ax.axvspan(start, end, alpha=0.2, color='red', label='Gap (epistemic)')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Data Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Noise pattern (if heteroscedastic)
        ax = axes[0, 1]
        if 'train_noise_std' in data:
            ax.plot(data['test_x'], data['test_noise_std'], 'b-', linewidth=2)
            ax.fill_between(data['test_x'].ravel(),
                           data['test_noise_std'] * 0,
                           data['test_noise_std'],
                           alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('Noise σ(x)')
            ax.set_title('Heteroscedastic Noise Pattern (Aleatoric Source)')
        else:
            # Homoscedastic case
            ax.axhline(y=data.get('noise_std', 0.3), color='b', linewidth=2)
            ax.set_xlabel('x')
            ax.set_ylabel('Noise σ')
            ax.set_title(f'Homoscedastic Noise (σ={data.get("noise_std", 0.3)})')
        ax.grid(True, alpha=0.3)

        # Plot 3: Data density (epistemic indicator)
        ax = axes[0, 2]
        train_x_flat = data['train_x'].ravel()
        cal_x_flat = data['cal_x'].ravel()
        all_x = np.concatenate([train_x_flat, cal_x_flat])

        counts, bins = np.histogram(all_x, bins=50)
        centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(centers, counts, width=bins[1]-bins[0], alpha=0.7)

        if 'gap_regions' in data:
            for start, end in data['gap_regions']:
                ax.axvspan(start, end, alpha=0.2, color='red')

        ax.set_xlabel('x')
        ax.set_ylabel('Data Count')
        ax.set_title('Data Density (Epistemic Indicator)')
        ax.grid(True, alpha=0.3)

        # Plot 4: Residuals analysis
        ax = axes[1, 0]
        train_residuals = data['train_y'] - data['train_y_true']
        ax.scatter(data['train_x'], train_residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        if 'train_noise_std' in data:
            ax.plot(data['test_x'], 2*data['test_noise_std'], 'b--', alpha=0.5, label='±2σ')
            ax.plot(data['test_x'], -2*data['test_noise_std'], 'b--', alpha=0.5)

        ax.set_xlabel('x')
        ax.set_ylabel('Residuals')
        ax.set_title('Training Residuals')
        ax.grid(True, alpha=0.3)

        # Plot 5: Expected uncertainty regions
        ax = axes[1, 1]
        ax.plot(data['test_x'], data['test_y_true'], 'g-', linewidth=2, label='True')

        # Show expected aleatoric bounds
        if 'test_noise_std' in data:
            upper = data['test_y_true'] + 2*data['test_noise_std']
            lower = data['test_y_true'] - 2*data['test_noise_std']
            ax.fill_between(data['test_x'].ravel(), lower, upper,
                           alpha=0.3, color='blue', label='Aleatoric (±2σ)')

        # Show epistemic regions
        if 'gap_regions' in data:
            for start, end in data['gap_regions']:
                ax.axvspan(start, end, alpha=0.2, color='red', label='Epistemic (gaps)')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Expected Uncertainty Regions')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Problem summary
        ax = axes[1, 2]
        summary_text = f"Problem: {data.get('description', 'Unknown')}\n\n"
        summary_text += f"Samples:\n"
        summary_text += f"  Train: {len(data['train_x'])}\n"
        summary_text += f"  Calibration: {len(data['cal_x'])}\n"
        summary_text += f"  Test: {len(data['test_x'])}\n\n"

        if 'train_noise_std' in data:
            summary_text += f"Noise type: Heteroscedastic\n"
            summary_text += f"  Min σ: {data['test_noise_std'].min():.3f}\n"
            summary_text += f"  Max σ: {data['test_noise_std'].max():.3f}\n"
        else:
            summary_text += f"Noise type: Homoscedastic\n"
            summary_text += f"  σ: {data.get('noise_std', 'Unknown')}\n"

        if 'gap_regions' in data:
            summary_text += f"\nGap regions: {data['gap_regions']}\n"
            gap_fraction = data['in_gap_test'].mean()
            summary_text += f"Test points in gaps: {gap_fraction:.1%}\n"

        summary_text += f"\nExpected behavior:\n"
        summary_text += f"• Aleatoric: {'Varies with x' if 'train_noise_std' in data else 'Constant'}\n"
        summary_text += f"• Epistemic: {'High in gaps' if 'gap_regions' in data else 'Low everywhere'}"

        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center', family='monospace')
        ax.axis('off')

        plt.suptitle('Toy Problem Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()

        return fig

    @staticmethod
    def analyze_uncertainty_sources(data: Dict) -> Dict:
        """
        Analyze and quantify the uncertainty sources in the data.
        Returns metrics about aleatoric and epistemic separability.
        """
        analysis = {}

        # Analyze aleatoric uncertainty
        if 'train_noise_std' in data:
            # Heteroscedastic case
            analysis['aleatoric_type'] = 'heteroscedastic'
            analysis['aleatoric_range'] = [
                data['test_noise_std'].min(),
                data['test_noise_std'].max()
            ]
            analysis['aleatoric_variation'] = data['test_noise_std'].std()
        else:
            # Homoscedastic case
            analysis['aleatoric_type'] = 'homoscedastic'
            analysis['aleatoric_range'] = [data.get('noise_std', 0), data.get('noise_std', 0)]
            analysis['aleatoric_variation'] = 0

        # Analyze epistemic uncertainty
        if 'gap_regions' in data:
            analysis['epistemic_source'] = 'gaps'
            analysis['gap_fraction'] = data['in_gap_test'].mean()
            analysis['gap_regions'] = data['gap_regions']
        else:
            analysis['epistemic_source'] = 'none'
            analysis['gap_fraction'] = 0
            analysis['gap_regions'] = []

        # Compute separability score
        # High score means aleatoric and epistemic are orthogonal
        if 'train_noise_std' in data and 'in_gap_test' in data:
            # Check if high noise regions overlap with gaps
            test_x = data['test_x'].ravel()
            noise_high = data['test_noise_std'] > np.median(data['test_noise_std'])

            overlap = np.mean(noise_high & data['in_gap_test'])
            analysis['separability_score'] = 1 - overlap  # 1 = perfectly separable
        else:
            analysis['separability_score'] = 1.0  # Perfect if only one source

        return analysis


if __name__ == "__main__":
    # Test the standard problems
    problems = StandardToyProblems()

    print("Testing Standard Toy Problems for CACD\n" + "="*50)

    # Test 1: Cubic with gaps (epistemic focus)
    print("\n1. Cubic with Gaps (Epistemic Uncertainty Focus)")
    data1 = problems.generate_cubic_with_gaps()
    analysis1 = problems.analyze_uncertainty_sources(data1)
    print(f"   Aleatoric: {analysis1['aleatoric_type']}")
    print(f"   Epistemic: {analysis1['epistemic_source']}")
    print(f"   Separability: {analysis1['separability_score']:.2f}")
    problems.visualize_problem(data1,
        '/ssd_4TB/divake/temporal_uncertainty/cacd/implementation/experiments/toy_regression/visualizations/cubic_gaps_problem.png')

    # Test 2: Heteroscedastic sine (aleatoric focus)
    print("\n2. Heteroscedastic Sine (Aleatoric Uncertainty Focus)")
    data2 = problems.generate_heteroscedastic_sine()
    analysis2 = problems.analyze_uncertainty_sources(data2)
    print(f"   Aleatoric: {analysis2['aleatoric_type']}")
    print(f"   Epistemic: {analysis2['epistemic_source']}")
    print(f"   Separability: {analysis2['separability_score']:.2f}")
    problems.visualize_problem(data2,
        '/ssd_4TB/divake/temporal_uncertainty/cacd/implementation/experiments/toy_regression/visualizations/heteroscedastic_sine_problem.png')

    # Test 3: Combined (both uncertainties, orthogonal)
    print("\n3. Combined Problem (Both Uncertainties, Orthogonal)")
    data3 = problems.generate_combined_uncertainty()
    analysis3 = problems.analyze_uncertainty_sources(data3)
    print(f"   Aleatoric: {analysis3['aleatoric_type']}")
    print(f"   Epistemic: {analysis3['epistemic_source']}")
    print(f"   Separability: {analysis3['separability_score']:.2f}")
    problems.visualize_problem(data3,
        '/ssd_4TB/divake/temporal_uncertainty/cacd/implementation/experiments/toy_regression/visualizations/combined_problem.png')

    print("\n" + "="*50)
    print("Recommendation: Use Problem 3 (Combined) for CACD testing")
    print("Reason: Clear orthogonal uncertainty sources that can be decomposed")