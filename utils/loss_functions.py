import numpy as np


def mape_loss(real_rates: np.ndarray, predicted_rates: np.ndarray) -> float:
    return np.average(abs(predicted_rates / real_rates - 1.0)) * 100.0


def amape_loss(
    real_rates: np.ndarray, predicted_rates: np.ndarray, alpha: float
) -> float:
    assert 0 <= alpha <= 1, f"Alpha out of range: {alpha}. Valid range: [0;1]"

    diff = real_rates - predicted_rates
    over_ind = np.where(diff < 0)[0]
    under_ind = np.where(diff >= 0)[0]

    over = (
        np.abs(diff[over_ind]) / real_rates[over_ind] if over_ind.shape[0] > 0 else 0.0
    )
    under = diff[under_ind] / real_rates[under_ind] if under_ind.shape[0] > 0 else 0.0

    return (alpha * np.average(over) + (1 - alpha) * np.average(under)) * 100.0
