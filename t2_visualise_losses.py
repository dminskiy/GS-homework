import os
import numpy as np
import matplotlib.pyplot as plt

from utils.loss_functions import mape_loss, amape_loss
from utils.data_location import *

if __name__ == "__main__":
    real = np.array([100] * 100)
    predicted = [
        np.linspace(0, 10, len(real)) + np.random.normal(scale=10, size=len(real)),
        np.linspace(0, 20, len(real)) + np.random.normal(scale=10, size=len(real)),
        np.linspace(0, 30, len(real)) + np.random.normal(scale=10, size=len(real)),
        np.linspace(50, 80, len(real)) + np.random.normal(scale=10, size=len(real)),
        np.linspace(60, 90, len(real)) + np.random.normal(scale=10, size=len(real)),
        np.linspace(80, 100, len(real)) + np.random.normal(scale=10, size=len(real)),
        np.linspace(90, 110, len(real)) + np.random.normal(scale=5, size=len(real)),
        np.linspace(95, 105, len(real)) + np.random.normal(scale=5, size=len(real)),
        np.linspace(90, 120, len(real)) + np.random.normal(scale=10, size=len(real)),
        np.linspace(120, 150, len(real)) + np.random.normal(scale=10, size=len(real)),
        np.linspace(150, 200, len(real)) + np.random.normal(scale=10, size=len(real)),
        np.linspace(400, 500, len(real)) + np.random.normal(scale=10, size=len(real)),
    ]

    mape = []
    amape_00 = []
    amape_02 = []
    amape_05 = []
    amape_07 = []
    amape_10 = []

    for predicted_arr in predicted:
        mape.append(mape_loss(real_rates=real, predicted_rates=predicted_arr))
        amape_00.append(
            amape_loss(real_rates=real, predicted_rates=predicted_arr, alpha=0.0)
        )
        amape_02.append(
            amape_loss(real_rates=real, predicted_rates=predicted_arr, alpha=0.2)
        )
        amape_05.append(
            amape_loss(real_rates=real, predicted_rates=predicted_arr, alpha=0.5)
        )
        amape_07.append(
            amape_loss(real_rates=real, predicted_rates=predicted_arr, alpha=0.7)
        )
        amape_10.append(
            amape_loss(real_rates=real, predicted_rates=predicted_arr, alpha=1.0)
        )

    x = list(range(len(predicted)))

    # Plotting the lines
    plt.figure(figsize=(8, 6))

    plt.plot(x, mape, label="MAPE", linestyle="-")
    plt.plot(x, amape_00, label="aMAPE: 0", linestyle="-.")
    plt.plot(x, amape_02, label="aMAPE: 0.2", linestyle="--")
    plt.plot(x, amape_05, label="aMAPE: 0.5", linestyle="--")
    plt.plot(x, amape_07, label="aMAPE: 0.7", linestyle="--")
    plt.plot(x, amape_10, label="aMAPE: 1", linestyle="-.")

    plt.ylabel("Loss, %")
    plt.xlabel("Sample")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(VISIALISATIONS_DIR, "losses.png"))
