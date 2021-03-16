from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import cv2


def visualize(
    config: List[Dict[str, str]],
    sample_size: int = 30,
    num_cols: int = 5
) -> None:
    """Visualize some sample from config file."""
    num_rows = sample_size // num_cols + int(sample_size % num_cols != 0)
    random_idxs = np.random.choice(len(config), size=sample_size, replace=False)
    plt.figure(figsize=(20, 2 * num_rows))
    for i, idx in enumerate(random_idxs, 1):
        item = config[idx]
        text = item["text"]
        image = cv2.imread(item["file"])

        plt.subplot(num_rows, num_cols, i)
        plt.imshow(image[:, :, ::-1])
        plt.title(text)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
