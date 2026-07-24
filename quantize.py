import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
data = np.array(data)

labels = data[:, 0]
pixels = data[:, 1:]

quantized_pixels = np.select(
    [pixels <= 43, pixels <= 128, pixels <= 212],
    [0,86,172],
    default=255
)

quantized_data = np.column_stack((labels, quantized_pixels))

pd.DataFrame(quantized_data).to_csv('train_quantized.csv', index=False, header=False)