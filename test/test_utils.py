import numpy as np
import csv
from tqdm import tqdm


def create_random_vector_csv(filepath: str, n: int, m: int):
    with open(filepath, "w") as f:
        writer = csv.writer(f)
        for i in tqdm(range(0, n)):
            v = np.random.randint(low=0, high=n, size=m)
            writer.writerow(v)
