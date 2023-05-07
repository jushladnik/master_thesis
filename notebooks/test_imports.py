import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import random


if __name__ == "__main__":

    ljudje = {}
    for i in range(1000):

        ljudje['ime'].append(''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N)))
        ljudje['priimek'].append(''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N)))
        ljudje['starost'].append(int(np.randint(1, 120)))

    print(ljudje)

