import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import cauchy
if __name__ == '__main__':
    x = cauchy.rvs(loc=100, scale=10, size=1000)
    b = x[0]
    print(x)
    plt.scatter(np.arange(1000),x)
    plt.show()