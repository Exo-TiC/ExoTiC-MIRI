import sys
import numpy as np


a = np.ones((417, 72, 10000))
print(sys.getsizeof(a)/1e9)

