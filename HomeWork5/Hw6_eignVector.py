from numpy import linalg as LA
import numpy as np

w, v = LA.eigh(np.matrix([
                        [91.43, 171.92, 297.99],
                        [171.92, 373.92, 545.21],
                        [171.92, 297.99, 1297.26],
                       ]), UPLO='U')
print w, v
