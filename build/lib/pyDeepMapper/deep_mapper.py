import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

class DeepMapper:
    """Transform features to an image matrix by direct mapping

    This class takes in data normalises if necessary,  and converts it to a
    CNN compatible 'image' matrix

    """

    def __init__() -> None:
        """Generate a DeepMapper instance
        """


   
    def map(X: np.ndarray, normalise:bool = False, plot: int = 0)
        size,xsize = X.shape
        dims = int(math.sqrt(xsize)
        pad_size = dims*dims-xsize
        X_p = np.pad(X, pad_width=((0,0), (0, pad_size)))
        X_img = np.resize(X_p,(sz,dims, dims,1))
        return(X_img)
                   




    @staticmethod                   
    def shuffle_to_seed(seed, lst):
        random.Random(seed).shuffle(lst)
        return(lst)


    @staticmethod
    def _mat_to_rgb(mat: np.ndarray) -> np.ndarray:
        """Convert image matrix to numpy rgb format

        Args:
            mat: {array-like} (M, N)

        Returns:
            An numpy.ndarray (M, N, 3) with original values repeated across
            RGB channels.
        """
        return np.repeat(mat[:, :, np.newaxis], 3, axis=2)
