import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import torch
import torchvision
from typing import Union, Any, Optional, Tuple
from typing_extensions import Protocol

class DeepmapperDataset(torch.utils.data.Dataset):
  def __init__(self, src_matrix, y_labels, inst_labels, map=False):
    
    tmp_x = src_matrix.astype(np.float32)  
    tmp_y = y_labels.astype(np.int64)
    
    if map:
        tmp_x = DeepMapper.map(tmp_x)

    self.x_data = torch.tensor(tmp_x, dtype=torch.float32)
    self.y_data = torch.tensor(tmp_y, dtype=torch.int64)
    self.id_data = inst_labels

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    preds = self.x_data[idx]
    spcs = self.y_data[idx] 
    id = self.id_data[idx]  # np.str
    sample = { 'data' : preds, 
               'class' : spcs,
               'id' : id }
    return sample
#ds=DeepmapperDataset(grand_matrix,labels,instance_labels,map=True)
#print(len(ds))
#train_set, val_set = torch.utils.data.random_split(ds, [8000, 2000])
#train_set[0]['data'].shape


#X = torch.tensor(grand_matrix.T)
#y = torch.tensor(labels).unsqueeze(1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
#train_tensor = TensorDataset(X_train, y_train) 
#test_tensor = TensorDataset(X_test, y_test)
#train_loader = DataLoader(dataset = train_tensor, batch_size = 256, shuffle = True) 
#test_loader = DataLoader(dataset = test_tensor, batch_size = 256, shuffle = True) 
#test_loader


class DeepMapper:
    """Transform features to an image matrix by direct mapping

    This class takes in data normalises if necessary,  and converts it to a
    CNN compatible 'image' matrix

    """

    def __init__(self) -> None:
        """Generate a DeepMapper instance
        """
        self.X=np.zeros([1])
        self.shape=self.X.shape

   
    def map(Xinput: np.ndarray, normalise:bool = False, plot: int = 0, buffer=0):
        size,xsize = Xinput.shape
        minsize= xsize+buffer
        dim = int(math.sqrt(minsize))
        if (dim*dim)<minsize:
            dim=dim+1 
        pad_size = dim*dim-minsize
        X_p = np.pad(Xinput, pad_width=((0,0), (0, pad_size)))
        X = np.resize(X_p,(size,dim, dim,1))
        shape=X.shape
        return(X)
                   

    def fit_transform(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return self.map(X)


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
