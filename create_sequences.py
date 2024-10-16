import numpy as np
def create_sequences(data: np.array, seq_length: int, flatten = False):
    """create_sequences of temporal data

    Args:
        data (np.array): original sequence data (date ascending order)
        seq_length (int): length of sequence to make

    Returns:
        tuple: X and y of seuqntial data
    """
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  
        y.append(data[i + seq_length])  
    if flatten:
        X = np.array(X)
        y = np.array(y)
        return X.reshape(X.shape[0],-1), y.reshape(y.shape[0],-1)
    else:
        return np.array(X), np.array(y)