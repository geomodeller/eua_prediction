# inverse transform of variables
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import matplotlib.pyplot as plt


def inverse_scaler_of_all_var(scaled_data: np.ndarray|list[np.ndarray], 
                       scaler: StandardScaler)->np.ndarray|list:
    
    """
    Applies the inverse transformation using the provided scaler to the scaled data.
    
    Parameters:
    scaled_data (np.ndarray | list[np.ndarray]): The scaled data which could be a single numpy array or a list of numpy arrays.
    scaler (StandardScaler): The scaler object that was used to scale the data.
    
    Returns:
    np.ndarray | list: The original data obtained after applying the inverse transformation. 
                       If the input is a single numpy array, a numpy array is returned.
                       If the input is a list of numpy arrays, a list of numpy arrays is returned.
    
    Raises:
    TypeError: If the scaled_data is not a numpy array or a list of numpy arrays.
    """
    if isinstance(scaled_data, (np.ndarray|list)) is not True: 
        raise TypeError('scaled_data should be either np.ndarray or list of np.ndarray')
    if type(scaled_data) == np.ndarray:
        scaled_dim = scaled_data.shape
        scaled_data = scaled_data.reshape(-1,  scaled_data.shape[-1])
        original_data = scaler.inverse_transform(scaled_data)
        return original_data.reshape(scaled_dim)
    else:
        original_data_ensemble = []
        for scaled_data_element in scaled_data:
            scaled_dim = scaled_data_element.shape
            scaled_data_element = scaled_data_element.reshape(-1,  scaled_data_element.shape[-1])
            original_data_element = scaler.inverse_transform(scaled_data_element)
            original_data_ensemble.append(original_data_element.reshape(scaled_dim))
        return original_data_ensemble
    

def forward_scaler_of_all_var(scaled_data: np.ndarray|list[np.ndarray], 
                       scaler: StandardScaler)->np.ndarray|list:
    
    
    """
    Applies the forward transformation using the provided scaler to the data.

    Parameters:
    scaled_data (np.ndarray | list[np.ndarray]): The data to be scaled, which could be a single numpy array or a list of numpy arrays.
    scaler (StandardScaler): The scaler object that is used to transform the data.

    Returns:
    np.ndarray | list: The scaled data obtained after applying the transformation. 
                       If the input is a single numpy array, a numpy array is returned.
                       If the input is a list of numpy arrays, a list of numpy arrays is returned.

    Raises:
    TypeError: If the scaled_data is not a numpy array or a list of numpy arrays.
    """
    if isinstance(scaled_data, (np.ndarray|list)) is not True: 
        raise TypeError('scaled_data should be either np.ndarray or list of np.ndarray')
    if type(scaled_data) == np.ndarray:
        scaled_dim = scaled_data.shape
        scaled_data = scaled_data.reshape(-1,  scaled_data.shape[-1])
        original_data = scaler.transform(scaled_data)
        return original_data.reshape(scaled_dim)
    else:
        original_data_ensemble = []
        for scaled_data_element in scaled_data:
            scaled_dim = scaled_data_element.shape
            scaled_data_element = scaled_data_element.reshape(-1,  scaled_data_element.shape[-1])
            original_data_element = scaler.transform(scaled_data_element)
            original_data_ensemble.append(original_data_element)
        return original_data_ensemble
    
def resursive_furture_prediction(model, 
                                 last_price_all: torch.Tensor, 
                                 future_period: int = 12, 
                                 input_data_time_length: int = 28,
                                 scaler: None | StandardScaler = None,
                                 train_test_split_date = None, df_all = None,):
    """
    This function predicts the future EUA prices by recursively applying the provided model on the last input data.

    Parameters:
    model (nn.Module): The model to be used for prediction.
    last_price_all (torch.Tensor): The last input data.
    future_period (int): The number of future periods to predict. Defaults to 12.
    input_data_time_length (int): The length of the input data. Defaults to 28.
    add_scaler_inversion (bool): Whether to apply inverse scaler to the predictions. Defaults to True.
    train_test_split_date (pd.Timestamp or None): The date to split the data into training and testing sets. Defaults to None.
    df_all (pd.DataFrame or None): The dataframe containing all the data. Defaults to None.

    Returns:
    future_price (list): A list of numpy arrays, each of which contains the predicted EUA prices for a future period.
    future_time (list or None): A list of lists of dates, each of which contains the dates for a future period. If train_test_split_date or df_all is None, it returns None.
    """
    future_price = []
    for iter in range(future_period):
        if iter == 0:
            next_price = model(last_price_all.reshape(1,input_data_time_length,-1))
        else:
            next_price = model(next_price)
        future_price.append(next_price.detach().cpu().numpy())
    if scaler is not None:
        future_price = inverse_scaler_of_all_var(future_price, scaler)

    if any((train_test_split_date is None, df_all is None)):
        future_time = None
    else:
        future_time = [[train_test_split_date + pd.to_timedelta(j, 'day') + pd.to_timedelta(input_data_time_length*i, 'day') for j in range(input_data_time_length)] for i in range(future_period)]
    return future_price, future_time

def resursive_furture_prediction_with_dropout(model, 
                                            last_price_all: torch.Tensor, 
                                            num_of_ensemble = 10,
                                            future_period: int = 12, 
                                            input_data_time_length: int = 28,
                                            scaler: None | StandardScaler = None,
                                            train_test_split_date = None, 
                                            df_all = None):
    """
    Recursive future prediction with dropout.

    Parameters:
    model (nn.Module): The model to make predictions.
    last_price_all (torch.Tensor): The last price data as input to the model.
    num_of_ensemble (int): Number of ensembles to generate. Defaults to 10.
    future_period (int): The number of days to predict in the future. Defaults to 12.
    input_data_time_length (int): The length of the input data. Defaults to 28.
    add_scaler_inversion (bool): Whether to apply inverse scaler to the predictions. Defaults to True.
    train_test_split_date (pd.Timestamp or None): The date to split the data into training and testing sets. Defaults to None.
    df_all (pd.DataFrame or None): The dataframe containing all the data. Defaults to None.

    Returns:
    future_price (list): A list of numpy arrays, each of which contains the predicted EUA prices for a future period.
    future_time (list or None): A list of lists of dates, each of which contains the dates for a future period. If train_test_split_date or df_all is None, it returns None.
    """
    future_price_ensemble = []
    for real in range(num_of_ensemble):
        future_price = []
        for iter in range(future_period):
            if iter == 0:
                next_price = model(last_price_all.reshape(1,input_data_time_length,-1))
            else:
                next_price = model(next_price)
            future_price.append(next_price.detach().cpu().numpy())
        if scaler is not None:
            future_price = inverse_scaler_of_all_var(future_price, scaler)
        future_price_ensemble.append(future_price)    
    if any((train_test_split_date is None, df_all is None)):
        future_time = None
    else:
        future_time = [[train_test_split_date + pd.to_timedelta(j, 'day') + pd.to_timedelta(input_data_time_length*i, 'day') for j in range(input_data_time_length)] for i in range(future_period)]
    return future_price_ensemble, future_time

def visual_train_n_valid_data_performance(y_train_pred: np.ndarray | list[np.ndarray],
                                          y_test_pred: np.ndarray,
                                          train_start_date: pd.Timestamp,
                                          train_test_split_date: pd.Timestamp,
                                          df_all: pd.DataFrame,
                                          figsize: tuple = (15,5),
                                          index_of_data: int = 0, # 0 is EUA price
                                          name_of_data: str = 'EUA',
                                          alpha: float = 0.3,
                                          train_color: str = 'gray',
                                          test_color: str = 'gray',
                                          decoration: dict = {},
                                          input_data_time_length: int = 28,
                                          ):
    """
    Visualize the predictions on training and validation datasets.

    Parameters:
    y_train_pred (np.ndarray or list of np.ndarray): The predicted values on the training dataset.
    y_test_pred (np.ndarray): The predicted values on the validation dataset.
    train_start_date (pd.Timestamp): The start date of the training dataset.
    train_test_split_date (pd.Timestamp): The date to split the data into training and validation sets.
    df_all (pd.DataFrame): The dataframe containing all the data.
    figsize (tuple, optional): The size of the figure. Defaults to (15,5).
    index_of_data (int, optional): The index of the data to plot. Defaults to 0.
    alpha (float, optional): The alpha value of the plot. Defaults to 0.3.
    train_color (str, optional): The color of the training data. Defaults to 'gray'.
    test_color (str, optional): The color of the test data. Defaults to 'gray'.
    decoration (dict, optional): A dictionary containing additional plot decorations. Defaults to {}.
    input_data_time_length (int, optional): The length of the input data. Defaults to 28.
    """
    plt.figure(figsize = figsize)
    plt.axvline(x = train_test_split_date, color = 'purple', label = 'train/valid split')
    for i in range(y_train_pred.shape[0]):
        x_val_start = train_start_date + pd.to_timedelta(i, unit = 'day')
        x_val = [x_val_start + pd.to_timedelta(j, unit = 'day') for j in range(input_data_time_length)]
        if i == 0:
            plt.plot(x_val, y_train_pred[i,:,index_of_data], label = 'prediction',color =train_color, alpha = alpha)
        else:
            plt.plot(x_val, y_train_pred[i,:,index_of_data], color =train_color, alpha = alpha)
    
    if type(y_test_pred) == np.ndarray:
        num_test = y_test_pred.shape[0]
    elif type(y_test_pred) == list and all([type(e)==np.ndarray] for e in y_test_pred):
        num_test = y_test_pred[0].shape[0]
    else:
        raise TypeError
    
    for i in range(num_test):
        x_val_start = train_test_split_date + pd.to_timedelta(i, unit = 'day')
        x_val = [x_val_start + pd.to_timedelta(j, unit = 'day') for j in range(input_data_time_length)]
        if type(y_test_pred) == np.ndarray:
            plt.plot(x_val, y_test_pred[i,:,index_of_data],color =test_color, alpha = alpha)
        else:
            _ = [plt.plot(x_val, y_test_pred[j][i,:,index_of_data],color =test_color, alpha =alpha) for j in range(len(y_test_pred))]

    plt.plot(df_all[df_all['Date']<train_test_split_date]['Date'], 
            df_all[df_all['Date']<train_test_split_date][name_of_data],
            '-r', label = 'history')
    plt.plot(df_all[df_all['Date']>train_test_split_date]['Date'], 
            df_all[df_all['Date']>train_test_split_date][name_of_data],
            '-b', label = 'future')
    if 'xlabel' in decoration:
        plt.xlabel(decoration['xlabel'])
    if 'ylabel' in decoration:
        plt.ylabel(decoration['ylabel'])
    if 'title' in decoration:
        plt.title(decoration['title'])
    if 'grid' in decoration:
        plt.grid(decoration['grid'])
        
    plt.legend()