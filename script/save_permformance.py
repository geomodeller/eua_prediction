from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
import pickle 
import numpy as np
def save_perform(model_name, y_train_true, y_train_pred, y_test_true,y_test_pred):
    mae_train = np.array([mean_absolute_error(i[:,0],j[:,0]) for i, j in zip(y_train_true,y_train_pred)])
    mae_test = np.array([mean_absolute_error(i[:,0],j[:,0]) for i, j in zip(y_test_true,y_test_pred)])

    mape_train = np.array([mean_absolute_percentage_error(i[:,0],j[:,0]) for i, j in zip(y_train_true,y_train_pred)])
    mape_test = np.array([mean_absolute_percentage_error(i[:,0],j[:,0]) for i, j in zip(y_test_true,y_test_pred)])

    p50ae_train = np.array([median_absolute_error(i[:,0],j[:,0]) for i, j in zip(y_train_true,y_train_pred)])
    p50ae_test = np.array([median_absolute_error(i[:,0],j[:,0]) for i, j in zip(y_test_true,y_test_pred)])

    with open('performance_metric.plk', 'ab') as f:
        pickle.dump({f'{model_name}_mse_train': mae_train}, f)
        pickle.dump({f'{model_name}_mse_test': mae_test}, f)
        pickle.dump({f'{model_name}_mspe_train': mape_train}, f)
        pickle.dump({f'{model_name}_mspe_test': mape_test}, f)
        pickle.dump({f'{model_name}_p50aese_train': p50ae_train}, f)
        pickle.dump({f'{model_name}_p50aese_test': p50ae_test}, f)
        