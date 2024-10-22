
import pandas as pd
from sklearn.preprocessing import StandardScaler
from create_sequences import create_sequences
def curate_training_test_data(df_all, Date_col_name = 'Date',
                              sequence_length = 7, 
                              test_date= '2024-07-01', 
                              flatten = False,
                              predictors_lst = ['EUA', 'Oil', 'Coal', 'NG', 'USEU', 'S&P_clean', 'DAX']):
    test_date = pd.to_datetime(test_date)



    test_overlap_time = pd.to_timedelta(sequence_length+1, unit = 'day')

    df_train = df_all[df_all[Date_col_name] < test_date].reset_index(drop=True)
    df_test  = df_all[df_all[Date_col_name] > test_date - test_overlap_time].reset_index(drop=True)

    train_data = df_train[predictors_lst].values 
    test_data  = df_test[predictors_lst].values
    scaler = StandardScaler()
    scaler.fit(train_data)

    train_data_scaled = scaler.transform(train_data)
    test_data_scaled  = scaler.transform(test_data)
    X_train, y_train = create_sequences(train_data_scaled, sequence_length, flatten = flatten) # LSTM should be flatten = False
    X_test, y_test = create_sequences(test_data_scaled, sequence_length, flatten = flatten)
    return X_train, y_train, X_test, y_test, scaler

