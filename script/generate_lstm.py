
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, LayerNormalization, LeakyReLU, Input


def generate_lstm(X_train, predictors_lst):
    # Learning rate scheduler function

        # else:
        #     return lr * np.exp(-0.1)

    model = Sequential()

    # First LSTM layer with LayerNormalization and recurrent dropout
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]),
                activation='tanh', recurrent_activation='sigmoid',
                return_sequences=True, recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(LayerNormalization())
    # model.add(Dropout(0.1))

    # Second LSTM layer with residual connection, LayerNormalization, and recurrent dropout
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                return_sequences=True, recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(LayerNormalization())
    # model.add(Dropout(0.1))

    # Third LSTM layer (final) without returning sequences, adding residual connection
    model.add(LSTM(256, activation='tanh', recurrent_activation='sigmoid',
                return_sequences=False, recurrent_dropout=0.2,
                kernel_regularizer=l2(0.001)))
    model.add(LayerNormalization())

    # Dense layer with LeakyReLU activation for flexibility in output
    model.add(Dense(len(predictors_lst)))
    # model.add(LeakyReLU(alpha=0.1))  # LeakyReLU is more flexible for output regression

    # Compile the model using AdamW optimizer and a learning rate scheduler
    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-5)  # AdamW improves generalization
    model.compile(optimizer=optimizer, loss='mse')

    return model
