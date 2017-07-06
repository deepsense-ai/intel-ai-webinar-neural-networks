import time
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, GlobalMaxPool2D

resolution = 28
classes = 10

def build_models_dict():
    models = {'log_reg':log_reg(),
          'mlp':mlp(),
          'conv1':conv1(),
          #'conv1_maxpool':conv1_maxpool(),
          #'conv2_maxpool':conv2_maxpool(),
          #'conv2_maxpool_large':conv2_maxpool_large(),
          #'conv2x2_maxpool':conv2x2_maxpool(),
          #'conv2x2_maxpool_dropout':conv2x2_maxpool_dropout(),
          #'conv2x2_maxpool_dropout_batchnorm':conv2x2_maxpool_dropout_batchnorm(),
          #'conv2x2_maxpool_dropout_batchnorm_dense':conv2x2_maxpool_dropout_batchnorm_dense(), 
      #'conv2x2_maxpool_dropout_batchnorm_fully_convo':conv2x2_maxpool_dropout_batchnorm_fully_convo(),
}
    return models

def log_reg():
    model = Sequential()

    model.add(Flatten(input_shape=(resolution, resolution, 1)))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def mlp():
    model = Sequential()

    model.add(Flatten(input_shape=(resolution, resolution, 1)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def conv1():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=(resolution, resolution, 1)))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def conv1_maxpool():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=(resolution, resolution, 1)))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def conv2_maxpool():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=(resolution, resolution, 1)))
    model.add(MaxPool2D())

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def conv2_maxpool_large():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu',
                     input_shape=(resolution, resolution, 1)))
    model.add(MaxPool2D())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def conv2x2_maxpool():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=(resolution, resolution, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def conv2x2_maxpool_dropout():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=(resolution, resolution, 1)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def conv2x2_maxpool_dropout_batchnorm():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=(resolution, resolution, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
    
def conv2x2_maxpool_dropout_batchnorm_dense():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=(resolution, resolution, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def conv2x2_maxpool_dropout_batchnorm_fully_convo():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=(resolution, resolution, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(GlobalMaxPool2D())

    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train_models_and_format_grid(X_train, Y_train, X_test, Y_test, plot_losses, epochs=10):
    all_models_trained = []
    all_training_times = []
    all_model_names = []
    all_model_specs = build_models_dict()
    for name, model in all_model_specs.items():
        print(name)
        start = time.time()
        model.fit(X_train, Y_train,
                   epochs=epochs,
                   batch_size=32,
                   validation_data=(X_test, Y_test), callbacks=[plot_losses])
        end = time.time()
        all_training_times.append(int(end-start))
        all_models_trained.append(model)
        all_model_names.append(name)
    df = score_model_grid(all_models_trained, all_model_names, all_training_times, 
                          X_train, Y_train, X_test, Y_test)
    
    df['acc_overfit'] = df['acc_train'] - df['acc_test']
    df['loss_overfit'] = df['loss_test'] - df['loss_train']
    return df

def plot_grid(save='../resources/cached_model_grid_scores.csv'):
    grid = pd.read_csv(save)   
    grid = pd.melt(grid, id_vars=['model_names','time_to_train','params'])

    def split_variable(x):
        s, d = x.split('_')
        return pd.Series({'data_fold':s, 'score': d})
    grid[['score', 'data_fold']] = grid['variable'].apply(lambda x: split_variable(x))

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12,12));
    grid_acc = grid[(grid['score'] == 'acc') 
                    & (grid['data_fold'] != 'overfit')]
    sns.swarmplot(data=grid_acc, 
                  y='variable', x='value', hue='model_names', ax=ax1);
    ax1.set(xlabel='scores', ylabel='');
    ax1.legend_.remove();
    grid_overfit= grid[(grid['score'] == 'acc')
                       & (grid['data_fold'] == 'overfit')]
    sns.swarmplot(data=grid_overfit, 
                        y='variable', x='value', hue='model_names', ax=ax2);
    ax2.set(xlabel='scores', ylabel='');
    ax2.legend_.remove();
    grid_loss = grid[(grid['score'] == 'loss') 
                     & (grid['data_fold'] != 'overfit')]
    sns.swarmplot(data=grid_loss, 
                        y='variable', x='value', hue='model_names', ax=ax3);
    ax3.set(xlabel='losses', ylabel='');
    ax3.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
    grid_overfit= grid[(grid['score'] == 'loss')
                       & (grid['data_fold'] == 'overfit')]
    sns.swarmplot(data=grid_overfit, 
                        y='variable', x='value', hue='model_names', ax=ax4);
    ax4.set(xlabel='losses', ylabel='');
    ax4.legend_.remove();
    plt.show();
    
def plot_complexity(models=None, save='../resources/cached_model_grid_scores.csv'):
    if save:
        grid = pd.read_csv(save)
    else:
        grid = score_model_grid(models)
        grid.to_csv(save, index=None)
        
    grid = pd.melt(grid, id_vars=['model_names','time_to_train','params'])
    def split_variable(x):
        s, d = x.split('_')
        return pd.Series({'data_fold':s, 'score': d})
    grid[['score', 'data_fold']] = grid['variable'].apply(lambda x: split_variable(x))

    plt.figure(figsize=(16,10));
    sns.lmplot(data=grid, x='time_to_train', y='params', 
               hue='model_names',fit_reg=False);
    plt.show();
    
def score_model_grid(models, names, train_times, X_train, Y_train, X_test, Y_test):
    ids, train_loss, test_loss, train_acc, test_acc, total_params = [], [], [], [], [], []
    for i, model in enumerate(models):
        tr_l, tr_ac = model.evaluate(X_train, Y_train) 
        ts_l, ts_ac = model.evaluate(X_test, Y_test)
        tot_prm = _get_params_nr(model)
        ids.append(i)
        train_loss.append(tr_l)
        test_loss.append(ts_l)
        train_acc.append(tr_ac)
        test_acc.append(ts_ac)
        total_params.append(tot_prm)
    df = pd.DataFrame({'acc_train':train_acc,
                       'acc_test':test_acc,
                       'loss_train':train_loss,
                       'loss_test':test_loss,
                       'params':total_params,
                       'time_to_train':train_times,
                       'model_names':names})
    return df

def _get_params_nr(model):
    params = 0
    for layer in model.layers:
        params += layer.count_params()
    return params