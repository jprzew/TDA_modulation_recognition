# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Tools

# +
import os
import platform
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

import keras
from keras import models
from keras.layers import *
from keras.optimizers import SGD
import argparse
# -

print(platform.python_version())

path="../hdf5_data/GOLD_XYZ_OSC.0001_1024.hdf5"
path2="C:/Users/aniaw/"

# +
classes = ['32PSK',
           '16APSK',
           '32QAM',
           'FM',
           'GMSK',
           '32APSK',
           'OQPSK',
           '8ASK',
           'BPSK',
           '8PSK',
           'AM-SSB-SC',
           '4ASK',
           '16PSK',
           '64APSK',
           '128QAM',
           '128APSK',
           'AM-DSB-SC',
           'AM-SSB-WC',
           '64QAM',
           'QPSK',
           '256QAM',
           'AM-DSB-WC',
           'OOK',
           '16QAM']


def load_data_from_hdf5(data_path):
    f = h5py.File(data_path, 'r')
    return f['X'][:30000], f['Y'][:30000], f['Z'][:30000]


# +
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=classes):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_lines(train_his, val_his, saved_name='images.png'):
    x = np.arange(1, len(train_his)+1)
    plt.plot(x, train_his, color='tomato', linewidth=2, label='train')
    plt.plot(x, val_his, color='limegreen', linewidth=2, label='val')
    plt.legend()
    # plt.show()
    plt.savefig(saved_name, format='png', bbox_inches='tight')
    plt.close()

#if __name__ == '__main__':
#    load_data_from_hdf5('/dataset/RadioML/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5')


# -

# # Model

# +

def cnn_model(num_classes):
    dr = 0.5
    model = models.Sequential()
    model.add(Conv2D(32, (2, 2), padding='valid', activation="relu", input_shape=[1024, 2, 1]))
#     model.add(Dropout(dr))
    model.add(Reshape([1023, 32]))
    model.add(Conv1D(64, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Conv1D(128, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Conv1D(256, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Conv1D(128, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Conv1D(64, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Conv1D(32, 3, strides=2, padding="valid", activation="relu"))
    # model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dr))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.summary()
    return model



# -

# # Train

def run(args):
    data, mod_label, snr_label = load_data_from_hdf5(args.data_path)
    data = np.expand_dims(data, axis=-1)
    np.random.seed(2016)
    n_examples = data.shape[0]
    n_train = n_examples * args.train_split 
    train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
    X_train = data[train_idx]
    X_test = data[test_idx]

    Y_train = mod_label[train_idx]
    Y_test = mod_label[test_idx]

    print(X_train.shape, X_test.shape)

    if args.mode == 'train':
        callbacks = []
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=args.log_path,
            histogram_freq=0, write_graph=True)
        callbacks.append(tensorboard)

        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_path, 'weights_{epoch:02d}.hdf5'),
            period=20,
            save_weights_only=False,
            save_best_only=False)
        callbacks.append(checkpoint)

        num_classes = len(classes)
        model = cnn_model(num_classes)
        # model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
        sgd = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

        history = model.fit(X_train, Y_train, batch_size=args.batch_size, epochs=args.nb_epoch,
                            verbose=2, validation_data=(X_test, Y_test), callbacks=callbacks)

        plot_lines(history.history['acc'], history.history['val_acc'],
                   saved_name=os.path.join(args.log_path, 'acc.png'))
        plot_lines(history.history['loss'], history.history['val_loss'],
                   saved_name=os.path.join(args.log_path, 'loss.png'))

    else:
        model = models.load_model(os.path.join(args.checkpoint_path, 'weights_{:02d}.hdf5'.format(args.nb_epoch)))
        score = model.evaluate(X_test, Y_test, verbose=1, batch_size=args.batch_size)
        print(score)

        acc = {}
        snrs = np.unique(snr_label)
        for snr in snrs:

            # extract classes @ SNR
            test_SNRs = np.where(snr_label[test_idx] == snr)[0]
            test_X_i = X_test[test_SNRs]
            test_Y_i = Y_test[test_SNRs]

            # estimate classes
            test_Y_i_hat = model.predict(test_X_i)
            conf = np.zeros([len(classes), len(classes)])
            confnorm = np.zeros([len(classes), len(classes)])
            for i in range(0, test_X_i.shape[0]):
                j = list(test_Y_i[i, :]).index(1)
                k = int(np.argmax(test_Y_i_hat[i, :]))
                conf[j, k] = conf[j, k] + 1
            for i in range(0, len(classes)):
                confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
            plt.figure()
            plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d dB)" % (snr))
            plt.savefig('{}_dB_confusion_matrix.png'.format(snr), format='png', bbox_inches='tight')
            plt.close()

            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            print ("Overall Accuracy: ", cor / (cor + ncor))
            acc[snr] = 1.0 * cor / (cor + ncor)
        # %%
        # Plot accuracy curve
        plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("CNN2 Classification Accuracy on RadioML 2018 dataset")
        plt.savefig('CNN2 Classification Accuracy on RadioML 2018 dataset.png', format='png', bbox_inches='tight')
        plt.close()


# +
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/dataset/RadioML/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
    parser.add_argument('--nb_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--mode', type=str, default='eval', help="'train' or 'eval")
    parser.add_argument('--train_split', type=float, default=0.5)

    args = parser.parse_args(argv)
    return args

#if __name__ == '__main__':
 #   run(parse_arguments(sys.argv[1:]))

# +


def run(data_path, train_split, mode, log_path, checkpoint_path,
        learning_rate, batch_size, nb_epoch):
    data, mod_label, snr_label = load_data_from_hdf5(data_path)
    data = np.expand_dims(data, axis=-1)
    np.random.seed(2016)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
    n_examples = data.shape[0]
    n_train = n_examples * train_split  # 对半
    train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
    X_train = data[train_idx]
    X_test = data[test_idx]

    Y_train = mod_label[train_idx]
    Y_test = mod_label[test_idx]

    print(X_train.shape, X_test.shape)

    if mode == 'train':
        callbacks = []
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=log_path,
            histogram_freq=0, write_graph=True)
        callbacks.append(tensorboard)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, 'weights_{epoch:02d}.hdf5'), period=20,
            save_weights_only=False, save_best_only=False)
        callbacks.append(checkpoint)

        num_classes = len(classes)
        model = cnn_model(num_classes)
        # model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
        sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                            verbose=2, validation_data=(X_test, Y_test), callbacks=callbacks)

        plot_lines(history.history['acc'], history.history['val_acc'],
                   saved_name=os.path.join(log_path, 'acc.png'))
        plot_lines(history.history['loss'], history.history['val_loss'],
                   saved_name=os.path.join(log_path, 'loss.png'))

    else:
        model = models.load_model(os.path.join(checkpoint_path, 'weights_{:02d}.hdf5'.format(nb_epoch)))
        score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
        print(score)

        acc = {}
        snrs = np.unique(snr_label)
        for snr in snrs:

            # extract classes @ SNR
            test_SNRs = np.where(snr_label[test_idx] == snr)[0]
            test_X_i = X_test[test_SNRs]
            test_Y_i = Y_test[test_SNRs]

            # estimate classes
            test_Y_i_hat = model.predict(test_X_i)
            conf = np.zeros([len(classes), len(classes)])
            confnorm = np.zeros([len(classes), len(classes)])
            for i in range(0, test_X_i.shape[0]):
                j = list(test_Y_i[i, :]).index(1)
                k = int(np.argmax(test_Y_i_hat[i, :]))
                conf[j, k] = conf[j, k] + 1
            for i in range(0, len(classes)):
                confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
            plt.figure()
            plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d dB)" % (snr))
            plt.savefig('{}_dB_confusion_matrix.png'.format(snr), format='png', bbox_inches='tight')
            plt.close()

            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            print ("Overall Accuracy: ", cor / (cor + ncor))
            acc[snr] = 1.0 * cor / (cor + ncor)
        # %%
        # Plot accuracy curve
        plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("CNN2 Classification Accuracy on RadioML 2018 dataset")
        plt.savefig('CNN2 Classification Accuracy on RadioML 2018 dataset.png', format='png', bbox_inches='tight')
        plt.close()



# -

run(path, 0.5, 'train','logs', 'checkpoints',0.01, 1024, 4)

run(path, 0.5, 'eval','logs', 'checkpoints',0.01, 1024, 4)

# !ls ../hdf5_data/


