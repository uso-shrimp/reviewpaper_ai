import tkinter as tk
from tkinter import filedialog
import threading

import numpy as np
import pandas as pd

from keras import callbacks
from keras import Input
from keras import layers
from keras import regularizers
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import time
import os
import re

import concurrent.futures
import logging
import matplotlib.pyplot as plt

from threading import Thread

MAX_LENGTH = 450
NUM_EMBEDDING_DIM = 100
NUM_LSTM_UNITS = 96

GLOVE_DIR = 'C:/Users/TR814-Public/PycharmProjects/nlpProject'

# model = Sequential()
# g_history = tk.keras.callbacks.History()
HISTORY_FLAG = False

FILE_PATH = None
COL_NAME = None
AB_NAME = None
LABEL_NAME = None
SAVE_DIR = None
WB_PATH = None


class Threader(threading.Thread):
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.daemon = True
        self.start()

    def run(self):
        run_AI()


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class CustomCallback(callbacks.Callback):
    def __init__(self):
        self.task_type = ''
        self.epoch = 1
        self.batch = 0
        self.epoch_time_start = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.epoch_time_start = time.time()
        epoch_counter = epoch + 1
        msg_IO('== Epoch {}/10 ==\n'.format(epoch_counter))

    def on_epoch_end(self, epoch, logs=None):
        msg_IO(' - time: {:7.0f}s\n'.format(time.time() - self.epoch_time_start))
        msg_IO(' - loss: {:7.4f}'.format(logs["loss"]))
        msg_IO(' - accuracy: {:7.4f}\n'.format(logs["accuracy"]))
        msg_IO(' - val_loss: {:7.4f}'.format(logs["val_loss"]))
        msg_IO(' - val_accuracy: {:7.4f}\n'.format(logs["val_accuracy"]))
        msg_IO('\n')


def run_AI():
    print("run AI")
    global FILE_PATH, COL_NAME, AB_NAME, LABEL_NAME, SAVE_DIR, WB_PATH
    filePath = path_entry.get()
    ti_name = col_entry.get()
    ab_name = ab_entry.get()
    label_name = label_entry.get()
    save_dir = folder_entry.get()
    wb_path = wb_entry.get()

    if FILE_PATH is not filePath:
        FILE_PATH = filePath

    if COL_NAME is not ti_name:
        COL_NAME = ti_name

    if AB_NAME is not ab_name:
        AB_NAME = ab_name

    if LABEL_NAME is not label_name:
        LABEL_NAME = label_name

    if SAVE_DIR is not save_dir:
        SAVE_DIR = save_dir

    if WB_PATH is not wb_path:
        WB_PATH = wb_path

    with open('output.txt', 'w+') as f:
        f.write(filePath+"\n")
        f.write(wb_path+"\n")
        f.write(save_dir + "\n")
        f.write(ti_name+"\n")
        f.write(ab_name+"\n")
        f.write(label_name+"\n")

    build_ai(FILE_PATH, COL_NAME, AB_NAME, LABEL_NAME, SAVE_DIR, WB_PATH)


def print_train_test_data(x_train, y_train, x_test, y_test):
    print("\nTraining Set")
    print("-" * 10)
    print(f"x_train: {x_train.shape}")
    print(f"y_train : {y_train.shape}")

    print("-" * 10)
    print(f"x_val:   {x_test.shape}")
    print(f"y_val :   {y_test.shape}")
    print("-" * 10)
    print("Test Set\n")


def print_history(history, save_dir):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(save_dir + '/acc.png')

    plt.figure()

    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # plt.show()
    print("save dir" + save_dir)
    plt.savefig(save_dir + '/loss.png')


def build_ai(filePath, col_name, ab_name, label_name, save_dir, wb_path):
    msg_IO('Reading data ...\n')
    try:
        f = open(wb_path)
        words = f.read()
        word_bank = words.split(',')
    except FileNotFoundError:
        msg_IO('Input file not found.\n')

    train = pd.read_excel(filePath, usecols=[col_name, ab_name, label_name], encoding="utf-8")
    print('AB')
    train[col_name] = train[col_name].astype(str)
    train[ab_name] = train[ab_name].astype(str)

    print(train['AB'].head(5).values)
    print(train['AB'].head(5))

    print(word_bank)

    input = np.append(word_bank, word_bank)

    for ti_sentence, ab_sentence in zip(train[col_name].values, train[ab_name].values):
        ti_list = []
        ab_list = []
        for word in word_bank:
            ti_list.append(len(re.findall(word, str(ti_sentence), re.IGNORECASE)))
            ab_list.append(len(re.findall(word, str(ab_sentence), re.IGNORECASE)))

        # print(ti_list)
        # print(ab_list)
        sentence_list = np.append(ti_list, ab_list)
        print(sentence_list)
        ti_list.clear()
        ab_list.clear()
        input = np.vstack([input, sentence_list])

    df_word = pd.DataFrame(input[1:], columns=word_bank+word_bank)
    df_word.to_excel("keywords statistics.xlsx")

    input = np.delete(input, 0, axis=0)

    msg_IO('Read ' + str(len(train[col_name])) + ' data \n')

    # Train-test split
    merger_train, merger_val, \
    y_train, y_val \
        = train_test_split(input, train[label_name].values, test_size=0.25, random_state=1000)

    msg_IO("merge Train on " + str(len(merger_train)) + " samples,")
    msg_IO("merge validate on " + str(len(merger_val)) + " samples\n")

    # print(input_ti_train)
    msg_IO("Constructing AI model...\n")

    model = Sequential()

    if len(word_bank) > 30:
        model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    start = time.clock()
    msg_IO("Training AI model...\n")
    history = model.fit(x=merger_train,
                        y=y_train,
                        epochs=10,
                        verbose=2,
                        validation_data=(merger_val, y_val),
                        batch_size=50,
                        callbacks=[CustomCallback()])
    model.summary()

    model.save(save_dir + '/model.h5')

    print_history(history, save_dir)


def browseFolder():
    tk.Tk().withdraw()

    folder_entry.delete(0, 'end')
    folder_entry.insert(0, filedialog.askdirectory())

def browseDataFile():
    tk.Tk().withdraw()

    path_entry.delete(0, 'end')
    path_entry.insert(0, filedialog.askopenfilename())

def browseWBFile():
    tk.Tk().withdraw()

    wb_entry.delete(0, 'end')
    wb_entry.insert(0, filedialog.askopenfilename())

def msg_IO(msg):
    msg_text.insert('end', msg)
    msg_text.update_idletasks()


if __name__ == '__main__':

    FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, filename='myLog.log', filemode='w', format=FORMAT)

    logging.debug('debug message')
    logging.info('info message')
    logging.warning('warning message')
    logging.error('error message')
    logging.critical('critical message')

    DEFAULT_FILE = True
    if os.path.isfile('output.txt'):
        with open('output.txt', 'r') as f:
            default_str = f.readlines()

        if len(default_str) is not 6:
            DEFAULT_FILE = False
    else:
        DEFAULT_FILE = False

    window = tk.Tk()
    window.title('AI Builder for Review Paper')
    window.geometry('800x800')
    window.configure(background='white')

    header_label = tk.Label(window,
                            text='\nBuild your AI for Review Paper',
                            bg='white',
                            font=('Times New Roman', 26))
    header_label.grid(row=0, column=0)

    top_frame = tk.Frame(window)
    top_frame.grid(row=1, column=0)
    content_label = tk.Label(top_frame,
                             text='\nThis program generate the review paper AI of input data',
                             bg='white',
                             font=('Times New Roman', 14))
    content_label.pack(side=tk.TOP)

    center_frame = tk.Frame(window, bg='white')
    center_frame.grid(row=2, column=0)
    Step_headline_label = tk.Label(center_frame,
                                   text='\n\nSteps.',
                                   bg='white',
                                   font=('Times New Roman', 14))
    Step_headline_label.pack(side=tk.TOP)
    Step_num_label = tk.Label(center_frame,
                              text='',
                              bg='white',
                              font=('Times New Roman', 12),
                              anchor='ne',
                              height='5')
    Step_num_label.pack(side=tk.LEFT)
    Step_label = tk.Label(center_frame,
                          text='Choose your xlsx file by pressing "Browse file" button.\n'
                               'Enter column name of data stream and label in file. \nPress Execute button.\n'
                               '\nPLEASE CLOSE any output file the program generates before executing',
                          bg='white',
                          font=('Times New Roman', 12),
                          justify='left',
                          anchor='nw',
                          wraplength='350')
    Step_label.pack(side=tk.RIGHT)

    line_frame = tk.Frame(window, bg='white', bd='0px')
    line_frame.grid(row=3, column=0)
    cv = tk.Canvas(line_frame, bg='white', bd=0, height='70', width='500', highlightthickness=0)
    cv.pack(side=tk.TOP)
    line = cv.create_line(0, 25, 500, 25)

    bottom_frame = tk.Frame(window, bg='white', bd='0px')
    bottom_frame.grid(row=4, column=0)
    Step_label = tk.Label(bottom_frame,
                          text='Input file path',
                          bg='white',
                          font=('Times New Roman', 10),
                          width='15',
                          anchor='nw')
    Step_label.grid(row=0, column=0, sticky='E')
    path_entry = tk.Entry(bottom_frame, width='40', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        path_entry.insert(0, default_str[0])
    path_entry.grid(row=0, column=1)

    parse_files_btn = tk.Button(bottom_frame, text='Browse file', anchor='nw', command=browseDataFile)
    parse_files_btn.grid(row=0, column=2)

    wb_label = tk.Label(bottom_frame,
                          text='Word bank path',
                          bg='white',
                          font=('Times New Roman', 10),
                          width='15',
                          anchor='nw')
    wb_label.grid(row=1, column=0, sticky='E')
    wb_entry = tk.Entry(bottom_frame, width='40', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        wb_entry.insert(0, default_str[1])
    wb_entry.grid(row=1, column=1)

    wb_files_btn = tk.Button(bottom_frame, text='Browse file', anchor='nw', command=browseWBFile)
    wb_files_btn.grid(row=1, column=2)

    Output_frame = tk.Frame(window, bg='white', bd='0px')
    Output_frame.grid(row=5, column=0)
    folder_label = tk.Label(Output_frame,
                            text='Output folder path',
                            bg='white',
                            font=('Times New Roman', 10),
                            width='63',
                            anchor='nw')
    folder_label.pack(side=tk.TOP)
    folder_entry = tk.Entry(Output_frame, width='50', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        folder_entry.insert(0, default_str[2])
    folder_entry.pack(side=tk.LEFT)

    parse_folder_btn = tk.Button(Output_frame, text='Browse folder', anchor='nw', command=browseFolder)
    parse_folder_btn.pack(side=tk.RIGHT)

    col_frame = tk.Frame(window, bg='white', bd='0px')
    col_frame.grid(row=6, column=0)
    Col_label = tk.Label(col_frame,
                         text='Title column name of training data',
                         bg='white',
                         font=('Times New Roman', 10),
                         width='30',
                         anchor='e')
    Col_label.grid(row=0, column=0, padx=5)
    col_entry = tk.Entry(col_frame, width='25', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        col_entry.insert(0, default_str[3])
    col_entry.grid(row=0, column=1)

    ab_label = tk.Label(col_frame,
                        text='Abstract column name of training data',
                        bg='white',
                        font=('Times New Roman', 10),
                        width='30',
                        anchor='e')
    ab_label.grid(row=1, column=0, padx=5)
    ab_entry = tk.Entry(col_frame, width='25', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        ab_entry.insert(0, default_str[4])
    ab_entry.grid(row=1, column=1)
    entry_frame = tk.Frame(window, bg='white', bd='0px', heigh='20')
    entry_frame.grid(row=7, column=0)
    Label_label = tk.Label(entry_frame,
                           text='Column name of label data',
                           bg='white',
                           font=('Times New Roman', 10),
                           width='25',
                           anchor='e')
    Label_label.grid(row=0, column=0, padx=5)
    label_entry = tk.Entry(entry_frame, width='25', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        label_entry.insert(0, default_str[5])
    label_entry.grid(row=0, column=1, padx=5)

    execute_btn = tk.Button(entry_frame, text='Execute', anchor='nw', command=lambda: Threader(name='exe'))
    execute_btn.grid(row=1, column=0)

    msg_frame = tk.Frame(window, bg='white', bd='0px')
    msg_frame.grid(row=8, column=0, padx=10)

    msg_label = tk.Label(msg_frame, bg='white',
                         text='Process',
                         font=('Times New Roman', 10),
                         width='25',
                         anchor='w')
    msg_label.grid(row=0, column=0)
    msg_text = tk.Text(msg_frame,
                       bg='white',
                       font=('Times New Roman', 12),
                       height='7',
                       width='70',
                       padx='5')
    msg_text.grid(row=1, column=0)

    window.mainloop()
