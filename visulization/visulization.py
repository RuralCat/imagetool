
from __future__ import absolute_import

import cv2
import pandas
from matplotlib import pyplot as plt
from imagetool.history import History as hist
from imagetool.dataset import fullfile
from imagetool.config import Config

name_list = [hist.MODEL_UNET_NUCLEUS_CS24_AUG_INPUTBATCH_122_1516]

import csv
import os
root_path = os.path.abspath('../')
filename = fullfile(root_path, 'results', name_list[0].value, 'training_log.csv')

def plot_metrics(model_list):
    for model in model_list:
        config = Config.load_config(root_path, model.value)
        name = config.train_log_path
        with open(filename, 'r') as f:
            d = csv.reader(f)
            for item in d:
                d

df = pandas.read_csv(filename)
titles = df.keys()
epochs = df['epoch']
loss = df['loss']
val_loss = df['val_loss']
print(df)
print(titles[0])
print(loss)

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('metrics')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])
plt.show()

