#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:46:02 2020

@author: shah38
"""

import nest_asyncio
nest_asyncio.apply()

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import ssl
import time
import collections
import os
from keras.utils.np_utils import to_categorical
import warnings
import sys
keras = tf.keras
ssl._create_default_https_context = ssl._create_unverified_context
np.random.seed(0)
tf.compat.v1.enable_v2_behavior()
warnings.filterwarnings("ignore")
# Timer helper class for benchmarking reading methods
class Timer(object):
    """Timer class
       Wrap a will with a timing function
    """
    
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.t = time.time()
        
    def __exit__(self, *args, **kwargs):
        print("{} took {} seconds".format(
        self.name, time.time() - self.t))
    

# make up our client scenario
NUM_CLIENTS=10
NUM_TRAIN_CLIENTS=(int)(0.8*NUM_CLIENTS)
NUM_TEST_CLIENTS = NUM_CLIENTS-NUM_TRAIN_CLIENTS
DATASET_SIZE = 30000

TRAIN_SIZE = int(0.8 * DATASET_SIZE)
VALIDATION_SIZE = int(0.2 * DATASET_SIZE)

client_ids = list(np.arange(NUM_CLIENTS))
client_ids_train = client_ids[0:NUM_TRAIN_CLIENTS]
client_ids_test = client_ids[NUM_TRAIN_CLIENTS:]
CLIENT_SIZE = (int)(DATASET_SIZE/NUM_CLIENTS)
BATCH_SIZE = 32
IMG_SHAPE=(50, 50, 3)
base_learning_rate = 0.001
# Fine-tune from this layer onwards
fine_tune_at = 15    

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return (int)(parts[-2] == '1')
def decode_img(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [50, 50])
def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def pretrained(labeled_ds, path, train=False):
    checkpoint_path = path + "/pretrained/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    train_ds = labeled_ds.take(TRAIN_SIZE)
    validation_ds = labeled_ds.skip(TRAIN_SIZE).take(VALIDATION_SIZE)
    
    train_batches = prepare_for_training(train_ds)
    validation_batches = prepare_for_training(validation_ds)
    
    
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(1)
    pretrained_model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])
    validation_steps=20
    if train:
        pretrained_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.BinaryAccuracy()])
        initial_epochs = 10
        
        loss0,accuracy0 = pretrained_model.evaluate(validation_batches, steps = validation_steps)
        with Timer("Pre-training"):
            history = pretrained_model.fit(train_batches,
                                epochs=initial_epochs,
                                validation_data=validation_batches, callbacks=[cp_callback])  
    else:
        print("Loading pretrained model")
        pretrained_model.load_weights(checkpoint_path)
        
    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model

    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False
    return pretrained_model


def make_federated_data(client_data, client_ids):
  return [
      prepare_for_training(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]
 

def get_data(path_data, is_iid):
    if is_iid :
        list_ds = tf.data.Dataset.list_files(path_data + '/data/balanced_IDC_30k/*/*', shuffle=True)  
        return list_ds
    else:
        list_ds_0 = tf.data.Dataset.list_files(path_data + '/data/balanced_IDC_30k/0/*', shuffle=True)  
        list_ds_1 = tf.data.Dataset.list_files(path_data + '/data/balanced_IDC_30k/1/*', shuffle=True)  
        list_ds = list_ds_1.concatenate(list_ds_0)
        return list_ds


def main():
    path_data = sys.argv[1]
    NUM_ROUNDS=sys.argv[2]
    is_iid = sys.argv[3]
    list_ds = get_data(path_data,is_iid=="iid")
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    toTrain = not sys.path.exists(path_data+"/pretrained/cp.ckpt")
    pretrained_model = pretrained(labeled_ds, path_data, train=toTrain)

    def create_tf_dataset_for_client_fn(client_id):
      client_set = labeled_ds.skip(client_id*CLIENT_SIZE).take(CLIENT_SIZE)
      return client_set   


    # split into train and test client data
    client_data = tff.simulation.ClientData.from_clients_and_fn(client_ids,create_tf_dataset_for_client_fn) 

    client_train, client_test = tff.simulation.ClientData.train_test_client_split(client_data, NUM_TEST_CLIENTS)
    
    federated_train_data = make_federated_data(client_train, client_ids_train)
    federated_test_data = make_federated_data(client_test, client_ids_test)
    
    
    sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                         next(iter(federated_train_data[0])))
    

    def model_fn():
        # We _must_ create a new model here, and _not_ capture it from an external
        # scope. TFF will call this within different graph contexts.
        #model = tf.keras.models.clone_model(pretrained_model)
        model = tf.keras.models.clone_model(pretrained_model)
        return tff.learning.from_keras_model(
            model,
            dummy_batch=sample_batch,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[keras.metrics.BinaryAccuracy()])
    
    fed_avg = tff.learning.build_federated_averaging_process(  model_fn, 
     client_optimizer_fn=lambda: tf.keras.optimizers.RMSprop(lr=base_learning_rate/10))

    evaluation = tff.learning.build_federated_evaluation(model_fn)
    
    print("Starting federated training")
    data_pretrained = []
    with Timer("Federated training"):
        # The state of the FL server, containing the model and optimization state.
        state = fed_avg.initialize()
        
        #intitialize the state with pretrained model weights
        state = tff.learning.state_with_new_model_weights(
        state,
        trainable_weights=[v.numpy() for v in pretrained_model.trainable_weights],
        non_trainable_weights=[
            v.numpy() for v in pretrained_model.non_trainable_weights]) 
        init_metrics = evaluation(state.model, federated_test_data)
        print('Initial model: {0:f} \n'.format( init_metrics[0]))  
        for round_num in range((int)(NUM_ROUNDS)):
          state, train_metrics = fed_avg.next(state, federated_train_data)    
          test_metrics = evaluation(state.model, federated_test_data)
          print('{0:2d}, {1:f}, {2:f}, {3:f}, {4:f} \n'.format(round_num, train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1]))  

          
if __name__ == "__main__":
    main()
