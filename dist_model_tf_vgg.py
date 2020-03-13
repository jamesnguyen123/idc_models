import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import time
keras = tf.keras
IMG_SHAPE = (50, 50, 3)
BATCH_SIZE = 32
DATASET_SIZE = 24257
TRAIN_SIZE = int(0.8 * DATASET_SIZE)
VALIDATION_SIZE = int(0.1 * DATASET_SIZE)
TEST_SIZE = DATASET_SIZE - TRAIN_SIZE - VALIDATION_SIZE
initial_epochs = 10
validation_steps=20
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs
# Timer helper class for benchmarking 
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

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == '1'
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

def log(path, history, history_fine, num_devices):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	acc += history_fine.history['accuracy']
	val_acc += history_fine.history['val_accuracy']

	loss += history_fine.history['loss']
	val_loss += history_fine.history['val_loss']

	plt.figure(figsize=(8, 8))
	plt.subplot(2, 1, 1)
	plt.plot(acc, label='Training Accuracy')
	plt.plot(val_acc, label='Validation Accuracy')
	#plt.ylim([0.8, 1])
	plt.plot([initial_epochs-1,initial_epochs-1],
	          plt.ylim(), label='Start Fine Tuning')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(2, 1, 2)
	plt.plot(loss, label='Training Loss')
	plt.plot(val_loss, label='Validation Loss')
	#plt.ylim([0, 1.0])
	plt.plot([initial_epochs-1,initial_epochs-1],
	         plt.ylim(), label='Start Fine Tuning')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.xlabel('epoch')
	plt.savefig(path+"/logs/plot_dev"+str(num_devices) + ".png")      
	print(history.history)
	print(history_fine.history)

def main():
	path = sys.argv[1]
	list_ds = tf.data.Dataset.list_files(path + '/data/balanced_IDC_30k/*/*')
	labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	labeled_ds.shuffle(DATASET_SIZE)
	train_ds = labeled_ds.take(TRAIN_SIZE)
	validation_ds = labeled_ds.skip(TRAIN_SIZE).take(VALIDATION_SIZE)
	test_ds = labeled_ds.skip(TRAIN_SIZE + VALIDATION_SIZE).take(TEST_SIZE)
	train_batches = prepare_for_training(train_ds)
	validation_batches = prepare_for_training(validation_ds)
	test_batches = prepare_for_training(test_ds)
	base_learning_rate = 0.001
	mirrored_strategy = tf.distribute.MirroredStrategy()
	num_devices = mirrored_strategy.num_replicas_in_sync
	with mirrored_strategy.scope():
	    # Create the base model from the pre-trained model MobileNet V2
	    base_model = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE,
	                                                    include_top=False,
	                                                    weights='imagenet')
	    base_model.trainable = False
	    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
	    prediction_layer = keras.layers.Dense(1)
	    model = tf.keras.Sequential([
	        base_model,
	        global_average_layer,
	        prediction_layer
	    ])
	    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
	                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	                  metrics=['accuracy'])

	loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
	with Timer("Pre-training with " + str(num_devices) + " devices"):
		history = model.fit(train_batches,
		                    epochs=initial_epochs,
		                    validation_data=validation_batches, verbose=0)
	      

	base_model.trainable = True
	# Let's take a look to see how many layers are in the base model
	print("Number of layers in the base model: ", len(base_model.layers))

	# Fine-tune from this layer onwards
	fine_tune_at = 15

	with mirrored_strategy.scope():
	    # Freeze all the layers before the `fine_tune_at` layer
	    for layer in base_model.layers[:fine_tune_at]:
	        layer.trainable =  False
	    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	                  optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
	                  metrics=['accuracy'])

	with Timer("Fine-tuning with " + str(num_devices) + " devices"):
		history_fine = model.fit(train_batches,
		                         epochs=total_epochs,
		                         initial_epoch =  history.epoch[-1],
		                         validation_data=validation_batches, verbose=0)
	log(path, history, history_fine, num_devices)	

if __name__ == "__main__":
    main()
