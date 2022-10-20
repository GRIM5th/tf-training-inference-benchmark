import os
import argparse
from absl import app
from absl import flags
import tensorflow as tf
import time
import numpy as np
from tensorboard.plugins.hparams import api as hp
import os
import argparse
from tensorflow.keras import mixed_precision




### DO NOT CHANGE ANYTHING BELOW THIS
FLAGS = flags.FLAGS
DATA_LENGTH = 1000

flags.DEFINE_boolean('use_fp16', False, 'Using mixed precision rather than single precision')
flags.DEFINE_boolean('use_bfloat16', False, 'Using mixed precision rather than single precision')
flags.DEFINE_integer('batch', 32, 'Batch used in training, default is 32', lower_bound=1)
flags.DEFINE_integer('num_epochs', 20, 'Number of training epoch, default is 20', lower_bound=1)

# creating fake data
class FakeData(object):
    def __init__(self):
        super(FakeData, self).__init__()
        self.length = DATA_LENGTH
        self.X_train = np.random.random((224, 224, 3)).astype('float32')
        zeros = np.zeros(1000)
        zeros[np.random.randint(1000)] = 1
        self.Y_train = zeros.astype('int32')

    def __iter__(self):
        for _ in range(self.length):
            yield self.X_train, self.Y_train

    def __len__(self):
        return self.length

    def output_shapes(self):
        return (self.X_train.shape, self.Y_train.shape)

    def output_types(self):
        return (tf.float32, tf.int32)

#saving time taken per step
class get_callback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.current_batch_times = []

    def on_train_batch_begin(self, batch, logs=None):
        self.start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.current_batch_times.append(time.time() - self.start)

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(np.mean(self.current_batch_times))

def get_data(df, BATCH):
    tdf = tf.data.Dataset.from_generator(
        generator=df.__iter__,
        output_types=df.output_types(),
        output_shapes=df.output_shapes())
    tdf = tdf.batch(BATCH)
    tdf = tdf.prefetch(tf.data.experimental.AUTOTUNE)
    return tdf

def get_model(df):
    model = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights=None,
        input_shape=df.output_shapes()[0],
        classes = 1000)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

def main(argv):
    if FLAGS.use_fp16:
        if FLAGS.use_bfloat16:
            print("Using bfloat16 instead of fp16")
            mixed_precision.set_global_policy('mixed_bfloat16')
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['CUDA_FORCE_PTX_JIT:'] = '0'
        else:
            print("Using fp16")
            mixed_precision.set_global_policy('mixed_float16')
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['CUDA_FORCE_PTX_JIT:'] = '0'
    else:
        if FLAGS.use_bfloat16:
            print("Using bfloat16")
            mixed_precision.set_global_policy('mixed_bfloat16')
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['CUDA_FORCE_PTX_JIT:'] = '0'
        else:
            print("Using single precision/fp32 computation")
    BATCH = FLAGS.batch
    NUM_EPOCHS = FLAGS.num_epochs
    df = FakeData()
    model = get_model(df)
    step_callback = get_callback()
    model.fit(get_data(df, BATCH), epochs=NUM_EPOCHS, callbacks=[step_callback])
    timing = np.average(([np.round(t*1000) for t in step_callback.times]), axis=0)
    print("Average time/step = " + str(timing) + " ms/step")

if __name__ == '__main__':
    app.run(main)
