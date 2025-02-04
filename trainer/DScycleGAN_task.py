"""
Copyright Chris Zhou, Leo Hu, Ouwen Huang 2019

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under t
]


he License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
from trainer.config import config
from trainer.utils import dataset
from trainer import utils
from trainer.models import networks
from trainer import models
from trainer import callbacks

LOG_DIR = config.job_dir
MODEL_DIR = config.model_dir

# Load Data (Build your custom data loader and replace below)
train_X, train_Y, test_X, test_Y = dataset.generate_dataset(config.ds_name)
dataset_count = config.ds_count
# Select and Compile Model
g_AB = networks.bicycle_generator(img_shape=(config.in_h, config.in_w, 3), z_shape=(config.latent_z_dim,), norm=config.generator_norm, skip=False, z_in='all')
g_BA = networks.bicycle_generator(img_shape=(config.in_h, config.in_w, 3), z_shape=(config.latent_z_dim,), norm=config.generator_norm, skip=False, z_in='all')
d_A = networks.create_discriminator(shape=(config.in_h, config.in_w, 3), norm=config.discriminator_norm)
d_B = networks.create_discriminator(shape=(config.in_h, config.in_w, 3), norm=config.discriminator_norm)

# Load existing models

# Compile model
model = models.DSGAN(shape = (None, None, 3),
                        g_AB=g_AB,
                        g_BA=g_BA,
                        d_B=d_B,
                        d_A=d_A)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
              d_loss='mse',
              g_loss = [
                 'mse', 'mse',
                 'mae', 'mae',
                 'mae', 'mae'
              ], loss_weights = [
                 config.disc_loss,  config.disc_loss,
                 config.cycle_consistency_loss, config.cycle_consistency_loss,
                 config.id_loss,  config.id_loss,
                 config.ds_loss, config.ds_loss
              ])
def scheduler(epoch):
  if epoch < config.startLRdecay:
    return 2e-4
  else:
    epochs_passed = epoch - config.startLRdecay
    decay_step = 2e-4 / (config.epochs - config.startLRdecay)
    return 2e-4 - epochs_passed * decay_step

LRscheduler = callbacks.MultiLRScheduler(scheduler, training_models=[model.d_A, model.d_B, model.combined])
# Generate Callbacks
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=True, update_freq='epoch')
start_tensorboard = callbacks.StartTensorBoard(LOG_DIR)

prog_bar = tf.keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=None)
log_code = callbacks.LogCode(LOG_DIR, './trainer')
copy_keras = callbacks.CopyKerasModel(MODEL_DIR, LOG_DIR)

saving = callbacks.MultiModelCheckpoint(MODEL_DIR + '/model.{epoch:02d}-{accuracy:.10f}.hdf5',
                                        monitor='accuracy', verbose=1, freq='epoch', mode='max', save_best_only=False,
                                        save_weights_only=True,
                                        multi_models=[('g_AB', g_AB), ('g_BA', g_BA), ('d_A', d_A), ('d_B', d_B)])

image_gen = callbacks.GenerateImages(g_AB, test_X, test_Y, LOG_DIR, interval=int(dataset_count/config.bs), z_shape=(config.latent_z_dim,))

# Fit the model
model.fit(train_X, train_Y,
    batch_size=config.bs,
    steps_per_epoch=(dataset_count // config.bs),
          epochs=config.epochs,
          validation_data=(test_X, test_Y),
          validation_steps=10,
          callbacks=[log_code, tensorboard, prog_bar, image_gen, saving,
                     copy_keras, start_tensorboard, LRscheduler])
