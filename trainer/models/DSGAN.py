"""
Copyright Ouwen Huang 2019

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
import numpy as np

class DSGAN:
    def __init__(self, g_AB=None, g_BA=None, d_B=None, d_A=None, shape = (None, None, 3), z_shape=(8,)):
        self.shape = shape
        self.z_shape = z_shape

        if d_A is None or d_B is None or g_AB is None or g_BA is None:
            raise Exception('d_A, d_B, g_AB, or g_BA cannot be None and must be a `tf.keras.Model`')
        self.d_A = d_A
        self.d_B = d_B
        self.g_AB = g_AB
        self.g_BA = g_BA

    def compile(self, lr_g=2e-4, lr_d=2e-5, metrics=[], d_loss='mse',
                g_loss = [
                    'mse', 'mse',
                    'mae', 'mae',
                    'mae', 'mae'
                ], loss_weights = [
                     1,  1,
                    10, 10,
                     1,  1,
                     1,  1
                ]):
        self.z1 = tf.random.normal(self.z_shape)
        self.z2 = tf.random.normal(self.z_shape)
        self.optimizer = tf.keras.optimizers.Adam(lr_g, .5)
        self.d_optimizer = tf.keras.optimizers.Adam(lr_d, .5)
        self.metrics = metrics
        self.d_loss = d_loss
        self.g_loss = g_loss + [self.ds_loss, self.ds_loss] # append ds_loss onto generator loss
        self.loss_weights = loss_weights

        self.d_A.trainable = True
        self.d_B.trainable = True
        self.d_A.compile(loss=self.d_loss, optimizer=self.d_optimizer, metrics=['accuracy'], loss_weights=[.5])
        self.d_B.compile(loss=self.d_loss, optimizer=self.d_optimizer, metrics=['accuracy'], loss_weights=[.5])

        # Build the generator block
        self.d_A.trainable = False
        self.d_B.trainable = False

        img_A = tf.keras.layers.Input(shape=self.shape)   # Input images from both domains
        img_B = tf.keras.layers.Input(shape=self.shape)
        rand_z1 = tf.keras.layers.Input(shape=self.z_shape)
        fake_B = self.g_AB([img_A, rand_z1])                         # Translate images to the other domain
        fake_A = self.g_BA([img_B, rand_z1])
        reconstr_A = self.g_BA([fake_B, rand_z1])                    # Translate images back to original domain
        reconstr_B = self.g_AB([fake_A, rand_z1])
        img_A_id = self.g_BA([img_A, rand_z1])                       # Identity mapping of images
        img_B_id = self.g_AB([img_B, rand_z1])

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = tf.keras.Model(inputs  = [img_A, img_B, rand_z1],
                                       outputs = [valid_A, valid_B,
                                                  reconstr_A, reconstr_B,
                                                  img_A_id, img_B_id,
                                                  fake_A, fake_B])
                                                  # EDIT
        self.combined.compile(loss=self.g_loss,
                              loss_weights=self.loss_weights,
                              optimizer=self.optimizer)


    def validate(self, validation_steps):
        metrics_summary = {}
        for metric in self.metrics:
            metrics_summary[metric.__name__] = []

        for step in range(validation_steps):
            # val_batch = next(self.dataset_val_next)
            A_batch = next(self.dataset_val_a_next)
            B_batch = next(self.dataset_val_b_next)
            fake_B = self.g_AB.predict([A_batch, self.z1])

            for metric in self.metrics:
                metric_output = metric(tf.constant(B_batch), tf.constant(fake_B)).numpy()
                metrics_summary[metric.__name__].append(metric_output[0])

        # average all metrics
        for key, value in metrics_summary.items():
            self.log['val_' + key] = np.mean(value)

        return metrics_summary

    def _fit_init(self, dataset_a, dataset_b, batch_size, steps_per_epoch, epochs, validation_data, callbacks, verbose):
        """Initialize Callbacks and Datasets"""
        self.stop_training = False # Flag for early stopping

        if not hasattr(self, 'dataset_a_next'):
            self.dataset_a_next = iter(dataset_a)
            self.dataset_b_next = iter(dataset_b)
            # self.dataset_a_next = dataset_a
            # self.dataset_b_next = dataset_b
            metric_names = ['d_loss', 'd_acc', 'g_loss', 'adv_loss', 'recon_loss', 'id_loss', 'lr', 'ds_loss']
            metric_names.extend([metric.__name__ for metric in self.metrics])

        if not hasattr(self, 'dataset_val_a_next') and validation_data is not None:
            self.dataset_val_a_next = iter(validation_data[0])
            self.dataset_val_b_next = iter(validation_data[1])
            metric_names.extend(['val_' + name for name in metric_names])

        for callback in callbacks:
            callback.set_model(self.g_AB) # only set callbacks to the forward generator
            callback.set_params({
                'verbose': verbose,
                'epochs': epochs,
                'steps': steps_per_epoch,
                'metrics': metric_names # for tensorboard callback to know which metrics to log
            })

        self.log = {
            'size': batch_size,
            'd_loss': 0,
            'd_acc': 0,
            'g_loss': 0,
            'adv_loss': 0,
            'recon_loss': 0,
            'id_loss': 0,
            'ds_loss': 0
        }
    # EDIT
    def mae(self, x, y): return tf.reshape(tf.math.reduce_mean(tf.math.abs((x - y))), [-1])

    def ds_loss(self, fake_z1, fake_z2):
        img_diff = self.mae(fake_z1, fake_z2)
        z_diff = self.mae(self.z1, self.z2)
        return -img_diff / (z_diff + 1e-5)

    # @tf.function
    def train_step(self):
        a_batch = next(self.dataset_a_next)
        b_batch = next(self.dataset_b_next)

        self.z1 = tf.random.normal((1, self.z_shape[0]))
        self.z2 = tf.random.normal((1, self.z_shape[0]))

        self.patch_gan_size = (a_batch.shape[0],) + self.d_A.get_output_shape_at(0)[1:]
        self.valid = tf.ones(self.patch_gan_size)
        self.fake = tf.zeros(self.patch_gan_size)

        # Translate images to opposite domain
        fake_B = self.g_AB([a_batch, self.z1])
        fake_A = self.g_BA([b_batch, self.z1])

        fake_A_z2 = self.g_BA([b_batch, self.z2])
        fake_B_z2 = self.g_AB([a_batch, self.z2])

        # Train the discriminators (original images = real / translated = Fake)
        dA_loss_real = self.d_A.train_on_batch(a_batch, self.valid)
        dA_loss_fake = self.d_A.train_on_batch(fake_A, self.fake)
        dA_loss = dA_loss_real[0] + dA_loss_fake[0]

        dB_loss_real = self.d_B.train_on_batch(b_batch, self.valid)
        dB_loss_fake = self.d_B.train_on_batch(fake_B, self.fake)
        dB_loss = dB_loss_real[0] + dB_loss_fake[0]

        d_loss = 0.5 * (dA_loss + dB_loss)

        g_loss = self.combined.train_on_batch([a_batch, b_batch, self.z1],
                                              [self.valid, self.valid,
                                               a_batch, b_batch,
                                               a_batch, b_batch,
                                               fake_A_z2, fake_B_z2])

        self.log['d_loss'] = d_loss
        self.log['d_acc'] = 100/4*(dA_loss_real[1] + dA_loss_fake[1] + dB_loss_real[1] + dB_loss_fake[1])
        self.log['g_loss'] = g_loss[0]
        self.log['adv_loss'] = tf.math.reduce_mean(g_loss[1:3])
        self.log['recon_loss'] = tf.math.reduce_mean(g_loss[3:5])
        self.log['id_loss'] = tf.math.reduce_mean(g_loss[5:7])
        self.log['ds_loss'] = tf.math.reduce_mean(g_loss[7:])

    def fit(self, dataset_a, dataset_b, batch_size=1, steps_per_epoch=10, epochs=3, validation_data=None, verbose=1, validation_steps=10,
            callbacks=[]):
        self._fit_init(dataset_a, dataset_b, batch_size, steps_per_epoch, epochs, validation_data, callbacks, verbose)
        for callback in callbacks: callback.on_train_begin(logs=self.log)
        for epoch in range(epochs):
            for callback in callbacks: callback.on_epoch_begin(epoch, logs=self.log)
            for step in range(steps_per_epoch):
                for callback in callbacks: callback.on_batch_begin(step, logs=self.log)
                self.train_step()
                for callback in callbacks: callback.on_batch_end(step, logs=self.log)
            # step = 0
            # for image_x, image_y in tf.data.Dataset.zip((self.dataset_a_next, self.dataset_b_next)):
            #     for callback in callbacks: callback.on_batch_begin(step, logs=self.log)
            #     self.train_step(image_x, image_y)
            #     for callback in callbacks: callback.on_batch_end(step, logs=self.log)
            #     step += 1
            if validation_data is not None:
                forward_metrics = self.validate(validation_steps)

            for callback in callbacks: callback.on_epoch_end(epoch, logs=self.log)
        for callback in callbacks: callback.on_train_end(logs=self.log)
