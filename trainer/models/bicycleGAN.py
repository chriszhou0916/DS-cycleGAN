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

class BicycleGAN:
    def __init__(self, e=None, g=None, d=None, shape=(None, None, 3)):
        self.shape = shape
        if e is None or g is None or d is None:
            raise Exception('e, g, and d cannot be None')
        self.e = e
        self.g = g
        self.d = d

def compile(self, optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=[]):
    # unfinished
    self.optimizer = optimizer
    self.metrics = metrics

    img_A = tf.keras.layers.Input(shape=self.shape)
    img_B = tf.keras.layers.Input(shape=self.shape)

    z = np.random.normal(size=(8,)) # where?

    # conditional VAE-GAN: B -> z -> B'
    z_encoded_mu, z_encoded_log_sigma, z_encoded = self.e(img_B)
    fake_B_encoded = self.g(img_A, z_encoded)

    # conditional LR-GAN: z -> B' -> z'
    fake_B = self.g(img_A, z)
    z_reconstr_mu, z_reconstr_log_sigma, z_reconstr = self.e(fake_B)

    # discriminate
    discrim_B = self.d(image_b)
    discrim_fake_B = self.d(fake_b)
    discrim_fake_B_encoded = self.d(fake_B_encoded)

    # is this the right way to do the losses?
    self.combined = tf.keras.Model(inputs = [img_A, img_B],
                                   outputs = [discrim_B, discrim_fake_B, discrim_fake_B_encoded,
                                              fake_B_encoded, z_encoded, z_encoded_mu, z_encoded_log_sigma])
