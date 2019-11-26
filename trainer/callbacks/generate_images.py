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
import argparse
import numpy as np

class GenerateImages(tf.keras.callbacks.Callback):
    def __init__(self, forward, datasetA, datasetB, log_dir, interval=1000, postfix='val', z_shape=(8,)):
        super()
        self.step_count = 0
        self.postfix = postfix
        self.interval = interval
        self.forward = forward
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.datasetA = iter(datasetA)
        self.datasetB = iter(datasetB)
        self.z_shape = z_shape

    def generate_images(self):
        real_A = next(self.datasetA)
        real_B = next(self.datasetB)
        z1 = tf.random.normal((1, self.z_shape[0]))
        fake_B = self.forward.predict([real_A, z1])
        fake_B = tf.clip_by_value(fake_B, 0, 1).numpy()

        with self.summary_writer.as_default():
            tf.summary.image('{}/fake_B'.format(self.postfix), fake_B, step=self.step_count)
            tf.summary.image('{}/real_A'.format(self.postfix), real_A, step=self.step_count)

    def on_batch_begin(self, batch, logs={}):
        self.step_count += 1
        if self.step_count % self.interval == 0:
            self.generate_images()

    def on_train_end(self, logs={}):
        self.generate_images()
