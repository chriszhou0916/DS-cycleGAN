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
from tensorflow.python.keras import backend as K
import numpy as np

class MultiLRScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, *args, training_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_models = training_models

    def set_new_lr(self, new_lr):
        for model in self.training_models:
            K.set_value(model.optimizer.lr, new_lr)

    def get_lr(self):
        return K.get_value(self.training_models[0].optimizer.lr)
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
          raise ValueError('The output of the "schedule" function '
                           'should be float.')
        set_new_lr(lr)
        if self.verbose > 0:
          print('\nEpoch %05d: LearningRateScheduler reducing learning '
                'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = self.get_lr()
