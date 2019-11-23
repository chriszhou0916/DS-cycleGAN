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

import time
import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # Input parser
    parser.add_argument('--bs',       default=1,    type=int, help='batch size')
    parser.add_argument('--in_h',     default=256,  type=int, help='image input size height')
    parser.add_argument('--in_w',     default=256,  type=int, help='image input size width')
    parser.add_argument('--epochs',   default=10,  type=int, help='number of epochs')
    parser.add_argument('--m',        default=True, type=bool, help='manual run or hp tuning')
    parser.add_argument('--is_test',  default=False, type=bool, help='is test')
    parser.add_argument('--cycle_consistency_loss', default=10, type=int, help='cycle consistency loss weight')
    parser.add_argument('--disc_loss', default=1, type=int, help='discriminators loss weight')
    parser.add_argument('--id_loss', default=5, type=int, help='identity loss weight')
    parser.add_argument('--buffer_size', default=1000, type=int, help='dataset shuffle buffer size')
    parser.add_argument('--generator_norm', default='instance', help='what kind of normalization to use')
    parser.add_argument('--discriminator_norm', default='instance', help='what kind of normalization to use')
    parser.add_argument('--startLRdecay', default=100, type=int, help='When to start linearly decaying LR')

    parser.add_argument('--ds_name', default='horse2zebra', help='what kind of normalization to use')
    parser.add_argument('--ds_count', default=1067, type=int, help='what kind of normalization to use')
    # Cloud ML Params
    parser.add_argument('--job-dir', default='gs://duke-bme590-cz/ds-cyclegan/tmp/{}'.format(str(time.time())), help='Job directory for Google Cloud ML')
    parser.add_argument('--model_dir', default='./trained_models', help='Directory for trained models')

    parsed, unknown = parser.parse_known_args()

    print('Unknown args:', unknown)
    print('Parsed args:', parsed.__dict__)

    return parsed

config = get_config()
