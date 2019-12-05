import tensorflow as tf
import tensorflow_datasets as tfds
import sys
sys.path.append('..')
from trainer.config import config

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = config.buffer_size
BATCH_SIZE = config.bs
IMG_WIDTH = config.in_w
IMG_HEIGHT = config.in_h

def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image

# normalizing the images to [0, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image

def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)
    # random mirroring
    image = tf.image.random_flip_left_right(image)
    return image

def preprocess_image_train(image, label):
    image = random_jitter(image)
    if IMG_HEIGHT != 256:
        image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = normalize(image)
    return image

def preprocess_image_test(image, label):
    if IMG_HEIGHT != 256:
        image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = normalize(image)
    return image

def generate_dataset(ds_name='cycle_gan/horse2zebra'):
    dataset, metadata = tfds.load(ds_name,
                                  with_info=True, as_supervised=True)

    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']

    train_horses = train_horses.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(2)

    train_zebras = train_zebras.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(2)

    test_horses = test_horses.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(2)

    test_zebras = test_zebras.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(2)

    return train_horses, train_zebras, test_horses, test_zebras

def generate_images(model, test_input, z1=None):
    predictions = []
    for j in range(5):
        z1 = tf.random.normal((1, 8))
        predictions.append(model([test_input, z1])[0])
        
    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], predictions[0], predictions[1], predictions[2], predictions[3], predictions[4]]
    title = ['Input Image', 'Predicted Image']

    for i in range(4):
        plt.subplot(1, 4, i+1)
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()

def generate_malaria(train_ds, test_ds):
    train0 = []
    train1 = []
    test0 = []
    test1 = []

    for ele in train_ds:
      if ele['label'] == 0:
        train0.append(ele['image'])
      else:
        train1.append(ele['image'])

    for ele in test_ds:
      if ele['label'] == 0:
        test0.append(ele['image'])
      else:
        test1.append(ele['image'])
    
    def gen_train0():
      for i in train0:
        yield (i, 0)
    def gen_train1():
      for i in train1:
        yield (i, 1)
    def gen_test0():
      for i in test0:
        yield (i, 0)
    def gen_test1():
      for i in test1:
        yield (i, 1)

    train_pos = tf.data.Dataset.from_generator(gen_train0, (tf.uint8, tf.int64), (tf.TensorShape([None, None, 3]), tf.TensorShape([])))
    train_neg = tf.data.Dataset.from_generator(gen_train1, (tf.uint8, tf.int64), (tf.TensorShape([None, None, 3]), tf.TensorShape([])))
    test_pos = tf.data.Dataset.from_generator(gen_test0, (tf.uint8, tf.int64), (tf.TensorShape([None, None, 3]), tf.TensorShape([])))
    test_neg = tf.data.Dataset.from_generator(gen_test1, (tf.uint8, tf.int64), (tf.TensorShape([None, None, 3]), tf.TensorShape([])))
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BUFFER_SIZE = 1000
    BATCH_SIZE = 1
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    
    def preprocess_malaria_train(image, label):
        image = random_jitter(image)
        image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = normalize(image)
        return image
    def preprocess_malaria_test(image, label):
        
        image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = normalize(image)
        return image

    train_neg = train_neg.map(preprocess_malaria_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
                              BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(1)
    train_pos = train_pos.map(preprocess_malaria_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
                              BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(1)
    test_neg = test_neg.map(preprocess_malaria_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
                              BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(1)
    test_pos = test_pos.map(preprocess_malaria_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
                              BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(1)
    
    return train_neg, train_pos, test_neg, test_pos