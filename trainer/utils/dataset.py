import tensorflow as tf
import tensorflow_datasets as tfds
AUTOTUNE = tf.data.experimental.AUTOTUNE


BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image
# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
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
  image = normalize(image)
  return image
def preprocess_image_test(image, label):
  image = normalize(image)
  return image
def generate_dataset():
    AUTOTUNE = tf.data.experimental.AUTOTUNE


    BUFFER_SIZE = 1000
    BATCH_SIZE = 1
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                                  with_info=True, as_supervised=True)

    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']

    train_horses = train_horses.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).repeat().batch(1).prefetch(buffer_size=AUTOTUNE)

    train_zebras = train_zebras.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(1).repeat().prefetch(buffer_size=AUTOTUNE)

    test_horses = test_horses.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(1).repeat().prefetch(buffer_size=AUTOTUNE)

    test_zebras = test_zebras.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(1).repeat().prefetch(buffer_size=AUTOTUNE)

    return train_horses, train_zebras, test_horses, test_zebras
