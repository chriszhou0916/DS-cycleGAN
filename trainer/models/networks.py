import tensorflow as tf
from trainer.utils.MyInstanceNorm import MyInstanceNorm
from trainer.utils.ReflectionPadding2D import ReflectionPadding2D

def normalization(intput_tensor, method='instance'):
  if method == 'instance':
    x = MyInstanceNorm(center=True, scale=True,
                                                  beta_initializer="random_uniform",
                                                  gamma_initializer="random_uniform")(intput_tensor)
  else:
    x = tf.keras.layers.BatchNormalization()(intput_tensor)
  return x

def conv_w_reflection(input_tensor,
               kernel_size,
               filters,
               stride,
               norm='instance'):
  initializer = tf.random_normal_initializer(0., 0.02)
  p = kernel_size // 2
  x = ReflectionPadding2D(padding=(p, p))(input_tensor)
  x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, use_bias=True, kernel_initializer=initializer)(x)
  x = normalization(x, method=norm)
  x = tf.keras.layers.Activation(tf.nn.relu)(x)
  return x

def conv_block(input_tensor, filters, norm='instance'):
  initializer = tf.random_normal_initializer(0., 0.02)
  x = ReflectionPadding2D(padding=(1, 1))(input_tensor)
  x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=(1, 1), use_bias=True, kernel_initializer=initializer)(x)
  x = normalization(x, method=norm)
  x = tf.keras.layers.Activation(tf.nn.relu)(x)
  x = ReflectionPadding2D(padding=(1, 1))(x)
  x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=(1, 1), use_bias=True, kernel_initializer=initializer)(x)
  x = normalization(x, method=norm)
  return x

def residual_block(input_tensor, filters, norm='instance'):
  b1 = conv_block(input_tensor, filters, norm=norm)
  x = tf.keras.layers.Add()([input_tensor, b1])
  return x

def upsample_conv(input_tensor, kernel_size, filters, stride, norm='instance'):
  initializer = tf.random_normal_initializer(0., 0.02)
  x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=stride, padding='same', use_bias=True, kernel_initializer=initializer)(input_tensor)
  x = normalization(x, method=norm)
  x = tf.keras.layers.Activation(tf.nn.relu)(x)
  return x

def create_generator(shape=(256, 256, 3), norm='instance', skip=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=shape)
    x = conv_w_reflection(inputs, 7, 64, 1, norm=norm)
    x = conv_w_reflection(x, 3, 128, 2, norm=norm)
    x = conv_w_reflection(x, 3, 256, 2, norm=norm)
    x = residual_block(x, 256, norm=norm)
    x = residual_block(x, 256, norm=norm)
    x = residual_block(x, 256, norm=norm)

    if shape[0] > 255:
        x = residual_block(x, 256, norm=norm)
        x = residual_block(x, 256, norm=norm)
        x = residual_block(x, 256, norm=norm)

    x = residual_block(x, 256, norm=norm)
    x = residual_block(x, 256, norm=norm)
    x = residual_block(x, 256, norm=norm)
    x = upsample_conv(x, 3, 128, 2, norm=norm)
    x = upsample_conv(x, 3, 64, 2, norm=norm)
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = tf.keras.layers.Conv2D(3, 7, strides=1, kernel_initializer=initializer)(x)
    if skip:
        x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.Activation(tf.nn.tanh)(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.scalar_mul(.5, x) + .5)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def dis_downsample(input_tensor,
               kernel_size,
               filters,
               stride, norm=None):
  initializer = tf.random_normal_initializer(0., 0.02)
  p = 1
  x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_initializer=initializer)(input_tensor)
  if norm is not None:
    x = normalization(x, method=norm)
  x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  return x

def create_discriminator(shape=(256, 256, 3), norm=None):
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=shape)
    x = dis_downsample(inputs, 4, 64, 2, norm=None)
    x = dis_downsample(x, 4, 128, 2, norm=norm)
    x = dis_downsample(x, 4, 256, 2, norm=norm)
    x = dis_downsample(x, 4, 512, 1, norm=norm)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same', kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def bicycle_generator(img_shape=(256, 256, 3), z_shape=(8,), norm='instance', skip='false', z_in='first'):
    initializer = tf.random_normal_initializer(0., 0.02)

    img_input = tf.keras.layers.Input(shape=img_shape)
    z_input = tf.keras.layers.Input(shape=z_shape)

    img_size = img_shape[0]
    z_rv = tf.keras.layers.RepeatVector(img_size*img_size)(z_input)
    z_rs = tf.keras.layers.Reshape([img_size, img_size, z_shape[0]])(z_rv)
    inputs = tf.keras.layers.Concatenate()([img_input, z_rs])

    x = conv_w_reflection(inputs, 7, 64, 1, norm=norm)
    x = conv_w_reflection(x, 3, 128, 2, norm=norm)
    if z_in == 'all':
        new_dim = int(img_size / 2)
        z_rv = tf.keras.layers.RepeatVector(new_dim*new_dim)(z_input)
        z_rs = tf.keras.layers.Reshape([new_dim, new_dim, z_shape[0]])(z_rv)
        x = tf.keras.layers.Concatenate()([x, z_rs])

    if z_in == 'all':
        x = conv_w_reflection(x, 3, 256-z_shape[0], 2, norm=norm)
        new_dim = int(new_dim / 2)
        z_rv = tf.keras.layers.RepeatVector(new_dim*new_dim)(z_input)
        z_rs = tf.keras.layers.Reshape([new_dim, new_dim, z_shape[0]])(z_rv)
        x = tf.keras.layers.Concatenate()([x, z_rs])
    else:
        x = conv_w_reflection(x, 3, 256, 2, norm=norm)
    x = residual_block(x, 256, norm=norm)
    x = residual_block(x, 256, norm=norm)
    x = residual_block(x, 256, norm=norm)

    if img_shape[0] > 255:
        x = residual_block(x, 256, norm=norm)
        x = residual_block(x, 256, norm=norm)
        x = residual_block(x, 256, norm=norm)

    x = residual_block(x, 256, norm=norm)
    x = residual_block(x, 256, norm=norm)
    x = residual_block(x, 256, norm=norm)
    x = upsample_conv(x, 3, 128, 2, norm=norm)
    x = upsample_conv(x, 3, 64, 2, norm=norm)
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = tf.keras.layers.Conv2D(3, 7, strides=1, kernel_initializer=initializer)(x)
    if skip:
        x = tf.keras.layers.Add()([x, img_input])
    x = tf.keras.layers.Activation(tf.nn.tanh)(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.scalar_mul(.5, x) + .5)(x)

    return tf.keras.Model(inputs=[img_input, z_input], outputs=x)

def bicycle_discriminator(shape=(256, 256, 3), norm=None):
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=shape)
    x = dis_downsample(inputs, 4, 64, 2, norm=None)
    x = dis_downsample(x, 4, 128, 2, norm=norm)
    x = dis_downsample(x, 4, 256, 2, norm=norm)
    x = dis_downsample(x, 4, 512, 1, norm=norm)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same', kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
