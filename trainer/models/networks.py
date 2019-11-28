import tensorflow as tf
from trainer.utils.MyInstanceNorm import MyInstanceNorm

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

    def get_config(self):
        config = {
            'padding':
            self.padding
        }
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
#     x = tf.keras.layers.Conv2DTranspose(output_dim, kernel_size=7, strides=1, padding='same', activation='tanh')(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.scalar_mul(.5, x) + .5)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
#     return x

def unet_downsample(input_tensor, filters, size, apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=True)(input_tensor)
    if apply_norm:
        x = normalization(x, method='instance')
    x = tf.keras.layers.Activation(tf.nn.leaky_relu)(x)
    return x

def unet_upsample(input_tensor, filters, size, apply_dropout=False, last=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=True)(input_tensor)
    x = normalization(x, method='instance')
    if apply_dropout:
        x = tf.keras.layers.Dropout(0.5)(x)
    if last:
        x = tf.keras.layers.Activation(tf.nn.tanh)(x)
    else:
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
    return x

def create_unet_generator(shape=(256, 256, 3)):
    inputs = tf.keras.layers.Input(shape=shape)
    down1 = unet_downsample(inputs, 64, 4, apply_norm=False) # 128,128,64
    down2 = unet_downsample(down1, 128, 4) # 64,64,128
    down3 = unet_downsample(down2, 256, 4) # 32,32,256
    down4 = unet_downsample(down3, 512, 4) # 16,16,512
    down5 = unet_downsample(down4, 512, 4) # 8,8,512
    down6 = unet_downsample(down5, 512, 4) # 4,4,512
    down7 = unet_downsample(down6, 512, 4) # 2,2,512
    down8 = unet_downsample(down7, 512, 4) # 1,1,512
    up1 = unet_upsample(down8, 512, 4, apply_dropout=True) # 2,2,512
    up1 = tf.keras.layers.Concatenate()([up1, down7]) # 2,2,1024
    up2 = unet_upsample(up1, 512, 4, apply_dropout=True) # 4,4,512
    up2 = tf.keras.layers.Concatenate()([up2, down6]) # 4,4,1024
    up3 = unet_upsample(up2, 512, 4, apply_dropout=True) # 8,8,512
    up3 = tf.keras.layers.Concatenate()([up3, down5]) # 8,8,1024
    up4 = unet_upsample(up3, 512, 4) # 16,16,512
    up4 = tf.keras.layers.Concatenate()([up4, down4]) # 16,16,1024
    up5 = unet_upsample(up4, 256, 4) # 32,32,256
    up5 = tf.keras.layers.Concatenate()([up5, down3]) # 32,32,512
    up6 = unet_upsample(up5, 128, 4) # 64,64,128
    up6 = tf.keras.layers.Concatenate()([up6, down2]) # 64,64,256
    up7 = unet_upsample(up6, 64, 4) # 128,128,64
    up7 = tf.keras.layers.Concatenate()([up7, down1]) # 128,128,128
    up8 = unet_upsample(up7, 3, 4, last=True) # 256,256,3
    x = tf.keras.layers.Lambda(lambda x: tf.math.scalar_mul(.5, x) + .5)(up8)
    return tf.keras.Model(inputs=inputs, outputs=x)

def dis_downsample(input_tensor,
               kernel_size,
               filters,
               stride, norm=None):
  initializer = tf.random_normal_initializer(0., 0.02)
  p = 1
  # x = ReflectionPadding2D(padding=(p, p))(input_tensor)
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
    # x = ReflectionPadding2D(padding=(1, 1))(x)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same', kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def create_LSdiscriminator(shape=(256, 256, 3)):
    inputs = tf.keras.layers.Input(shape=shape)
    x = dis_downsample(inputs, 5, 64, 2, norm=None)
    x = dis_downsample(x, 5, 128, 2, norm='instance')
    x = dis_downsample(x, 5, 256, 2, norm='instance')
    x = dis_downsample(x, 5, 512, 2, norm='instance')
    x = tf.keras.layers.Dense(1, activation='linear')(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def bicycle_encoder_convnet(shape=(256, 256, 3), norm='instance'):
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=shape)
    down1 = dis_downsample(inputs, 4, 64, 2, norm=None) # 128,128,64
    down2 = dis_downsample(down1, 4, 128, 2, norm=norm) # 64,64,128
    down3 = dis_downsample(down2, 4, 256, 2, norm=norm) # 32,32,256
    down4 = dis_downsample(down3, 4, 512, 2, norm=norm) # 16,16,512
    down5 = dis_downsample(down4, 4, 512, 2, norm=norm) # 8,8,512
    down6 = dis_downsample(down5, 4, 512, 2, norm=norm) # 4,4,512
    down7 = dis_downsample(down6, 4, 512, 2, norm=norm) # 2,2,512
    if shape[0] > 255:
        down7 = dis_downsample(down7, 4, 512, 2, norm=norm) # 1,1,512
    x = tf.keras.layers.Flatten()(down7)
    mu = tf.keras.layers.Dense(8)(x)
    log_sigma = tf.keras.layers.Dense(8)(x)
    z = mu + tf.random.normal(shape=(8,))*tf.exp(log_sigma)
    return tf.keras.Model(inputs=inputs, outputs=[mu, log_simga, z])

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
    x = conv_w_reflection(x, 3, 256, 2, norm=norm)
    if z_in == 'all':
        new_dim = int(new_dim / 2)
        z_rv = tf.keras.layers.RepeatVector(new_dim*new_dim)(z_input)
        z_rs = tf.keras.layers.Reshape([new_dim, new_dim, z_shape[0]])(z_rv)
        x = tf.keras.layers.Concatenate()([x, z_rs])
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
