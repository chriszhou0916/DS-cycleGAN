import tensorflow as tf
class MyInstanceNorm(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(MyInstanceNorm, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super(MyInstanceNorm, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)
        mean, variance = tf.nn.moments(inputs, [1, 2], keepdims=True)
        weight_shape = self._create_broadcast_shape(input_shape)
        expanded_beta, expanded_gamma = self._get_reshaped_weights(input_shape, weight_shape, broadcast=False)
        outputs = tf.nn.batch_normalization(inputs, mean, variance, offset=expanded_beta, scale=expanded_gamma,
                                            variance_epsilon=self.epsilon)
        
        return outputs
    
    def _get_reshaped_weights(self, input_shape, weight_shape, broadcast=False):
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, weight_shape)
        if self.center:
            beta = tf.reshape(self.beta, weight_shape)
        return gamma, beta
    
    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * (len(input_shape) - 1)
        broadcast_shape[self.axis] = input_shape[self.axis]
        return broadcast_shape
    
    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name='gamma',
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint)
        else:
            self.gamma = None
    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name='beta',
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint)
        else:
            self.beta = None

    def get_config(self):
        config = {
            'axis':
            self.axis,
            'epsilon':
            self.epsilon,
            'center':
            self.center,
            'scale':
            self.scale,
            'beta_initializer':
            tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer':
            tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer':
            tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer':
            tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint':
            tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint':
            tf.keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super(MyInstanceNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
