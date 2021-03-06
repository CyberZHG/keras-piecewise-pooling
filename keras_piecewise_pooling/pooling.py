import keras
import keras.backend as K
from keras_piecewise import Piecewise, Piecewise2D


class MaxPool1D(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MaxPool1D, self).__init__(**kwargs)

    def call(self, inputs):
        return K.max(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[2:]


class AvePool1D(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(AvePool1D, self).__init__(**kwargs)

    def call(self, inputs):
        return K.sum(inputs, axis=1) / (K.cast(K.shape(inputs)[1], K.floatx()) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[2:]


class PiecewisePooling1D(Piecewise):

    POOL_TYPE_MAX = 'max'
    POOL_TYPE_AVERAGE = 'average'

    def __init__(self,
                 pool_type=POOL_TYPE_MAX,
                 **kwargs):
        self.pool_type = pool_type
        if callable(self.pool_type):
            layer = keras.layers.Lambda(self.pool_type)
        elif pool_type == self.POOL_TYPE_MAX:
            layer = MaxPool1D()
        elif pool_type == self.POOL_TYPE_AVERAGE:
            layer = AvePool1D()
        else:
            raise NotImplementedError('No implementation for pooling type : ' + pool_type)
        self.supports_masking = True
        super(PiecewisePooling1D, self).__init__(layer, **kwargs)

    def get_config(self):
        config = {
            'pool_type': self.pool_type,
        }
        base_config = super(PiecewisePooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config.pop('layer')
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {
            'Piecewise': Piecewise,
            'PiecewisePooling1D': PiecewisePooling1D,
        }


class MaxPool2D(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MaxPool2D, self).__init__(**kwargs)

    def call(self, inputs):
        return K.max(inputs, axis=[1, 2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[3:]


class AvePool2D(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(AvePool2D, self).__init__(**kwargs)

    def call(self, inputs):
        sums = K.sum(inputs, axis=[1, 2])
        size = K.cast(K.shape(inputs)[1] * K.shape(inputs)[2], dtype=K.floatx())
        return sums / (size + K.epsilon())

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[3:]


class PiecewisePooling2D(Piecewise2D):

    POOL_TYPE_MAX = 'max'
    POOL_TYPE_AVERAGE = 'average'

    def __init__(self,
                 pool_type=POOL_TYPE_MAX,
                 **kwargs):
        self.pool_type = pool_type
        if callable(self.pool_type):
            layer = keras.layers.Lambda(self.pool_type)
        elif pool_type == self.POOL_TYPE_MAX:
            layer = MaxPool2D()
        elif pool_type == self.POOL_TYPE_AVERAGE:
            layer = AvePool2D()
        else:
            raise NotImplementedError('No implementation for pooling type : ' + pool_type)
        self.supports_masking = True
        super(PiecewisePooling2D, self).__init__(layer, **kwargs)

    def get_config(self):
        config = {
            'pool_type': self.pool_type,
        }
        base_config = super(PiecewisePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config.pop('layer')
        return cls(**config)

    @staticmethod
    def get_custom_objects():
        return {
            'Piecewise': Piecewise,
            'PiecewisePooling2D': PiecewisePooling2D,
        }
