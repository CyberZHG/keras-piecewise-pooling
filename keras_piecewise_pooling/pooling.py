import keras
import keras.backend as K


class PiecewisePooling1D(keras.layers.Layer):

    POOL_TYPE_MAX = 'max'
    POOL_TYPE_AVERAGE = 'average'

    def __init__(self,
                 piece_num,
                 pool_type=POOL_TYPE_MAX,
                 **kwargs):
        self.piece_num = piece_num
        self.pool_type = pool_type
        self.supports_masking = True
        super(PiecewisePooling1D, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'piece_num': self.piece_num,
            'pool_type': self.pool_type,
        }
        base_config = super(PiecewisePooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(PiecewisePooling1D, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        inputs, positions = inputs
        return K.map_fn(
            lambda i: self._call_sample(inputs, positions, i),
            K.arange(K.shape(inputs)[0]),
            dtype=K.floatx(),
        )

    def _call_sample(self, inputs, positions, index):
        inputs = inputs[index]
        positions = positions[index]
        return K.map_fn(
            lambda i: self._call_piece(inputs, positions, i),
            K.arange(self.piece_num),
            dtype=K.floatx(),
        )

    def _call_piece(self, inputs, positions, index):
        piece = K.switch(
            K.equal(index, 0),
            inputs[:positions[index]],
            inputs[positions[index - 1]:positions[index]],
        )
        if self.pool_type == self.POOL_TYPE_MAX:
            return K.max(piece, axis=0)
        if self.pool_type == self.POOL_TYPE_AVERAGE:
            print(piece)
            return K.sum(piece, axis=0) / K.cast(K.shape(piece)[0], K.floatx())
        raise NotImplementedError('No implementation for poooling type : ' + self.pool_type)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.piece_num) + tuple(input_shape[0][2:])

    def compute_mask(self, inputs, mask=None):
        return None
