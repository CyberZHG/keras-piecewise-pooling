import os
import tempfile
import random
import unittest
import keras
import keras.backend as K
import numpy as np
from keras_piecewise_pooling import PiecewisePooling2D


class TestPool2D(unittest.TestCase):

    @staticmethod
    def _build_model(input_shape, row_num, col_num, pool_type):
        data_input = keras.layers.Input(shape=input_shape)
        row_input = keras.layers.Input(shape=(row_num,), dtype='int32')
        col_input = keras.layers.Input(shape=(col_num,), dtype='int32')
        pool_layer = PiecewisePooling2D(
            pool_type=pool_type,
        )([data_input, row_input, col_input])
        model = keras.models.Model(inputs=[data_input, row_input, col_input], outputs=pool_layer)
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)
        model.summary()
        return model

    def test_max_2d(self):
        data = [
            [
                [1, 3, 5, 2],
                [2, 5, 6, 1],
                [7, 1, 5, 3],
                [7, 2, 2, 4],
            ],
            [
                [1, 3, 5, 2],
                [2, 5, 6, 1],
                [7, 1, 5, 3],
                [7, 2, 2, 4],
            ],
        ]
        rows = [
            [2, 4],
            [3, 4],
        ]
        cols = [
            [1, 2, 4],
            [1, 3, 4],
        ]
        model = self._build_model(
            input_shape=(None, None),
            row_num=len(rows[0]),
            col_num=len(cols[0]),
            pool_type=PiecewisePooling2D.POOL_TYPE_MAX,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'keras_piece_test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=PiecewisePooling2D.get_custom_objects())
        predicts = model.predict([np.asarray(data), np.asarray(rows), np.asarray(cols)]).tolist()
        expected = [
            [
                [2.0, 5.0, 6.0],
                [7.0, 2.0, 5.0],
            ],
            [
                [7.0, 6.0, 3.0],
                [7.0, 2.0, 4.0],
            ],
        ]
        self.assertEqual(expected, predicts)
        model = self._build_model(
            input_shape=(None, None),
            row_num=len(rows[0]),
            col_num=len(cols[0]),
            pool_type=lambda x: K.max(x, axis=[1, 2]),
        )
        predicts = model.predict([np.asarray(data), np.asarray(rows), np.asarray(cols)]).tolist()
        expected = [
            [
                [2.0, 5.0, 6.0],
                [7.0, 2.0, 5.0],
            ],
            [
                [7.0, 6.0, 3.0],
                [7.0, 2.0, 4.0],
            ],
        ]
        self.assertEqual(expected, predicts)
        model = self._build_model(
            input_shape=(None, None),
            row_num=len(rows[0]),
            col_num=len(cols[0]),
            pool_type=PiecewisePooling2D.POOL_TYPE_AVERAGE,
        )
        predicts = model.predict([np.asarray(data), np.asarray(rows), np.asarray(cols)])
        expected = np.asarray([
            [
                [1.5, 4.0, 3.5],
                [7.0, 1.5, 3.5],
            ],
            [
                [3.333333, 4.166666, 2.0],
                [6.999999, 2.0, 3.999999],
            ],
        ])
        self.assertTrue(np.allclose(expected, predicts))

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self._build_model(
                input_shape=(None, None),
                row_num=3,
                col_num=4,
                pool_type='not_implemented',
            )
