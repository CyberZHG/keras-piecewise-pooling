import os
import tempfile
import random
import unittest
import keras
import keras.backend as K
import numpy as np
from keras_piecewise_pooling import PiecewisePooling1D


class TestPool1D(unittest.TestCase):

    @staticmethod
    def _build_model(input_shape, piece_num, pool_type, pos_type=PiecewisePooling1D.POS_TYPE_SEGMENTS):
        data_input = keras.layers.Input(shape=input_shape)
        position_input = keras.layers.Input(shape=(piece_num,), dtype='int32')
        pool_layer = PiecewisePooling1D(
            pool_type=pool_type,
            pos_type=pos_type,
        )([data_input, position_input])
        model = keras.models.Model(inputs=[data_input, position_input], outputs=pool_layer)
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)
        model.summary()
        return model

    def test_max_2d(self):
        data = [
            [1, 3, 5, 2],
            [1, 3, 5, 2],
        ]
        positions = [
            [2, 4],
            [3, 4],
        ]
        model = self._build_model(
            input_shape=(None,),
            piece_num=len(positions[0]),
            pool_type=PiecewisePooling1D.POOL_TYPE_MAX,
        )
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [
            [3.0, 5.0],
            [5.0, 2.0],
        ]
        self.assertEqual(expected, predicts)
        model = self._build_model(
            input_shape=(None,),
            piece_num=len(positions[0]),
            pool_type=PiecewisePooling1D.POOL_TYPE_MAX,
            pos_type=PiecewisePooling1D.POS_TYPE_PAIRS,
        )
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [
            [5.0],
            [2.0],
        ]
        self.assertEqual(expected, predicts)

    def test_average_3d(self):
        data = [[[1, 3, 2, 5], [7, 9, 2, 3], [0, 1, 7, 2], [4, 7, 2, 5]]]
        positions = [[1, 3, 4]]
        model = self._build_model(
            input_shape=(None, None),
            piece_num=len(positions[0]),
            pool_type=PiecewisePooling1D.POOL_TYPE_AVERAGE,
        )
        predicts = model.predict([np.asarray(data), np.asarray(positions)])
        expected = np.asarray([[
            [1.0, 3.0, 2.0, 5.0],
            [3.5, 5.0, 4.5, 2.5],
            [4.0, 7.0, 2.0, 5.0],
        ]])
        self.assertTrue(np.allclose(expected, predicts))

    def test_empty_interval(self):
        data = [[[1, 3, 2, 5], [7, 4, 2, 3], [0, 1, 2, 2], [4, 7, 2, 5]]]
        positions = [[2, 2, 4]]
        model = self._build_model(
            input_shape=(None, None),
            piece_num=len(positions[0]),
            pool_type=PiecewisePooling1D.POOL_TYPE_MAX,
        )
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [[
            [7.0, 4.0, 2.0, 5.0],
            [float('-inf')] * 4,
            [4.0, 7.0, 2.0, 5.0],
        ]]
        self.assertEqual(expected, predicts)

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self._build_model(
                input_shape=(None,),
                piece_num=3,
                pool_type='not_implemented',
            )

    def test_custom_pool(self):
        data = [
            [2, 3, 5, 7],
            [2, 3, 5, 7],
        ]
        positions = [
            [2, 4],
            [3, 4],
        ]
        model = self._build_model(
            input_shape=(None,),
            piece_num=len(positions[0]),
            pool_type=lambda x: K.min(x, axis=1),
        )
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [
            [2.0, 5.0],
            [2.0, 7.0],
        ]
        self.assertEqual(expected, predicts)

    def test_save_load(self):
        data = [[[1, 5, 2, 4], [7, 9, 2, 3], [0, 1, 7, 2], [4, 8, 2, 5]]]
        positions = [[2, 4]]
        model = self._build_model(
            input_shape=(None, None),
            piece_num=len(positions[0]),
            pool_type=PiecewisePooling1D.POOL_TYPE_AVERAGE,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'keras_piece_test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=PiecewisePooling1D.get_custom_objects())
        predicts = model.predict([np.asarray(data), np.asarray(positions)])
        expected = np.asarray([[
            [4.0, 7.0, 2.0, 3.5],
            [2.0, 4.5, 4.5, 3.5],
        ]])
        self.assertTrue(np.allclose(expected, predicts))
