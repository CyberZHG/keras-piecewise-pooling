
Keras Piecewise Pooling
=======================


.. image:: https://travis-ci.org/CyberZHG/keras-piecewise-pooling.svg
   :target: https://travis-ci.org/CyberZHG/keras-piecewise-pooling
   :alt: Travis


.. image:: https://coveralls.io/repos/github/CyberZHG/keras-piecewise-pooling/badge.svg?branch=master
   :target: https://coveralls.io/github/CyberZHG/keras-piecewise-pooling
   :alt: Coverage


.. image:: https://img.shields.io/pypi/pyversions/keras-piecewise-pooling.svg
   :target: https://pypi.org/project/keras-piecewise-pooling/
   :alt: PyPI


Piecewise pooling layer in Keras.


.. image:: https://user-images.githubusercontent.com/853842/45488448-07e08e80-b794-11e8-8b67-ae650aa017b5.png
   :target: https://user-images.githubusercontent.com/853842/45488448-07e08e80-b794-11e8-8b67-ae650aa017b5.png
   :alt: 


Install
-------

.. code-block:: bash

   pip install keras-piecewise-pooling

Layers
------

``PiecewisePooling1D``
^^^^^^^^^^^^^^^^^^^^^^^^^^

The layer is used for pooling sequential data with given slicing positions:

.. code-block:: python3

   import keras
   import numpy as np
   from keras_piecewise_pooling import PiecewisePooling1D


   data = [[[1, 3, 2, 5], [7, 9, 2, 3], [0, 1, 7, 2], [4, 7, 2, 5]]]
   positions = [[1, 3, 4]]
   piece_num = len(positions[0])

   data_input = keras.layers.Input(shape=(None, None))
   position_input = keras.layers.Input(shape=(piece_num,), dtype='int32')
   pool_layer = PiecewisePooling1D(
       piece_num=piece_num,
       pool_type=PiecewisePooling1D.POOL_TYPE_AVERAGE,
   )([data_input, position_input])
   model = keras.models.Model(inputs=[data_input, position_input], outputs=pool_layer)
   model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)
   model.summary()

   print(model.predict([np.asarray(data), np.asarray(positions)]).tolist())
   # The result will be:
   # [[
         [1.0, 3.0, 2.0, 5.0],
         [3.5, 5.0, 4.5, 2.5],
         [4.0, 7.0, 2.0, 5.0],
   # ]]

``PiecewisePooling1D`` has two input layers, the first is the layer to be processed, the second is the layer representing positions. The last column of the positions must be the lengths of the sequences.
