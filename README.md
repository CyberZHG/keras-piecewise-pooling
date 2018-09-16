# Keras Piecewise Pooling

[![Travis](https://travis-ci.org/CyberZHG/keras-piecewise-pooling.svg)](https://travis-ci.org/CyberZHG/keras-piecewise-pooling)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-piecewise-pooling/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-piecewise-pooling)
[![PyPI](https://img.shields.io/pypi/pyversions/keras-piecewise-pooling.svg)](https://pypi.org/project/keras-piecewise-pooling/)

Piecewise pooling layer in Keras.

![](https://user-images.githubusercontent.com/853842/45488448-07e08e80-b794-11e8-8b67-ae650aa017b5.png)

## Install

```bash
pip install keras-piecewise-pooling
```

## `PiecewisePooling1D`

### Basic

The layer is used for pooling sequential data with given slicing positions:

```python3
import keras
import numpy as np
from keras_piecewise_pooling import PiecewisePooling1D


data = [[[1, 3, 2, 5], [7, 9, 2, 3], [0, 1, 7, 2], [4, 7, 2, 5]]]
positions = [[1, 3, 4]]
piece_num = len(positions[0])

data_input = keras.layers.Input(shape=(None, None))
position_input = keras.layers.Input(shape=(piece_num,), dtype='int32')
pool_layer = PiecewisePooling1D(pool_type=PiecewisePooling1D.POOL_TYPE_AVERAGE)([data_input, position_input])
model = keras.models.Model(inputs=[data_input, position_input], outputs=pool_layer)
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)
model.summary()

print(model.predict([np.asarray(data), np.asarray(positions)]).tolist())
# The result will be close to:
# [[
#     [1.0, 3.0, 2.0, 5.0],
#     [3.5, 5.0, 4.5, 2.5],
#     [4.0, 7.0, 2.0, 5.0],
# ]]
```

`PiecewisePooling1D` has two input layers, the first is the layer to be processed, the second is the layer representing positions. The last column of the positions must be the lengths of the sequences.

### Custom

You can write your own pooling functions:

```python
PiecewisePooling1D(pool_type=lambda x: K.min(x, axis=1))
```

### Load

Remember to set `custom_objects`:

```python
keras.models.load_model(model_path, custom_objects=PiecewisePooling1D.get_custom_objects())
```
