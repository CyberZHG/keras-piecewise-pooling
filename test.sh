#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_piecewise_pooling tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=keras_piecewise_pooling --with-doctest