from setuptools import setup

setup(
    name='keras-piecewise-pooling',
    version='0.6',
    packages=['keras_piecewise_pooling'],
    url='https://github.com/CyberZHG/keras-piecewise-pooling',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='Piecewise pooling layer in Keras',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
