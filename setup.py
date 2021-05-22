from setuptools import setup

setup(name='mbs',
      version='0.1',
      description='Utilities for MB Scientific analyzer data',
      url='http://github.com/mueslo/mbs',
      author='mueslo',
      author_email='mueslo@mueslo.de',
      license='GPLv3',
      packages=['mbs'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'pandas',
      ],
      zip_safe=False)