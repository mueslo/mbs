from setuptools import setup

setup(name='mbs',
      version='0.1',
      description='The funniest joke in the world',
      url='http://github.com/mueslo/mbs',
      author='mueslo',
      author_email='mueslo@mueslo.de',
      license='GPLv3',
      packages=['mbs'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
      ],
      zip_safe=False)