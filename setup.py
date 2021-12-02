from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='Bishop',
      version='3',
      description='Cognitive model of Theory of Mind',
      long_description=readme(),
      url='http://gibthub.com/julianje/bishop',
      author='Julian Jara-Ettinger',
      license='MIT',
      packages=['Bishop'],
      scripts=['bin/Bishop'],
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'argparse',
          'Image'
      ],
      include_package_data=True,
      zip_safe=False)
