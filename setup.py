from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='Bishop',
      version='2.0.1',
      description='Bayesian Theory of Mind library',
      long_description=readme(),
      url='http://gibthub.com/julianje/bishop',
      author='Julian Jara-Ettinger',
      author_email='jjara@mit.edu',
      license='MIT',
      packages=['Bishop'],
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy'
      ],
      include_package_data=True,
      zip_safe=False)
