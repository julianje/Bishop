from setuptools import setup

def readme():
	with open('README.rst') as f:
		return f.read()

setup(name='bishop',
	version='0.01',
	description='Bayesian Theory of Mind library',
	url='http://gibthub.com/julianje/bishop',
	author='Julian Jara-Ettinger',
	author_email='jjara@mit.edu',
	license='MIT',
	packages=['bishop'],
	install_requires=[
		'numpy',
		'math',
		'random'
	],
	scripts=['bin/bishop'],
	zip_safe=False)