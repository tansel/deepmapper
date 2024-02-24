from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'scipy',
    'scikit-learn>=0.22',
    'pandas',
    'matplotlib',
    'torch',
    'captum'
]

setup(
    name='pyDeepMapper',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/tansel/pyDeepMapper',
    license='Apache 2',
    author='Tansel Ersavas',
    author_email='t.ersavas@unsw.edu.au',
    description='Analysis of non-image data using CNNs',
    install_requires=install_requires,
)
