import os
import codecs
from setuptools import setup


def read(*parts):
    h = os.path.dirname(os.path.realpath(__file__))
    with codecs.open(os.path.join(h, *parts), 'rb', 'utf-8') as f:
        return f.read()


setup(
    name='exotic-miri',
    version='0.0.1',
    author='David Grant',
    author_email='david.grant@bristol.ac.uk',
    url='https://github.com/Exo-TiC/ExoTiC-MIRI',
    license='MIT',
    packages=['exotic_miri'],
    description='ExoTiC MIRI data reduction steps',
    long_description=read('README.rst'),
    long_description_content_type='text/x-rst',
    package_data={
        '': ['README.rst', 'LICENSE']
    },
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'jwst>=1.4.3'],
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    zip_safe=True,
)
