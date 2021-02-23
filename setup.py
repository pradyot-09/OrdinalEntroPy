#
# Copyright (C) 2021 Pradyot Patil

DESCRIPTION = "OrdinalEntroPy: Ordinal entropy metods of time-series in Python"
DISTNAME = 'OrdinalEntroPy'
MAINTAINER = 'Pradyot Patil'
MAINTAINER_EMAIL = 'pradyotpatil@gmail.com'
URL = 'https://raphaelvallat.com/entropy/build/html/index.html'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/pradyot-09/OrdinalEntroPy'
VERSION = '0.1.1'
PACKAGE_DATA = {'entropy.data.icons': ['*.ico']}

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup


def check_dependencies():
    install_requires = []

    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')

    return install_requires


if __name__ == "__main__":

    install_requires = check_dependencies()

    setup(name=DISTNAME,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=install_requires,
          include_package_data=True,
          packages=['OrdinalEntroPy'],
          package_data=PACKAGE_DATA,
          classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 3.6',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'],
          )