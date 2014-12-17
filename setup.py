#!/usr/bin/env python3
"""Plotypus: variable star light curve analysis and plotting library.

Plotypus is a library for the analysis of the light curves of variable
stars.

Plotypus is built on top of numpy, matplotlib, and scikit-learn.
"""

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Programming Language :: Python
Programming Language :: Python :: 3
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)
Operating System :: OS Independent
Topic :: Scientific/Engineering :: Astronomy
Topic :: Software Development :: Libraries :: Python Modules
"""

MAJOR      = 0
MINOR      = 2
MICRO      = 5
ISRELEASED = False
PRERELEASE = 1
VERSION    = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

def get_version_info():
    FULLVERSION = VERSION

    if not ISRELEASED:
        FULLVERSION += '-pre' + str(PRERELEASE)

    return FULLVERSION

def setup_package():
    metadata = dict(
        name = 'plotypus',
        url = 'https://github.com/astroswego/plotypus',
        description = DOCLINES[0],
        long_description = "\n".join(DOCLINES[2:]),
        version = get_version_info(),
        package_dir = {'': 'src'},
        packages = ['plotypus', 'plotypus.resources'],
        package_data = {'plotypus': ['resources/matplotlibrc']},
        entry_points = {
            'console_scripts': [
                'plotypus = plotypus.plotypus:main'
            ]
        },
        keywords = [
            'astronomy',
            'light curve',
            'stellar pulsation',
            'variable star'
        ],
        classifiers = [f for f in CLASSIFIERS.split('\n') if f],
        requires = [
            'numpy (>= 1.8.0)',
            'matplotlib (>= 1.4.0)',
            'scikit (>= 0.14.0)'
        ]
    )

    from setuptools import setup

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
