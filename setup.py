from setuptools import setup, find_packages

setup(
    name="ambond",
    version="0.1.0",
    description="Altermagnet bond checker CLI",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'ambond = ambond.ambond:cli',
        ],
    },
    install_requires=[
        'ase',
        'spglib',
        'numpy',
        'amcheck',
        'diophantine'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
