# setup.py
from setuptools import setup, find_packages

setup(
    name='protodyn',
    version='0.2.0',
    author='Pratik Patil & Dr. Bhushan Bonde',
    author_email='p.patil@uos.ac.uk',
    description='A package for studying protein dynamics using Graph Neural Network',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pratikp204/protodyn',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    install_requires=[
        'MDAnalysis>=2.7.0',
        'pandas>=2.2.2',
        'requests>=2.32.3',
        'torch>=2.5.0',
        'torch_geometric>=2.6.1',
        'OpenMM==8.1.1',
        "rich>=13.9.2",
        # "dash>=2.17.0",
        # "dash-bootstrap-components>=1.6.0",
        # "dash-core-components>=2.0.0",
        # "dash-html-components>=2.0.0",
        # "dash-molstar==1.1.2"
    ],
    extras_require={
        "dev": [
            "pytest",  # Development dependencies
            "black",
            "flake8",
        ]
    },
    entry_points={
        'console_scripts': [
            'protodyn=protodyn.__main__:main',
        ],
    },
)
