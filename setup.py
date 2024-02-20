from setuptools import setup, find_packages

setup(
    name='spacer3d',
    version='0.1.0',
    packages=find_packages(),
    description='3D Spatial Pattern Analysis with Comparable and Extendable Ripley’s K',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='dkermany',
    url='https://github.com/dkermany/spacer3d',
    install_requires=[
        # Any dependencies your project needs, e.g.,
        'numpy>=1.26.4',
        'matplotlib>=3.8.2',
        'ipympl>=0.9.3',
        'seaborn>=0.13.2',
        'pandas>=2.2.0',
        'scipy>=1.12.0',
        'tqdm>=4.66.2',
        'pynrrd>=1.0.0',
        'raster_geometry>=0.1.4.2',
        'oiffile>=2023.8.30',
        'tifffile>=2024.2.12',
        'opencv-python>=4.9.0.80'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)

