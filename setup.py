#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
import re
from setuptools import setup, find_packages


with open(os.path.join('cellmaps_image_embedding', '__init__.py')) as ver_file:
    for line in ver_file:
        line = line.rstrip()
        if line.startswith('__version__'):
            version = re.sub("'", "", line[line.index("'"):])
        elif line.startswith('__description__'):
            desc = re.sub("'", "", line[line.index("'"):])
        elif line.startswith('__repo_url__'):
            repo_url = re.sub("'", "", line[line.index("'"):])

print(repo_url)

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['cellmaps_utils',
                'tqdm',
                'pillow',
                'numpy',
                'pandas>0.23.1',
                'torch',
                'torchvision',
                'opencv-python',
                'mlcrate',
                'scikit-image',
                'scikit-learn>=0.19.0']

setup_requirements = []

test_requirements = []

setup(
    author="Gege Qian",
    author_email='geqian@ucsd.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description=desc,
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='cellmaps_image_embedding',
    name='cellmaps_image_embedding',
    packages=find_packages(include=['cellmaps_image_embedding']),
    package_dir={'cellmaps_image_embedding': 'cellmaps_image_embedding'},
    scripts=['cellmaps_image_embedding/cellmaps_image_embeddingcmd.py'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url=repo_url,
    version=version,
    zip_safe=False)
