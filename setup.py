"""Setup file."""
import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='psixy',
    version='0.1.0',
    description='Toolbox for fitting psychological category learning models.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    author='Brett D. Roads',
    author_email='brett.roads@gmail.com',
    license='Apache Licence 2.0',
    packages=['psixy'],
    python_requires='>=3, <3.8',
    install_requires=[
        'tf-nightly', 'tensorflow-probability', 'numpy', 'scipy',
        'pandas', 'scikit-learn', 'h5py', 'matplotlib'
    ],
    include_package_data=True,
    url='https://github.com/roads/psixy',
)
