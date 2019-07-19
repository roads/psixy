from setuptools import setup


def readme():
    """Read in README file."""
    with open('README.md') as f:
        return f.read()

setup(name='psixy',
      version='0.1.0',
      description='Toolbox for fitting psychological category learning models.',
      long_description=readme(),
      classifiers=[
          'Programming Language :: Python :: 3',
      ],
      author='Brett D. Roads',
      author_email='brett.roads@gmail.com',
      license='Apache Licence 2.0',
      packages=['psixy'],
      install_requires=['numpy', 'scipy', 'pandas', 'scikit-learn', 'h5py', 'matplotlib'],
      include_package_data=True,
      )
