import os
from setuptools import setup,find_packages
version="0.1.6.4"
README = os.path.join(os.path.dirname(__file__),"README")
long_description = open(README).read() + '\n\n'
setup(name="pyssp",
      version=version,
      description = ('python speech signal processing library for education'),
      long_description = long_description,
      classifiers = ["Programming Language :: Python","Topic :: Scientific/Engineering","Intended Audience :: Science/Research"],
      keywords = 'scipy, speech processing',
      author = 'Shunsuke Aihara',
      author_email = "s.aihara gmail com",
      url = 'https://bitbucket.org/ynil/pyssp/',
      license = "CC3.0-BY-SA",
      packages = find_packages(),
      namespace_packages = [],
      install_requires=["numpy","scipy"]
      )
