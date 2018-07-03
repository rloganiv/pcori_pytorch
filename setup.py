from setuptools import find_packages, setup


setup(name='pcori_pytorch',
      description='PyTorch models and code for the PCORI project.',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      packages=find_packages(exclude=["*.tests", "*.tests.*",
                                      "tests.*", "tests"]),
      install_requires=[
          'allennlp==0.5.1'
      ],
      tests_require=['pytest', 'flaky'],
      python_requires='>=3.6')

