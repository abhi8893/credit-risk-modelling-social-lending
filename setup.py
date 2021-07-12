from setuptools import setup, find_packages


setup(
    author='Abhishek Bhatia',
    author_email='bhatiaabhishek8893@gmail.com',
    name='src',
    version='0.0.1',
    packages=find_packages(include=['src', 'src.*']),
    python_requires='>=3.6.*'
)