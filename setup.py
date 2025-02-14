from setuptools import setup, find_packages

setup(
    name="popularl",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "gymnasium",
        "numpy",
        "stable_baselines3",  # Remove if unused
    ],
    author="Aadhavan",
    description="A Meta-RL implementation using MAML and VAE.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aksamban/popularl",  # Change to your repo
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
