from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="mrsynth2",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="MRI sequence synthesis with GANs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mrsynth2",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.5b0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.2",
            "myst-parser>=0.14.0",
        ],
        "registration": [
            "antspyx>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mrsynth-train=scripts.train:main",
            "mrsynth-predict=scripts.predict:main",
            "mrsynth-evaluate=scripts.evaluate:main",
            "mrsynth-optimize=scripts.optimize:main",
            "mrsynth-preprocess=scripts.preprocess:main",
        ],
    },
)