from setuptools import setup, find_packages

setup(
    name="liquid_neural_framework",
    version="0.1.0",
    description="Continuous-time adaptive neural networks with JAX for robotics and control",
    author="Daniel Schmidt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core dependencies will be added based on research needs
    ],
    python_requires=">=3.8",
)
