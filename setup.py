from setuptools import setup, find_packages

setup(
    name="liquid_neural_framework",
    version="0.1.0",
    description="Continuous-time adaptive neural networks with JAX for robotics and control",
    author="Daniel Schmidt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "jax>=0.4.0", 
        "jaxlib>=0.4.0",
        "equinox>=0.11.0",
        "diffrax>=0.4.0",
        "optax>=0.1.0",
        "torch>=2.0.0",
        "scipy>=1.7.0"
    ],
    python_requires=">=3.8",
)
