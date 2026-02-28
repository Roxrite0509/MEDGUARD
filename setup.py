from setuptools import setup, find_packages

setup(
    name="qhi-probe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["scikit-learn>=1.0", "numpy>=1.21", "pandas>=1.3", "scipy>=1.7"],
    author="Pranav",
    description="QHI-Probe: Quantified Hallucination Index for Clinical LLMs",
    url="https://github.com/Roxrite0509/QHI",
    license="MIT",
)
