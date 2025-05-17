from setuptools import setup, find_packages

setup(
    name="waka",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["waka", "utils", "knn_shapley_valuation", "privacy_evaluation_and_audit", 
               "valuation_analysis", "utility_driven_data_minimization"],
    install_requires=open("requirements.txt").read().splitlines(),
) 