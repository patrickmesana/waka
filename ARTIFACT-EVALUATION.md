# Artifact Appendix

Paper title: **WaKA: Data Attribution using K-Nearest Neighbors and Membership Privacy Principles**

Artifacts HotCRP Id: 28

Requested Badge: Available

## Description
This artifact contains the official implementation of the WaKA (Wasserstein K-nearest-neighbors Attribution) algorithm as described in the paper "WaKA: Data Attribution using K-Nearest Neighbors and Membership Privacy Principles". WaKA is a novel attribution method that leverages principles from the LiRA (Likelihood Ratio Attack) framework and k-nearest neighbors classifiers (k-NN). It efficiently measures the contribution of individual data points to the model's loss distribution, analyzing every possible k-NN that can be constructed using the training set, without requiring subset sampling.


### Security/Privacy Issues and Ethical Concerns (All badges)
This artifact contains implementations of membership inference attacks (MIA) which are used to evaluate privacy risks in machine learning models. These attacks are implemented for research purposes to assess and improve privacy protection mechanisms. No actual sensitive data or malware is included. The experiments use standard ML datasets for evaluation purposes.

The artifact implements privacy attack methods that could potentially be misused if applied to real-world systems without proper authorization. Users should only apply these methods to their own data or with explicit permission for security auditing purposes.

## Basic Requirements 

### Hardware Requirements
- CPU: Multi-core processor recommended for parallel computation (4+ cores)
- Memory: At least 8GB RAM recommended for larger datasets
- Storage: Approximately 5GB of free disk space for results and intermediate files
- No specific hardware requirements, works on standard desktop/laptop computers

### Software Requirements
**Operating System:** Linux, macOS, or Windows with Python support

**Required Software:**
- Python 3.7 or higher
- pip package manager

**Dependencies (automatically installed via requirements.txt):**
- numpy (≤2.0.0)
- scipy
- scikit-learn
- pandas
- tqdm
- matplotlib
- numba
- seaborn
- ray (for parallelization)


### Estimated Time and Storage Consumption
**Time Estimates:**
- Basic environment setup: 5-10 minutes
- Complete reproduction of all paper results: 72 hours



## Environment 

### Accessibility (All badges)
The artifact is available via the following persistent sources:
- **GitHub Repository:** https://github.com/[username]/waka (to be updated with actual repository)
- **arXiv Paper:** https://arxiv.org/abs/2411.01357





1. **Clone the repository:**
```bash
git clone https://github.com/[username]/waka.git
cd waka
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
pip install -U "ray[default]"
```

3. **Ray parallelization:**
```bash
pip install -e ../rayPlus/ --config-settings editable_mode=strict
```


### Main Results

#### Main Result 1: WaKA Performance for Membership Inference Attacks
WaKA demonstrates performance very close to LiRA when used as a membership inference attack on k-NN classifiers, but with greater computational efficiency. This is supported by experiments comparing attack success rates across different datasets and k-NN configurations.

**Paper Reference:** Section 5.1 (Privacy Evaluation and Auditing)


#### Main Result 2: WaKA Robustness for Data Minimization  
WaKA shows greater robustness than Shapley Values for data minimization tasks (both removal and addition) on imbalanced datasets. The method effectively identifies high-value data points for utility-driven data minimization.

**Paper Reference:** Section 5.2 (Utility-Driven Data Minimization)  

#### Main Result 3: Self-Attribution Correlation with Privacy Risk
Self-attribution values computed by WaKA are more strongly correlated with attack success rates than the contribution of a point to model generalization, demonstrating WaKA's effectiveness in bridging data attribution and privacy risk assessment.

**Paper Reference:** Section 4.3 (Analysis of Results)










