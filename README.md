# Green-Lab-xxx

# Greener ML Software

### Overview
The **Greener ML Software** project aims to evaluate the energy efficiency of different machine learning (ML) frameworks. By comparing energy consumption and performance, our goal is to help developers select the most energy-efficient ML frameworks for specific algorithms without sacrificing performance.

This project compares **scikit-learn**, **sklearnex**, and **XGBoost** across various algorithms, leveraging energy-monitoring tools like **PyJoules** to measure energy consumption in real-time. We also use performance metrics such as execution time, CPU utilization, and accuracy to balance energy consumption with ML task efficiency.

---

### Project Goals
- **Energy Efficiency**: Measure and compare the energy consumption of different ML frameworks running the same algorithms.
- **Performance**: Analyze the trade-off between energy consumption and performance.
- **Framework Optimization**: Identify frameworks and configurations that provide optimal energy efficiency without significant performance loss.

---

### Frameworks and Algorithms
This project evaluates the following ML frameworks and algorithms:

- **Frameworks**:
  - Scikit-learn
  - Sklearnex

- **Algorithms**:
  - K-Means
  - Support Vector Machine (SVM)
  - Linear Regression
  - Logistic Regression
  - DBSCAN
  - Random Forest
  - GradientBoostingClassifier
  - GradientBoostingRegressor

---

### Tools and Technologies
- **PyJoules**: Used for real-time energy monitoring of the CPU and memory.
- **Scikit-learn_bench**: A benchmark testing tool for evaluating the performance of various machine learning algorithms.
- **Intel RAPL Interface**: Provides detailed power monitoring for the system's energy consumption during the experiment.

---

### Experiment Setup
The experiments are performed on a Linux workstation with the following configurations:
- **OS**: Ubuntu 22.04.2 LTS (GNU/Linux 5.15.0-84-generic x86_64)
- **CPU**: Intel i7-13650HX (14 cores, 20 threads)
- **RAM**: 32GB
- **Storage**: 1TB SSD

Each algorithm is executed 10 times in different frameworks to measure energy consumption, performance, and accuracy. Results are averaged across runs to reduce variability.

---

### How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourRepo/GreenerMLSoftware.git
   cd GreenerMLSoftware

Here is your content converted to markdown format:

```
markdown复制代码## Install Dependencies:

```bash
pip install -r requirements.txt
```

## Run Experiments:

```
bash

python run_experiment.py
```

## Analyze Results:

Collected data will be saved in the `results` directory. Use the provided `analysis_tools.py` script to generate plots and reports based on the results.

## Results

The results include:

- **Energy Consumption**: Measured in joules using PyJoules.
- **Accuracy**: Evaluated for each model after training.
- **Execution Time**: Captured for each run.
- **Energy Efficiency Ratio (EER)**: Computed as the amount of work performed per unit of energy consumed.

## Future Work

We plan to extend the project by experimenting with more complex ML models and evaluating energy consumption in GPU-based environments.

## Contributors

- WenZhi Zhang
- Hongyu Chen
- Ziniu Jin
- Weinuo Huang
- XinYi Lu
- Guanghe Xie
