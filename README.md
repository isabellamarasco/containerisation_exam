# Containerisation_exam

This project is designed to run anomaly detection experiments using various scikit-learn models in a containerized environment. It automates the process of model training, hyperparameter tuning, and evaluation, ensuring reproducibility by tracking experiment configurations.

### How it works
The core of the project is the anomaly_detection.py script. This script performs the following steps:
1. Configuration: It reads configuration from environment variables. These include the dataset ID from OpenML, test set size, cross-validation folds, and model-specific hyperparameters.
2. Experiment ID: It computes a unique EXPERIMENT_ID based on the current experiment's configuration. This is used to prevent re-running identical experiments.
3. Data Loading & Preprocessing: It downloads the specified dataset (BoT-IoT) from OpenML, applies one-hot encoding to categorical features, and prepares the target variable for anomaly detection.
4. Model Training: It uses GridSearchCV to train and tune the specified machine learning models with the provided hyperparameters. It includes special handling for models like IsolationForest that do not require standard cross-validation.
5. Evaluation & Comparison: The script evaluates the models based on their cross-validation accuracy scores, and identifies the best-performing model.

### Containerization
The project uses Docker and Docker Compose to manage the experimental environment. The docker-compose.yml file defines services to run experiments for different models in isolation:
- experiment-rf: Tests a RandomForestClassifier.
- experiment-mlp: Tests an MLPClassifier.

Each service is built from a common Dockerfile and runs the same Python script, but with different environment variables to specify the model and its hyperparameters.

### Pre-requisites
- Docker and Docker Compose installed on your machine.
- Access to OpenML datasets (the script uses the KDD-99 dataset by default).

### Running the Project
To run the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone
   ```
2. Navigate to the project directory:
   ```bash
   cd Containerisation_exam
   ```
3. Build the Docker images:
   ```bash
    docker-compose build
    ```
4. Run the experiments:
    ```bash
    docker-compose up
    ```
5. Monitor the output logs to see the results of each experiment.
6. After the experiments complete, you can check the results in the `results` directory.

