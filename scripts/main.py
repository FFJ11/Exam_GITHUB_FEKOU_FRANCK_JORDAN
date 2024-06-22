import os

if __name__ == "__main__":
    # Data preparation
    os.system("python data_preparation.py")

    # Data split
    os.system("python data_split.py")

    # Data scaling
    os.system("python data_scaling.py")

    # Model training
    os.system("python model_training.py")

    # Model evaluation
    os.system("python model_evaluation.py")
