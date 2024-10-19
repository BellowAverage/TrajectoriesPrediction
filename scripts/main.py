from DataPreprocessing import drop_no_labels, data_integration_individual
from KalmanFilter import kf_handle
from TrajectoryAnalysis import map_matching_by_time_interval
from MachineLearningModels import main as ml_main

def main():
    # Data preprocessing steps
    drop_no_labels()
    data_integration_individual()

    # Kalman filter handling
    # kf_handle(data)

    # Trajectory analysis
    # map_matching_by_time_interval("010")

    # Machine Learning models handling
    ml_main()


if __name__ == "__main__":
    main()