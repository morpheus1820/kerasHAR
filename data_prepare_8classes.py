import numpy as np


# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "acc_x_",
    "acc_y_",
    "acc_z_",
    "gyro_x_",
    "gyro_y_",
    "gyro_z_"
]

# Output classes to learn how to classify
LABELS = [
    "DRINKING",
    "BRUSHING TEETH",
    "WASHING DISHES",
    "WALKING AROUND",
    "USE PC",
    "PLAY GUITAR",
    "IDLE",
    "WASHING HANDS"
]

DATASET_PATH = "data/"

# Prepare dataset
TRAIN = "train/"
TEST = "test/"

# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'rb')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(',') for row in file
                ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial_Signals/" + signal + "train_300.txt" for signal in INPUT_SIGNAL_TYPES
    ]
X_train = load_X(X_train_signals_paths)

X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test_300.txt" for signal in INPUT_SIGNAL_TYPES
    ]
X_test = load_X(X_test_signals_paths)


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    y_ = np.loadtxt(y_path, delimiter=',', dtype=np.int32)
    y_ = y_.reshape((y_.shape[0], 1))


    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1


y_train_path = DATASET_PATH + TRAIN + "y_train_300.txt"
y_test_path = DATASET_PATH + TEST + "y_test_300.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

# Input Data

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
# test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep

