SAVE_FOLDER = "./pre-trained/model/" # folder to save model
LOG_FOLDER = "./pre-trained/log/"  # folder to save tensorboard log

# Dataset
TRAINING_DATA_DIR = "/media/csu/data/CSU/2021/publications-2021/self-driving(tran)/data/udacity" #"./data"
TRAINING_CSV_FILE_DIR = "../" # file path related to TRAINING_DATA_DIR, see README for format details
TRAINING_DATA_FILE = "driving_log.csv" # file path related to TRAINING_DATA_DIR, see README for format details
LANE_DATA_FILE = "lane_infor_file.csv" # file path related to TRAINING_DATA_DIR, see README for format details


ANGLE_DELTA_CORRECTION_LEFT  = 0.2 # steering angle correction for the left camera image
ANGLE_DELTA_CORRECTION_RIGHT = -0.2
INPUT_IMAGE_CROP = [60, -25, 0, 319] # Keeping regions in the input image in order [start_x, end_x, start_y, end_y]


# Training Parameters
KEEP_PROB = 0.5      # keep probability used by dropout layer
REGULARIZER_COEF = 0 # coefficient used by L2 regularizer
BATCH_SIZE = 128     # training batch size 

LOG_INTERVAL = 1
SAVE_INTERVAL = 1000
