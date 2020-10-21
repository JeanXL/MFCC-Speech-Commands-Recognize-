from easydict import EasyDict as edict
import datetime

cfg = edict()

cfg.num_classes = 4
cfg.epochs = 30
cfg.batch_size = 64
cfg.save_mode_path = "tfv230_kws.h5"
cfg.log_dir = ".\\logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  

# MFCC options
cfg.mfcc = edict()
cfg.mfcc.samp_freq = 8000
cfg.mfcc.frame_shift_ms = 20
cfg.mfcc.frame_length_ms = 32
cfg.mfcc.pre_emphasis = 0.97


# training options
cfg.train = edict()
cfg.train.num_samples = 7017
cfg.train.learning_rate = 1e-3
cfg.train.dataset = "./tfrecords/train.tfrecords"

# training options
cfg.val = edict()
cfg.val.num_samples = 1757
cfg.val.dataset = "./tfrecords/val.tfrecords"


