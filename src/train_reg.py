from __future__ import print_function
import os
import random


from keras.callbacks import EarlyStopping, TensorBoard
from keras.applications.densenet import DenseNet121
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Lambda
from keras import optimizers

from CNN_train_package.utils.data_augment import *
from CNN_train_package.utils.gt_fns import *
from CNN_train_package.utils.loss_fns import *
from CNN_train_package.utils.data_generation import *
from CNN_train_package.output_constraint import *
from CNN_train_package.hyper_params import batch_size, input_shape, num_epoch


PROJECT_FILE_PATH = (dirname(dirname((dirname(abspath(__file__)))))) + "/"
TRAIN_DIRECTORY = None
VALIDATION_DIRECTORY = None
SAVE = PROJECT_FILE_PATH + "data/training/regression/"


###################################################################
# Get filepaths for training and validation
###################################################################


d = '/home/ben/datasets/sun360/levelled_images'
levelled_paths = [os.path.join(d, x) for x in os.listdir(d)]
seed = 1
random.seed(seed)
random.shuffle(levelled_paths)
# Build the model for training
dataset = 'sun360'
model_name = 'paper_model_' + dataset

if dataset == 'sun360':
    n_train = int((1. - 0.11111111111111)*len(levelled_paths))
else:
    n_train = int(0.9*len(levelled_paths))

train_filenames = levelled_paths[:n_train]
valid_filenames = levelled_paths[n_train:]


###################################################################
# Define training model: network, loss and optimiser
###################################################################


loss_function = tf.keras.losses.LogCosh()#angle_spherical
constraint_function = sph_clipping
num_output = 2
gt_type = give_gt_sph_reg
base_model = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=input_shape,
    pooling='avg')
cnn_output = base_model.output
final_output = Dense(num_output, name='fc_layer')(cnn_output)  # always take care of activation!!!!
constraint_layer = Lambda(constraint_function)
model = Model(input=base_model.input, output=final_output)

# training aparatus
learning_rate = 0.00001
adam = optimizers.Adam(lr=learning_rate)
model.compile(
    loss=loss_function,
    optimizer=adam,
    metrics=[arc_error])
model.summary()

print('model compiled')


###################################################################
# define some folders for output
###################################################################


# Save
output_folder = SAVE + model_name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_folder = output_folder + "/"

output_log = output_folder + "Log"
if not os.path.exists(output_log):
    os.makedirs(output_log)
output_log = output_log + "/"

output_weight = output_folder + "Best"
if not os.path.exists(output_weight):
    os.makedirs(output_weight)
output_weight = output_weight + "/"

model_json = model.to_json()

with open(output_folder + "model.json", "w") as json_file:
    json_file.write(model_json)


###################################################################
# keras callbacks
###################################################################


tensorboard = TensorBoard(log_dir=output_log)

checkpointer = CustomModelCheckpoint(
    model_for_saving=model,
    filepath=output_weight + "weights_{epoch:02d}_{val_loss:.2f}.h5",
    save_best_only=True,
    monitor='val_loss',
    save_weights_only=True,
    period=5
)


###################################################################
# data generators
###################################################################


training_generator = RotNetDataGenerator(
    gt_function=gt_type,
    data_augmentation_fn=IEEE_VR_combo,
    input_shape=input_shape,
    batch_size=batch_size,
    one_hot=False,
    preprocess_func=preprocess_input,
    shuffle=True,
    noise=False).generate(train_filenames)
valid_generator = RotNetDataGenerator(
    gt_function=gt_type,
    data_augmentation_fn=IEEE_VR_combo,
    input_shape=input_shape,
    batch_size=batch_size,
    one_hot=False,
    preprocess_func=preprocess_input,
    shuffle=False,
    noise=False).generate(valid_filenames)


###################################################################
# train
###################################################################


model.fit_generator(
    generator=training_generator,
    steps_per_epoch=(len(train_filenames) // batch_size),
    epochs=num_epoch,
    validation_data=valid_generator,
    validation_steps=(len(valid_filenames) // batch_size),
    callbacks=[tensorboard, checkpointer],
    use_multiprocessing=True,
    workers=8,
    verbose=1)
