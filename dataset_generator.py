from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from PIL import ImageFile

# global
input_dir = 'input/all'
seed = 1
validation_split = 0.7


ImageFile.LOAD_TRUNCATED_IMAGES = True

data_gen_args = dict(rescale=1/255.0,
                    zoom_range=0.2,
                    rotation_range=30.,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    validation_split=validation_split)

color_datagen = ImageDataGenerator(**data_gen_args)
bw_datagen = ImageDataGenerator(**data_gen_args)


train_color_generator = color_datagen.flow_from_directory(input_dir, class_mode=None, seed=seed, subset='training')
train_bw_generator = bw_datagen.flow_from_directory(input_dir, color_mode='grayscale', class_mode=None, seed=seed, subset='training')
valid_color_generator = color_datagen.flow_from_directory(input_dir, class_mode=None, seed=seed, subset='validation')
valid_bw_generator = bw_datagen.flow_from_directory(input_dir, color_mode='grayscale', class_mode=None, seed=seed, subset='validation')

train_generator = zip(train_bw_generator, train_color_generator)
validation_generator = zip(valid_bw_generator, valid_color_generator)