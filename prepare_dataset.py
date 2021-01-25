import tensorflow as tf
import pathlib
import config
from config import image_height, image_width, channels

def load_and_process_image(img_path):
    # read
    img = tf.io.read_file(img_path)
    # decode
    img = tf.image.decode_jpeg(img, channels=channels)
    # resize
    img = tf.image.resize(img, [image_height, image_width])
    img = tf.cast(img, tf.float32)
    # normalization
    img = img / 255.0
    return img

def get_images_and_labels(data_root_dir):
    # get all images' paths
    data_root = pathlib.Path(data_root_dir)
    all_img_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels name
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # label to index => dict: {label:index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_img_label = [label_to_index[pathlib.Path(img_path).parent.name] for img_path in all_img_path]
    
    return all_img_path, all_img_label

def get_dataset(dataset_root_dir):
    all_img_path, all_img_label = get_images_and_labels(dataset_root_dir)
    # create image dataset
    image_dataset = tf.data.Dataset.from_tensor_slices(all_img_path).map(load_and_process_image)
    # create label dataset
    label_dataset = tf.data.Dataset.from_tensor_slices(all_img_label)
    # combine image and label dataset with zip
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    img_count = len(all_img_path)
    return dataset, img_count

def generate_datasets():
    train_dataset, train_count = get_dataset(config.train_dir)
    valid_dataset, valid_count = get_dataset(config.valid_dir)
    test_dataset, test_count = get_dataset(config.test_dir)

    # read the original_dataset in the form of batch
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)
    
    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count