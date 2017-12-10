import glob

import tensorflow as tf
from IPython import embed

DATASET_BATCH_SIZE = 50
IMAGE_X = 200
IMAGE_Y = 200
IMAGE_N_CHANNELS = 3
FLATTEN_IMAGE = False
NUM_CATEGORIES = 2

PATH_REGEX = '/Users/sai/dev/catsdogs/data/train/*.jpg'


def _parse_images(filename, filename_2):
    """
    Helper function to read images and labels from filenames.
    filename_2 = filename
    """
    img_str = tf.read_file(filename)
    img_decoded = tf.image.decode_jpeg(img_str)
    img_resized = tf.image.resize_images(img_decoded, [IMAGE_X, IMAGE_Y])
    # if FLATTEN_IMAGE:
    #     # This tensor will have one row of (IMAGE_X * IMAGE_Y * IMAGE_N_CHANNELS) columns
    #     img_resized = tf.reshape(img_resized, [-1])

    filename_split = tf.string_split([filename_2], '.').values
    animal_str = tf.slice(filename_split, [0], [1])
    animal_str = tf.string_split(animal_str, '/').values
    animal_str = tf.slice(animal_str, [6], [1])
    is_cat = tf.equal(animal_str, tf.constant('cat'))
    is_dog = tf.equal(animal_str, tf.constant('dog'))
    is_cat = tf.cast(is_cat, tf.int64)
    is_dog = tf.cast(is_dog, tf.int64)
    label = tf.concat([is_cat, is_dog], 0)
    return img_resized, label


def get_iterator(filenames):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, filenames))
    dataset = dataset.map(_parse_images)
    # Randomly shuffle dataset with a buffer size
    dataset = dataset.shuffle(1000)
    # Infinite repeat of the dataset
    dataset = dataset.repeat()
    # Batch size
    dataset = dataset.batch(DATASET_BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    return iterator


def get_file_paths():
    training_filenames = glob.glob(PATH_REGEX)
    return training_filenames


def initialize_weights(shape):
    w = tf.truncated_normal(shape, stddev=0.1)
    w = tf.Variable(w)
    return w


def initialize_biases(shape):
    b = tf.constant(0.1, shape=shape)
    b = tf.Variable(b)
    return b


def conv2d(x, W):
    """
    Convolutional layer with stride 1 in each direction and no padding.
    i.e. output size is same as input size
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    Pooling over non overlapping 2x2 windows
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def convolution_model(x, y_, keep_prob):
    # Conv layer 1
    W_conv1 = initialize_weights(shape=[5, 5, 3, 32])   # [patch_size, patch_size, num_channels, output_depth]
    # b_conv1 = initialize_biases(shape=[IMAGE_X, IMAGE_Y, IMAGE_N_CHANNELS, 32])     # Maybe this works too?
    b_conv1 = initialize_biases(shape=[32])
    z_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    z_conv1 = max_pool_2x2(z_conv1)

    # Conv layer 2
    W_conv2 = initialize_weights(shape=[3, 3, 32, 32])  # [patch_size, patch_size, num_channels, output_depth]
    # b_conv2 = initialize_biases(shape=[IMAGE_X, IMAGE_Y, IMAGE_N_CHANNELS, 32])     # Maybe this works too?
    b_conv2 = initialize_biases(shape=[32])
    z_conv2 = tf.nn.relu(conv2d(z_conv1, W_conv2) + b_conv2)
    z_conv2 = max_pool_2x2(z_conv2)

    W_fc1 = initialize_weights(shape=[50 * 50 * 32, 512])
    b_fc1 = initialize_biases(shape=[512])
    z_conv2_flat = tf.reshape(z_conv2, [-1, 50 * 50 * 32])
    z_fc1 = tf.nn.relu(tf.matmul(z_conv2_flat, W_fc1) + b_fc1)

    z_dropout1 = tf.nn.dropout(z_fc1, keep_prob)

    W_out1 = initialize_weights(shape=[512, 64])
    b_out1 = initialize_biases(shape=[64])
    z_out1 = tf.matmul(z_dropout1, W_out1) + b_out1

    W_out2 = initialize_weights(shape=[64, 2])
    b_out2 = initialize_biases(shape=[2])
    z_out2 = tf.matmul(z_out1, W_out2) + b_out2

    y = z_out2
    return y


def feed_forward_model(x, y_):

    w1 = tf.Variable(tf.zeros([IMAGE_X * IMAGE_Y * IMAGE_N_CHANNELS, 20]))
    b1 = tf.Variable(tf.zeros(20))
    z1 = tf.matmul(x, w1) + b1

    w2 = tf.Variable(tf.zeros([20, 2]))
    b2 = tf.Variable(tf.zeros(2))
    z2 = tf.matmul(z1, w2) + b2

    y = z2
    return y


def train():
    training_filenames = get_file_paths()
    filenames = tf.placeholder(tf.string, shape=[None])
    iterator = get_iterator(filenames)

    # x = tf.placeholder(tf.float32, [None, IMAGE_X * IMAGE_Y * IMAGE_N_CHANNELS])
    x = tf.placeholder(tf.float32, [None, IMAGE_X, IMAGE_Y, IMAGE_N_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CATEGORIES])
    keep_prob = tf.placeholder(tf.float32)

    y = convolution_model(x, y_, keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            batch = iterator.get_next()
            batch = sess.run(batch)     # Convert the tensor into a numpy array because that is what feed_dict accepts
            if i % 50 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print "Batches Completed: {} Accuracy: {}".format(i, train_accuracy)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


if __name__ == "__main__":
    train()
