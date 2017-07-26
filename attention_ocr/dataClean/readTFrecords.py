import tensorflow as tf
import skimage.io as io

IMAGE_HEIGHT = 150
IMAGE_WIDTH = 600

tfrecords_filename = '/home/wangjianbo_i/models/attention_ocr/python/datasets/data/oldfsns/train/train-00000-of-00512'


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/encoded': tf.FixedLenFeature([], tf.string)
                                       })

    img = tf.decode_raw(features['image/encoded'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

    return img 

if __name__=='__main__':
	img = read_and_decode(tfrecords_filename)
	img_batch = tf.train.shuffle_batch([img],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
	    sess.run(init)
	    threads = tf.train.start_queue_runners(sess=sess)
	    for i in range(3):
		val= sess.run([img_batch])
		#l = to_categorical(l, 12) 
		print(val.shape)
