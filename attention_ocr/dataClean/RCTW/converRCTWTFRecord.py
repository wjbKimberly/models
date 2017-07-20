import tensorflow as tf
import os
from PIL import Image
import numpy
import json
import cv2
import re
import PIL

#parameters  
###############################################################################################  
#Input
#In summary, there are 6 datasets
datasetIndexRange = 6 
imgFoldPath = "/home/wangjianbo_i/models/attention_ocr/datasets/"
#train model tfrecords
charsetPath = "/home/wangjianbo_i/models/attention_ocr/python/datasets/data/fsns/charset_size=3507.txt"
output_path="/home/wangjianbo_i/models/attention_ocr/python/datasets/data/fsns/"
output_train_path=output_path+"train/train-00000-of-00512"
output_test_path=output_path+"test/test-00000-of-00512"
resize_width=150
resize_height=150 
#other parameters
num_of_views = 1	
length = 110
null_char_id=0
###############################################################################################  


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

"""
convert a unicode string into a list of character ids padded to a fixed length and unpadded as well. For example:
char_ids_padded, char_ids_unpadded = encode_utf8_string(
   text='abc',
   charset={'a':0, 'b':1, 'c':2},
   length=5,
   null_char_id=3)
the result should be:

char_ids_padded = [0,1,2,3,3]
char_ids_unpadded = [0,1,2]
"""
def encode_utf8_string(text, charset, length, null_char_id):
    char_ids_padded = []
    char_ids_unpadded = []
    for ti in text:
        mapNum=charset[ti]
        char_ids_padded.append(mapNum)
        char_ids_unpadded.append(mapNum)
    lenRemain=length-len(char_ids_unpadded)
    for i in range(0,lenRemain):
        char_ids_padded.append(null_char_id)
    return char_ids_padded, char_ids_unpadded

def extract_image(filename,  resize_height, resize_width):  
    image = cv2.imread(filename)  
    image = cv2.resize(image, (resize_height, resize_width))  
    b,g,r = cv2.split(image)         
    rgb_image = cv2.merge([r,g,b])       
    return rgb_image  

def write_examples(image_data, output_path,num_of_views,charset,length,null_char_id):
    """
    Create a tfrecord file.
  
    Args:
      image_data (List[(image_file_path (str), label (int), instance_id (str)]): the data to store in the tfrecord file. 
        The `image_file_path` should be the full path to the image, accessible by the machine that will be running the 
        TensorFlow network. The `label` should be an integer in the range [0, number_of_classes). `instance_id` should be 
        some unique identifier for this example (such as a database identifier). 
      output_path (str): the full path name for the tfrecord file. 
      img, num_of_views: Assume you have an numpy ndarray img which has num_of_views images stored side-by-side 
      charset: map all characters to num unrepeat
      length: longest length of text in datasets
      null_char_id: if there is no character, use this id to fill in the blank value
    """
    writer = tf.python_io.TFRecordWriter(output_path)
    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
	encoded_image = tf.image.encode_png(image_placeholder)
	with tf.Session('') as sess:
	    for imgPathi,text in image_data:
		img = extract_image(imgPathi, resize_height, resize_width)
		png_string = sess.run(encoded_image, feed_dict={image_placeholder: img})
		#encoded_image = tf.image.encode_jpeg(img)
		char_ids_padded, char_ids_unpadded = encode_utf8_string(text, charset, length, null_char_id)
		example = tf.train.Example(features=tf.train.Features(
		    feature={
			'image/format': _bytes_feature("PNG"),
			'image/encoded': _bytes_feature(png_string), 
			'image/class': _int64_feature(char_ids_padded),
			'image/unpadded_class': _int64_feature(char_ids_unpadded),
			'height': _int64_feature([resize_height]),
			'width': _int64_feature([resize_width]),
			'orig_width': _int64_feature([resize_width / num_of_views]),
			'image/text': _bytes_feature(text.encode('utf-8'))
		    }
		))
		writer.write(example.SerializeToString())
    writer.close()

def getimage_data(imgFoldPath):
    image_datai=[]
    #gain files' names in imgFoldPath
    files = os.listdir(imgFoldPath)
    files.sort()
    index=0
    fw=open("error.txt","a")
    fw.write(str(files))
    fw.flush()
    while index<len(files):
	imgPathi=imgFoldPath+"/"+files[index]
        imgLabelPath=imgFoldPath+"/"+files[index+1]
        index+=2
        # if not os.path.isdir(imgPathi):
        #     print imgPathi
        # if not os.path.isdir(imgLabelPath):
        #     print imgLabelPath
	labeli=open(imgLabelPath).read().decode("utf-8")
        # print labeli
        image_datai.append((imgPathi,labeli))
    return image_datai

def loadCharset(charsetPath):
    charset = {}
    fo = open(charsetPath, "r")
    patt= re.compile('(.+?)\t(.+?)')
    while True:
        # read one by one
        linei = fo.readline().decode('utf-8')
        if linei:
            linei = linei.replace("\n", "")
            linei=re.findall(patt,linei)[0]
	    charset[linei[1]]=int(linei[0])
        else:
            break
    return charset

if __name__=='__main__':
    #create image_data
    image_train_data = []
    image_test_data=[]
    #charsetstr=str(charsetReader.read())
    #charsetstr=charsetstr.replace("\n", "")
    #charset=json.loads(charsetstr)
    charset=loadCharset(charsetPath)
    #for parti in range(2,datasetIndexRange+1):
    for parti in range(1,2):
        imgFoldPathi=imgFoldPath+"/part%d"%parti
        # convert image to ndarray
        # combine ndarrays and labels to tuples
        image_datai=getimage_data(imgFoldPathi)
        image_train_data.extend(image_datai)
        print "train part %d load Finished!"%parti
    write_examples(image_train_data, output_train_path, num_of_views,charset, length, null_char_id)
    print "train write Finished!"
    #test model tfrecords
    for parti in range(1,2):
        imgFoldPathi=imgFoldPath+"/part%d"%parti
        # convert image to ndarray
        # combine ndarrays and labels to tuples
        image_datai=getimage_data(imgFoldPathi)
        image_test_data.extend(image_datai)
        print "test part %d load Finished!"%parti
    write_examples(image_test_data, output_test_path, num_of_views,charset, length, null_char_id)
    print "test write Finished!"
