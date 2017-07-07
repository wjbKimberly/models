import tensorflow as tf
import os
from PIL import Image
import numpy
import json
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
	#print charset[ti]
        char_ids_padded.append(mapNum)
        char_ids_unpadded.append(mapNum)
    lenRemain=length-len(char_ids_unpadded)
    for i in range(0,lenRemain):
        char_ids_padded.append(null_char_id)
    return char_ids_padded, char_ids_unpadded

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

    for imgPathi,text in image_data:
        img = numpy.array(Image.open(imgPathi))
	#print text.encode("utf-8")
        print imgPathi," Finished!"
	char_ids_padded, char_ids_unpadded = encode_utf8_string(text, charset, length, null_char_id)
        #print char_ids_padded,char_ids_unpadded
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image/format': _bytes_feature("JPG"),
                'image/encoded': _bytes_feature(img.tostring()),
                'image/class': _int64_feature(char_ids_padded),
                'image/unpadded_class': _int64_feature(char_ids_unpadded),
                'height': _int64_feature([img.shape[0]]),
                'width': _int64_feature([img.shape[1]]),
                'orig_width': _int64_feature([img.shape[1] / num_of_views]),
                'image/text': _bytes_feature(text.encode('utf8'))
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

if __name__=='__main__':
    #create image_data
    image_data = []
    #output_path="/home/wangjianbo_i/google_model/RCTWdata/icdar2017_RCTW_cut/train.tfrecords"
    output_path="train.tfrecords"
    num_of_views = 1	
    charsetReader = open("RCTWcharset.txt", "r")
    charsetstr=str(charsetReader.read())
    charsetstr=charsetstr.replace("\n", "")
    charset=json.loads(charsetstr)
    length = 110
    null_char_id=-1
    #In summary, there are 6 datasets
    datasetIndexRange = 6
    imgFoldPath = "/home/wangjianbo_i/google_model/RCTWdata/icdar2017_RCTW_cut/"
    for parti in range(1,datasetIndexRange+1):
        imgFoldPathi=imgFoldPath+"/part%d"%parti
        # convert image to ndarray
        # combine ndarrays and labels to tuples
        image_datai=getimage_data(imgFoldPathi)
        image_data.extend(image_datai)
        #fw = open("/home/wangjianbo_i/google_model/RCTWdata/icdar2017_RCTW_cut/train.txt", "a")
        #fw.write(str(image_datai)+"\n\n")
        #fw.flush()
        print "part %d Finished!\n"%parti
        # print image_data

    # print image_data
    write_examples(image_data, output_path, num_of_views,charset, length, null_char_id)
