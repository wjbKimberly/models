from PIL import Image
import numpy


# pic = Image.open("foo.jpg")
# pix = numpy.array(pic)

i = Image.open('image_0_0.jpg')
a = numpy.asarray(i) # a is readonly
print a.shape[0]
i = Image.fromarray(a)
i.show()

image_data=[(1,2),(2,3)]
for img, text in image_data:
    print img,text



