from PIL import Image, ImageDraw
import re
import time

import cv2
import numpy as np

errorGroundTruthPath="/home/wangjianbo_i/models/attention_ocr/resizedatasets/errorMessage.txt"

def cutImageOnRCTW(imgPath,newImgPath,polygonxList,polygonyList):
    #"https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region"
    #rectangle
    img = Image.open(imgPath)
    x = polygonxList
    y = polygonyList

    x0 = min(x[0], x[3])
    y0 = min(y[0], y[1])

    x1 = max(x[1], x[2])
    y1 = max(y[2], y[3])

    imgCut = img.crop((x0, y0, x1, y1))
    imgCut.save(newImgPath)
    print x0,y0,x1,y1,"   ",newGroundTruthPath

    #resize
    size=(150,150)
    
    from skimage import transform, data
    import matplotlib.pyplot as plt
    img = data.load(newImgPath)
    dst = transform.resize(img, size)
    from scipy.misc import imsave
    imsave(newImgPath, dst)
    

def cutImgaeFold(imgFoldPath,newImgFoldPath,indexStart,indexEnd):
    
    fe=open(errorGroundTruthPath,"w")
    for imgIndex in range(indexStart, indexEnd + 1):
        imgPath = imgFoldPath + "/image_%d.jpg" % imgIndex
        groundTruthPathi=imgFoldPath + "/image_%d.txt" % imgIndex
        fr=open(groundTruthPathi,"r")
        indexCurrent=0
        while True:
            newImgPathi = newImgFoldPath + "/image_%d_%d.jpg" %(imgIndex,indexCurrent)
            newGroundTruthPathi = newImgFoldPath + "/image_%d_%d.txt" %(imgIndex,indexCurrent)
            # read one by one
            linei = fr.readline().decode("utf-8")
            linei = linei.replace("\n", "")
            if linei!="":
                #check if the line is full of "#"
                flag=True
                for wi in linei:
                    if wi!="#":
                        flag=False
                        break
                if flag:
                    continue
                #get text
                groundTruthLine = linei.encode("utf-8")
                print indexCurrent,":",groundTruthLine
                pattText = re.compile('.+?"(.+?)"')
                groundTruth = re.findall(pattText, groundTruthLine)[0]

                #get coordinate
                pattCoordinate=re.compile('(.+?),')
                coordinateListi=re.findall(pattCoordinate,groundTruthLine)
                polygonxList=[]
                polygonyList=[]
		for xi in [0,2,4,6]:
                    polygonxList.append(float(coordinateListi[xi]))
                for yi in [1,3,5,7]:
                    polygonyList.append(float(coordinateListi[yi]))

                try:
                    cutImageOnRCTW(imgPath, newImgPathi, newGroundTruthPathi, polygonxList,polygonyList)
                
		    #write groundTruth
		    fw=open(newGroundTruthPath,"wb")
	    	    fw.write(str(groundTruth))
		except:
                    fe.write(str(imgPath)+"\t"+str(groundTruthLine)+"\n")
                    fe.flush()
		
                indexCurrent+=1
            else:
                break
        print "Img:",imgIndex,"Finished!"


if __name__ == '__main__':
    #time record
    ftime=open("/home/wangjianbo_i/models/attention_ocr/resizedatasets/timerecord.txt","w")

    imgFoldPath="/nfs/project/icdar2017_RCTW"
    newImgFoldPath = "/home/wangjianbo_i/models/attention_ocr/resizedatasets"
    indexStart=[999,1999,2000,4424,5924,5925]
    indexEnd=[999,1999,2999,4424,5924,8033]


    for i in range(0,len(indexStart)):
        localtime1 = "Cut%d begin:"%(i+1)+time.asctime(time.localtime(time.time()))+"\n"
        ftime.write(localtime1)
        ftime.flush()

        imgFoldPathi=imgFoldPath+"/part%d"%(i+1)
        newImgFoldPathi = newImgFoldPath + "/part%d" % (i + 1)
        cutImgaeFold(imgFoldPathi, newImgFoldPathi, indexStart[i], indexEnd[i])

        localtime2 = "Cut%d finished!"%(i+1)+time.asctime(time.localtime(time.time()))+"\n"
        ftime.write(localtime2)
        ftime.flush()

