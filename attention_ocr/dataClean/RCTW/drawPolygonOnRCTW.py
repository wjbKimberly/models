from PIL import Image, ImageDraw
import re
import time

errorGroundTruthPath="/home/wjb/OCR/datasets/icdar2017_RCTW_cut/errorMessage.txt"

def drawPolygonOnRCTW(imgPath,newImgPath,polygonxList,polygonyList):
    img = Image.open(imgPath)
    draw = ImageDraw.Draw(img)
    x=polygonxList
    y=polygonyList

    x0=min(x[0],x[3])
    y0=min(y[0],y[1])

    x1=max(x[1],x[2])
    y1=max(y[2],y[3])

    print "(%d,%d),(%d,%d)"%(x0,y0,x1,y1)
    draw.polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)], outline = (255, 0, 0))

    img.save(newImgPath)
    img.show()


def cutImageOnRCTW(imgPath,newImgPath,groundTruth,newGroundTruthPath,polygonxList,polygonyList):
    "https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region"

    #rectangle
    img = Image.open(imgPath)
    x = polygonxList
    y = polygonyList

    x0 = min(x[0], x[3])
    y0 = min(y[0], y[1])

    x1 = max(x[1], x[2])
    y1 = max(y[2], y[3])


    imgCut = img.crop((x0, y0, x1, y1))
    print x0,y0,x1,y1
    # imgCut.show()
    imgCut.save(newImgPath)

    #write groundTruth
    fw=open(newGroundTruthPath,"wb")
    fw.write(str(groundTruth))

    #polygon
    #get color of (0,0) as mask color
    # img = Image.open(imgPath).convert("RGBA")
    # maskColor = img.getpixel((0, 0))
    # # print maskColor
    # image = cv2.imread(imgPath, -1)
    # # mask defaulting to black for 3-channel and transparent for 4-channel
    # # (of course replace corners with yours)
    #
    # mask = np.zeros(image.shape, dtype=np.uint8)
    # for i in range(0,len(mask)):
    #     for j in range(0,len(mask[0])):
    #         mask[i][j]=np.array(maskColor[0:3])
    # roi_corners = np.array([[(x[0], y[0]), (x[1], y[1]), (x[2], y[2]),(x[3],y[3])]], dtype=np.int32)
    # # fill the ROI so it doesn't get wiped out when the mask is applied
    # channel_count = 4 # i.e. 3 or 4 depending on your image
    #
    # ignore_mask_color = maskColor
    # cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
    # # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    #
    # # apply the mask
    # masked_image = cv2.bitwise_and(image, mask)
    #
    # # save the result
    # cv2.imwrite(newImgPath, masked_image)



def cutImgaeFold(imgFoldPath,newImgFoldPath,indexStart,indexEnd):
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
                    cutImageOnRCTW(imgPath, newImgPathi, groundTruth, newGroundTruthPathi, polygonxList,polygonyList)
                except:
                    fe=open(errorGroundTruthPath,"a")
                    fe.write(str(imgPath)+"\t"+str(groundTruthLine)+"\n")
                    fe.flush()
                indexCurrent+=1
            else:
                break
        print "Img:",imgIndex,"Finished!"


if __name__ == '__main__':
    #time record
    ftime=open("/home/wjb/OCR/datasets/icdar2017_RCTW_cut/timerecord.txt","a")

    imgFoldPath="/home/wjb/OCR/datasets/icdar2017_RCTW"
    newImgFoldPath = "/home/wjb/OCR/datasets/icdar2017_RCTW_cut"
    indexStart=[0,1000,2000,3000,4425,5925]
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

