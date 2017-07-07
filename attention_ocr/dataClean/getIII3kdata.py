
import scipy.io as sio

#training from IIIT5k dataset
def getIIIT5kImageNameAndGroundTruth(matPath,imgFoldPath,dataClass,lexiconPath=None):

    data = sio.loadmat(matPath)
    imagePathList=[]
    labelList=[]
    lexiconList=[]
    # print data
    traindata=data[dataClass][0]
    for imgi in traindata:
        # print imgi['ImgName'][0],imgi['GroundTruth'][0]
        imagePathList.append(str(imgFoldPath)+"/"+str(imgi['ImgName'][0]))
        labelList.append(imgi['GroundTruth'][0])
    #append lexoconList
    if lexiconPath==None:
        lexiconList=None
    else:
        fo = open(lexiconPath, "r")
        while True:
            # read one by one
            lexi = str(fo.readline())
            if lexi:
                # the latest "\n"
                lexi = lexi.replace("\n", "")
                lexiconList.append(lexi)
            else:
                break
    return imagePathList,labelList,lexiconList




