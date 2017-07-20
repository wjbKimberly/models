#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wjbKimberly on 17-5-4

import re
import json

def extractWord(linei,previousDic,previousList):
    # print linei
    for wi in linei:
        # print wi
        if previousDic.has_key(wi)==True:
            previousDic[wi] +=1
            continue
        else:
            previousDic[wi]=1
            previousList.append(wi)


#format RCTW dataset
def getICDAR2017RCTWlexiconList(imgFoldPath,previousDic,previousList,indexStart,indexEnd,longestLength):
    #imgFoldPath -->> the pathe of images in one subset
    #imgIndex    -->> the index of current img
    #previousDic -->> previous dictionary
    #previousList-->> previous list of characters
    #[indexStart,indexEnd]  -->> index range of images in one fold

    for imgIndex in range(indexStart,indexEnd+1):
        imgPathi=imgFoldPath+"/image_%d.txt"%imgIndex
        fo = open(imgPathi, "r")
        patt= re.compile('.+?"(.+?)"')
        while True:
            # read one by one
            linei = fo.readline().decode("utf-8")
            if linei:
                # the latest "\n"
                linei = linei.replace("\n", "")
                linei=re.findall(patt,linei)[0]
                #if the text is full of "#",break it
                lineitmp=linei.replace("#","")
                if lineitmp=="":
                    continue
                #Get the longest length of text
                if longestLength<len(linei):
                    longestLength=len(linei)
                    print linei,"\t",longestLength,imgPathi
                linei=lineitmp
                # print linei
                extractWord(linei,previousDic,previousList)
            else:
                break
        # print "Img:",imgIndex,"Finished!"
    print longestLength


if __name__ == '__main__':
    #extract lexiconlist
    #scan datasets and get the longest text so we get the "length" and the "charset"
    longestLength=0
    imgFoldPath="/home/wangjianbo_i/models/attention_ocr/datasets"
    previousDic={}
    previousList=[]
    indexStart=[0,1000,2000,3000,4425,5925]
    indexEnd=[999,1999,2999,4424,5924,8033]
    #train model
    for i in range(0,len(indexStart)):
        imgFoldPathi=imgFoldPath+"/part%d"%(i+1)
        getICDAR2017RCTWlexiconList(imgFoldPathi, previousDic, previousList, indexStart[i], indexEnd[i],longestLength)
    print len(previousList)
    print longestLength
    previousList.sort(key=lambda element: (len(element), element))
    # sorted(previousList)
    #wirite lexiolist to file
    flexioList=open("/home/wangjianbo_i/models/attention_ocr/python/datasets/data/rctw/RCTWlexiolist.txt","a")
    for wi in previousList:
        # print wi
        flexioList.write(wi.encode("utf-8")+"\t"+str(previousDic[wi]).encode("utf-8")+"\n")
        flexioList.flush()
    # print previousList
    # write charset
    fcharset=open("/home/wangjianbo_i/models/attention_ocr/python/datasets/data/rctw/RCTWcharset.txt","a")
    index=0
    for wi in previousList:
        fcharset.write(wi.encode("utf-8") + "\t" + str(previousDic[wi]).encode("utf-8") + "\n")
        fcharset.flush()
        index+=1
    # print charset
    # write longestLength

