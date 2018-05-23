import _pickle
import numpy as np
import os
import time
import cv2
import sys
from random import randint
sys.path.append('/home/ubuntu/DSP_DATA')
from PythonFunctions.DSSResources import DSSResources
from PythonFunctions.asset import Asset
from PythonFunctions.stream import PhotoStream
from PythonFunctions.FaceFeature import FaceFeature
from PythonFunctions.Face import Face

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

dssr = DSSResources('dev')
i=0
angles=[]
facelist=[]
imglist=[]
ff=FaceFeature(dssr=dssr)
stream= PhotoStream(dssr, uuid='3b9c0645d0fc40e789ff1c907fe82e9d')
assets=stream.asset_map(results='uuid')
for uuid in assets:
    asset=Asset(dssr,uuid=uuid)
    fids = asset.get_faceid()
    for faceid in fids:
        curation= dssr.dynamodb.read_face_attr(faceid,'curation')
        if curation is not None and 'orientationVector' in curation.keys():
            location = dssr.dynamodb.read_face_attr(faceid,'locations')
            faceimg=ff.get_faceimg(asset.image_data,location)
            cv2.imwrite('/home/ubuntu/zk/orientation/faceimg/'+faceid+'.jpg', cv2.cvtColor(faceimg, cv2.COLOR_RGB2BGR))
#             faceimg=cv2.resize(faceimg,(224,224))
#             imglist.append(faceimg)
            bottom=curation['orientationVector']['bottom']
            top=curation['orientationVector']['top']
            angle = np.rad2deg(np.arctan2(bottom['y']-top['y'], bottom['x']-top['x']))
            angles.append(angle)
            facelist.append(faceid)
            rotate=randint(-45,45)
            rotateimg=rotate_bound(faceimg,rotate)
            cv2.imwrite('/home/ubuntu/zk/orientation/faceimg/' + faceid + 'rt.jpg',
                        cv2.cvtColor(rotateimg, cv2.COLOR_RGB2BGR))
            angle2=angle+rotate
            if angle2<0:
                angle2+=360
            facelist.append(faceid+'rt')
            angles.append(angle2)
            i+=1
            if i%20==0:
                print('processed ', i, ' faces')
    if i>3500:
        break


with open ('/home/ubuntu/zk/orientation/facelist.pkl','wb') as pk:
    _pickle.dump(facelist, pk)
with open ('/home/ubuntu/zk/orientation/angles.pkl','wb') as pk:
    _pickle.dump(angles, pk)