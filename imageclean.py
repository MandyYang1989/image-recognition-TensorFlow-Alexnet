#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
import os
import numpy as np

from PIL import Image
from sklearn import preprocessing
    
'''把labels转换为one-hot形式'''
def dense_to_one_hot(labels_dense, num_classes=25):
    print "Convert class labels from scalars to one-hot vectors."
    label_size = labels_dense.shape[0]
    
    enc = preprocessing.OneHotEncoder(sparse= True, n_values = num_classes)

    enc.fit(labels_dense)

    array = enc.transform(labels_dense).toarray()

    return array

'''NumPy的数组没有这种动态改变大小的功能，numpy.append()函数每次都会重新分配整个数组，并把原来的数组复制到新数组中
append效率太低。一次性把 数组 大小建好，再改改里面的数据即可，最后一步截取有效文件个数前size数据'''
def getImageMatrix(input_path='/home/s-20/Image/train_data/'):
    
    #获取目录下文件分类
    folderList = os.listdir(input_path)
      
    #所有文件的个数
    size = len(sum([i[2] for i in os.walk(input_path)],[]))
    #存放data
    image_list = np.empty((size, 224, 224, 3), np.float)
    #print "=====初始化image_list======",image_list.shape[0],image_list.shape
        
    #存放labels
    labels_list = np.empty((0,25),np.float)
    
    #遍历子目录，把类别转成int型===========遍历文件方式太复杂！！！！！！！！！待改!!!!!!!!!!!!!!
    for label,folder_name in enumerate(folderList,start=0):
        files = os.path.join(input_path, folder_name)
        print "目录的名字：",files
        
        #每个分类下有效文件的个数
        file_size = 0
        
        #要替换的data的索引
        index = 0
        
        #遍历目录，获取每个图像,变成224*224*3的向量
        for parent,dirnames,file_list in os.walk(files):
            
            file_size = len(file_list)
            print "====目录总文件的个数：",file_size
            for file in file_list:
                #通过调用 array() 方法将图像转换成NumPy的数组对象
                image_path = os.path.join(parent,file)
                image = np.array(Image.open(image_path))
                
                #判断图片的维数是否相同，过滤黑白的图片
                if image.shape == (224,224,3):   
                    image_list[index] = image
                    index += 1
                    '''
                    #加入到集合中--------append效率太低
                    #image_list = np.append( image_list,image,axis = 0)
                    '''
                else:
                    print "!!!!!!!!!!!!!!!!!格式不对删除图片!!!!!!!!!!!!!!!!!!!!!!!!!!!!", image_path

        #获取label的one-hot矩阵
        labels = np.array([label] * file_size).reshape(-1,1)   
        #目录下格式合格的文件不为空
        if labels.size:
            labels_one_hot = dense_to_one_hot(labels)
            labels_list =  np.append(labels_list, labels_one_hot, axis = 0)
            
            
    print "总文件个数：",size
    print "label的个数",labels_list.shape[0]
    print "label的格式",labels_list.shape
    print "image的个数",image_list.shape[0]
    print "image的格式",image_list.shape
    
    return image_list, labels_list

if __name__=='__main__': 
    
    start_time = time.time()
    #准备数据
    getImageMatrix()
    end_time = time.time()
    print "准备数据的时间开销：",end_time - start_time
    
    #TODO 迁移数据，使用goolenet
    
    