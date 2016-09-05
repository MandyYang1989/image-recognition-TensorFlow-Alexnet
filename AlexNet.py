#!/usr/bin/env python
# coding=utf-8

import time
import os
import sys
import gc
import logging
import numpy as np

from PIL import Image
from sklearn import preprocessing
import tensorflow as tf


logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='/home/s-20/python-homework/tudou/project/homework/python-test/Image/log/alexnet.log',
                filemode='w')
logging.debug("This is debug message")

# TensorBoard可视化网络结构和参数
'''
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/AlexNet_logs', 'Summaries directory')
'''

'''把labels转换为one-hot形式'''


def dense_to_one_hot(labels_dense, num_classes=25):
    logging.debug( "Convert class labels from scalars to one-hot vectors.")
    label_size = labels_dense.shape[0]

    enc = preprocessing.OneHotEncoder(sparse=True, n_values=num_classes)

    enc.fit(labels_dense)

    array = enc.transform(labels_dense).toarray()

    return array


# 对原始数据洗牌，跑不动
def shuffle_1(*arrs):
    start = time.time()
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))

    logging.debug("==排序后的索引值====%s,%s"% (type(p), p))
    data_shape = arrs[0].shape
    logging.debug( "=====data_shape==========%s"% data_shape)
    new_data = np.empty(data_shape, np.float)
    logging.debug( "new_data占内存%s" % sys.getsizeof(new_data))
    new_label = np.empty(arrs[1].shape, np.float)
    # 最蠢的方法
    data = arrs[0]
    label = arrs[1]
    for i in range(len(data)):
        tmp_data = data[p[i]]
        new_data[i] = tmp_data

        tmp_label = label[p[i]]
        new_label[i] = tmp_label
    end = time.time()
    logging.debug( "shuffle花费的时间：%d" % ((end - start) / 1000))

    return new_data, new_label


# 对原始数据洗牌，跑不动
def shuffle_2(*arrs):
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


# 只获取洗牌后的索引
def shuffle_3(size):
    p = np.random.permutation(size)
    return p


'''NumPy的数组没有这种动态改变大小的功能，numpy.append()函数每次都会重新分配整个数组，并把原来的数组复制到新数组中
append效率太低。一次性把 数组 大小建好，再改改里面的数据即可，最后一步截取有效文件个数前size数据'''


def getImageMatrix(input_path='/home/s-20/Image/train_data/'):
    # 获取目录下文件分类
    folderList = os.listdir(input_path)

    # 所有文件的个数
    size = len(sum([i[2] for i in os.walk(input_path)], []))
    # 存放data
    image_list = np.empty((size, 227, 227, 3), np.float)
    # logging.debug( "=====初始化image_list======",image_list.shape[0],image_list.shape

    # 存放labels
    labels_list = np.empty((0, 25), np.float)

    # 遍历子目录，把类别转成int型===========遍历文件方式太复杂！！！！！！！！！待改!!!!!!!!!!!!!!
    for label, folder_name in enumerate(folderList, start=0):
        files = os.path.join(input_path, folder_name)
        logging.debug( "目录的名字：%s" %  files)

        # 每个分类下有效文件的个数
        file_size = 0

        # 要替换的data的索引
        index = 0

        # 遍历目录，获取每个图像,变成227*227*3的向量
        for parent, dirnames, file_list in os.walk(files):

            file_size = len(file_list)
            logging.debug( "====目录总文件的个数：%d" % file_size)
            for file in file_list:
                # 通过调用 array() 方法将图像转换成NumPy的数组对象
                image_path = os.path.join(parent, file)
                image = np.array(Image.open(image_path))

                # 判断图片的维数是否相同，过滤黑白的图片
                if image.shape == (227, 227, 3):
                    image_list[index] = image
                    index += 1
                    '''
                    #加入到集合中--------append效率太低
                    #image_list = np.append( image_list,image,axis = 0)
                    '''
                else:
                    logging.debug( "!!!!!!!!!!!!!!!!!格式不对删除图片!!!!!!!!!!!!!!!!!!!!!!!!!!!!%s" % image_path)

        # 获取label的one-hot矩阵
        labels = np.array([label] * file_size).reshape(-1, 1)
        # 目录下格式合格的文件不为空
        if labels.size:
            labels_one_hot = dense_to_one_hot(labels)
            labels_list = np.append(labels_list, labels_one_hot, axis=0)

    logging.debug( "总文件个数：%d" % size)
    logging.debug( "label的个数:%d" % labels_list.shape[0])
    #logging.debug( "label的格式：%s" % labels_list.shape)
    logging.debug( "image的个数:%d" % image_list.shape[0])
    #logging.debug( "image的格式:%s" % image_list.shape)
    logging.debug( "image的占数据大小:%d" % sys.getsizeof(image_list))

    '''直接对原始数据洗牌，内存错误
    # 洗牌
    image_list_new, labels_list_new = shuffle(image_list, labels_list)
    # 分70%做train，30%做验证集算准确率
    train_size = int(size * 0.7)
    train_data = image_list_new[:train_size]
    train_label = labels_list_new[:train_size]
    validation_data = image_list_new[train_size:]
    validation_label = labels_list_new[train_size:]
    '''
    # 获取洗牌后的索引
    index_shuffle = shuffle_3(len(labels_list))
    # 分70%做train，30%做验证集算准确率
    train_size = int(size * 0.7)
    train_index = index_shuffle[:train_size]
    validation_index = index_shuffle[train_size:]
    logging.debug( "训练的个数:%d" % len(train_index))
    logging.debug( "验证数据的个数:%d" % len(validation_index))
    return image_list, labels_list, train_index, validation_index


# 卷积层
def conv2d(name, input, w, b, stride, padding='SAME'):
    # 测试
    x = input.get_shape()[-1]

    x = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding=padding)

    x = tf.nn.bias_add(x, b)

    data_result = tf.nn.relu(x, name=name)
    # 输出参数
    # tf.histogram_summary(name + '/卷积层', data_result)
    gc.collect()
    return data_result


# 最大下采样
def max_pool(name, input, k, stride):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)


# 归一化操作 ToDo 正则方式待修改
def norm(name, input, size=4):
    return tf.nn.lrn(input, size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


# 定义整个网络 input：227*227*3
def alex_net(x, weights, biases, dropout):
    # 卷积层1  96个11*11*3 filter
    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'], stride=4)
    # 下采样层 kernel：3*3 步长：2
    pool1 = max_pool('pool1', conv1, k=3, stride=2)
    # 正则化 96 个27*27
    norm1 = norm('norm1', pool1, size=5)

    # 卷积层2 ToDo 两组filter， 每组256个（5*5*48） padding=2
    # conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'], stride=1, padding=2)
    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'], stride=1, padding="VALID")
    # 下采样
    pool2 = max_pool('pool2', conv2, k=3, stride=2)
    # 归一化
    norm2 = norm('norm2', pool2, size=5)

    # 卷积层3 padding=1
    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'], stride=1, padding="VALID")

    # 卷积层4 padding=1
    conv4 = conv2d('conv4', conv3, weights['wc4'], biases['bc4'], stride=1, padding="VALID")

    # 卷积层5 padding=1
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'], stride=1, padding="VALID")
    # 下采样 input:13*13*256 output:6*6*256
    pool5 = max_pool('pool5', conv5, k=3, stride=2)

    # 全连接层1
    # 先把特征图转为向量
    fc1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1, name='fc1')
    # Dropout
    drop1 = tf.nn.dropout(fc1, dropout)

    # 全连接层2
    fc2 = tf.add(tf.matmul(drop1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2, name='fc2')
    # Dropout
    drop2 = tf.nn.dropout(fc2, dropout)

    # out
    out = tf.add(tf.matmul(drop2, weights['out']), biases['out'])

    return out


if __name__ == '__main__':
    start_time = time.time()
    # 准备数据
    # train_data, train_label, validation_data, validation_label= getImageMatrix()
    image_list, labels_list, train_index, validation_index = getImageMatrix()
    end_time = time.time()
    logging.debug( "准备数据的时间开销：%d" % (end_time - start_time))

    # TODO 迁移数据，使用AlexNet

    # 训练参数
    learning_rate = 0.001
    training_iters =100
    batch_size = 64
    display_step = 10

    # network 参数
    n_input = [None, 227, 227, 3]
    n_classes = 25
    dropout = 0.5

    # tf graph input
    x = tf.placeholder(tf.float32, n_input, name='x')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')
    keep_prob = tf.placeholder(tf.float32)

    # 存储所有的参数
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
        'wd1': tf.Variable(tf.random_normal([4096, 4096])),
        'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        'out': tf.Variable(tf.random_normal([4096, 25]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([96])),
        'bc2': tf.Variable(tf.random_normal([256])),
        'bc3': tf.Variable(tf.random_normal([384])),
        'bc4': tf.Variable(tf.random_normal([384])),
        'bc5': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([4096])),
        'bd2': tf.Variable(tf.random_normal([4096])),
        'out': tf.Variable(tf.random_normal([25]))
    }

    # 预测值
    pred = alex_net(x, weights, biases, keep_prob)

    # 定义损失函数和学习步骤

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 初始化所有的共享变量
    init = tf.initialize_all_variables()

    '''以下的操作全是针对索引，而不是数据本身'''
    with tf.Session() as sess:
        sess.run(init)
        # 在traindata上做训练
        start_train_time = time.time()
        for i in range(training_iters):
            logging.debug( "第%d轮:" % i)
            logging.debug( "对数据的索引再洗牌，获取新的索引列表")
            index_tmp = shuffle_3(len(train_index))
            train_index = np.array(train_index)
            train_index_new = train_index[index_tmp]
            logging.debug( "新的索引列表：%s %d" % (train_index_new, len(train_index_new)))

            data_size = len(train_index)
            start = 0
            while start < data_size:
                logging.debug( "第%d批:" % start)
                batch_index = train_index_new[start: start + batch_size]
                logging.debug( " 截取后的索引列表：%s %d" % (batch_index, len(batch_index)))
                s_batch_time = time.time()
                batch_x = image_list[batch_index]
                batch_y = labels_list[batch_index]
                e_batch_time = time.time()
                logging.debug("截取数据消耗的时间：%d" % ((e_batch_time - s_batch_time) / 1000))
                # reshape数据

                # 喂数据
                s_batch_train_time = time.time()
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
                e_batch_train_time = time.time()
                logging.debug("训练一批数据消耗的时间：%d" % ((e_batch_train_time - s_batch_train_time) / 1000))

                start += batch_size
            #每轮释放一次内存
            gc.collect()
        end_train_time = time.time()
        logging.debug( "训练共耗时：%d" % ((end_train_time - start_train_time) / 1000))
        logging.debug( "Optimization Finished!")
        gc.collect()
        # 在testdata上做训练
        # 测试准确度
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        validation_data = image_list[validation_index]
        validation_label = labels_list[validation_index]
        logging.debug ("Accuarcy on Test-dataset: %s" %
              sess.run(accuracy, feed_dict={x: validation_data[0:100], y: validation_label[0:100], keep_prob: 1.}))

