#! python3
# -*- coding = utf-8 -*-

"""
对MINIST手写数字数据进行处理
"""

import os
import numpy as np
from PIL import Image


IMAGE_WIDTH  =28
IMAGE_HEIGHT =28
IMAGE_SIZE   =784
TRAIN_NUMBER =60000
TEST_NUMBER  =10000


OUTPUT_DIR = 'images/'


# 将第number幅图的字节流bytes转为图片image.png
def bytes_to_image(image_bytes, number):
    # 创建一张空白的图片，其中的’L’代表这张图片是灰度图
    image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT))
    index = IMAGE_SIZE * number
    # 把像素点填充到空白图中
    for y in range(IMAGE_HEIGHT):
        for x in range(IMAGE_WIDTH):
            image.putpixel((x, y), int(image_bytes[index]))
            index += 1
    # 如果文件夹不存在 就创建文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    # 保存图片
    image.save(OUTPUT_DIR+'%d.png'%number)


# 将第number幅图的字节流转为float格式
def image_bytes_to_float(image_bytes, number):
    # 创建零矩阵 存放图片像素信息
    image_float = np.zeros((number, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)
    index = 0
    for num in range(number):
        for x in range(IMAGE_WIDTH):
            for y in range(IMAGE_HEIGHT):
                image_float[num][x][y] = float(image_bytes[index])
                index += 1
    return image_float


# 将答案标签字节流转为float格式的矩阵
def table_bytes_to_float(table_bytes, number):
    table_float = np.zeros((number, 10), dtype=np.float32)
    for index in range(number):
        table_float[index][table_bytes[index]] = 1.0
    return table_float


# 读取训练文件和测试文件 字节流数据
def get_sample(TRAIN_IMAGE_FILE, TRAIN_TABLE_FILE, TEST_IMAGE_FILE, TEST_TABLE_FILE):
    # 读取字节流bytes
    def _get_bytes(path, start, number):
        file = open(path ,'rb')
        # 定位读取起点
        file.seek(start, 0)
        bytes = file.read(number)
        file.close()
        return(bytes)
    train_images = _get_bytes(TRAIN_IMAGE_FILE, 16, TRAIN_NUMBER*IMAGE_SIZE)
    train_tables = _get_bytes(TRAIN_TABLE_FILE, 8,  TRAIN_NUMBER)
    test_images  = _get_bytes(TEST_IMAGE_FILE,  16, TEST_NUMBER*IMAGE_SIZE)
    test_tables  = _get_bytes(TEST_TABLE_FILE,  8,  TEST_NUMBER)
    train_images = image_bytes_to_float(train_images, TRAIN_NUMBER)
    train_tables = table_bytes_to_float(train_tables, TRAIN_NUMBER)
    test_images  = image_bytes_to_float(test_images, TEST_NUMBER)
    test_tables  = table_bytes_to_float(test_tables, TEST_NUMBER)
    return train_images, train_tables, test_images, test_tables


# 获取从指定位置start开始的number个数据
def get_data(images, tables, start, number):
    images_   = np.zeros((number,IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)
    answers_  = np.zeros((number,10), dtype=np.float32)
    for x in range(number):
        images_[x, :, :]  = images[start+x, :, :]
        answers_[x, :]    = tables[start+x, :]
    return images_, answers_
