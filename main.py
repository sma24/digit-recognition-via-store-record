
#!usr/bin/python
# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
from flask import Flask,request
from PIL import Image
from io import BytesIO
import datetime
import logging
from cassandra.cluster import Cluster
from werkzeug.utils import secure_filename
from cassandra.query import SimpleStatement

#connect to cassandra
log = logging.getLogger()

log.setLevel('INFO')

handler = logging.StreamHandler()

handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

log.addHandler(handler)

KEYSPACE = "mykeyspace"
session = 0

def createKeySpace():
    cluster = Cluster(contact_points=['cassandra'], port=9042)

    global session

    session = cluster.connect()

    log.info("Creating keyspace...")

    try:

        session.execute("""

            CREATE KEYSPACE %s

            WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }

            """ % KEYSPACE)

        log.info("setting keyspace...")

        session.set_keyspace(KEYSPACE)

        log.info("creating table...")

        session.execute("""

              CREATE TABLE record (
                   id int,
                   imgname text,
                   img text,
                   time text,
                   result text,

                   PRIMARY KEY (id)

               )

               """)

    except Exception as e:

        log.error("Unable to create keyspace")

        log.error(e)

createKeySpace();

app = Flask(__name__)
#build num-recognition model
def Number_recognition(file):
    #convertion
    im = Image.open(BytesIO(file))
    imout=im.convert('L')
    xsize, ysize=im.size
    if xsize != 28 or ysize!=28:
        imout=imout.resize((28,28),Image.ANTIALIAS)
    arr = []
    for i in range(28):
        for j in range(28):
            pixel = float(1.0 - float(imout.getpixel((j, i)))/255.0)
            arr.append(pixel)
    #keep_prob=tf.get_default_graph().get_tensor_by_name('dropout/Placeholder:0')
    #x=tf.get_default_graph().get_tensor_by_name('x:0')
    #y=tf.get_default_graph().get_tensor_by_name('fc2/add:0')
    #arr1 = np.array(arr).reshape((1,28*28))
    #pre_vec=sess.run(y,feed_dict={x:arr1,keep_prob:1.0})

    # build convolutional model
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x = tf.placeholder(tf.float32, [None, 784], name='x')
    keep_prob = tf.placeholder(tf.float32)
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # full connection
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout function
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    init_op = tf.initialize_all_variables()
    #begin to test number-image
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "./mnist/data_/convolutional.ckpt")
        arr1 = np.array(arr).reshape((1, 28 * 28))
        pre_vec = sess.run(y, feed_dict={x: arr1, keep_prob: 1.0})
        pre = str(np.argmax(pre_vec[0], 0)) + '\n'
    return pre
result = -1
imgname=""
idnum=1
@app.route('/upload', methods=['POST'])
def upload():
    global  result
    global imgname
    global idnum
    f = request.files['file']
    img = f.read()
    result = Number_recognition(img)
    imgname = str(secure_filename(f.filename))
    # store to cassandra
    global session
    session.execute("""INSERT INTO """+KEYSPACE+"""."""+"""record (id)
        VALUES (""" + str(idnum) + """);""")
    return ''' 
    <!doctype html> 
    <html> 
    <body> '''+str(result)+'''
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'> 
    <input type='submit' value='Upload'> 
    </form> 
    '''


@app.route('/')
def index():
    return ''' 
    <!doctype html> 
    <html> 
    <body> 
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'> 
    <input type='submit' value='Upload'> 
    </form> 
    '''

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)



