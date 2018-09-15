import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import tensorflow as tf
import numpy as np
from IPython import embed
import TensorflowUtils as utils
import readPotatoeDataset as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
import matplotlib.pyplot as plt
import time
import json
from IPython import embed
import ntpath
import random
import cv2
import glob

#0 = normal
#1 = clump
#2 = bad

#Segment only sees background and potatoe
NUM_OF_CLASSES_FCN = 2
FCN_IMAGE_SIZE = 512
CLASSIFIER_IMAGE_SIZE = 224

#We're only going to do batch size 1
class Dataset():
    imagesWithLabels = []
    imagesWithMasks = []
    def __init__(self, jsonFile):
        with open(jsonFile) as f:
            data = json.load(f)
            
        clumpCount = 0
        normalCount = 0
        badCount = 0
        imagesWithLabels = []

        normalsWithLabel = []
        clumpsWithLabel = []
        badsWithLabel = []

        #For every image that has a classification
        for key in data.keys():
            #key = unicode('raw_images_2018.Aug.14/' +  ntpath.basename(maskPath), "utf-8")

            labelList = data[key]

            try:
                a = len(data[key])
            except:
                continue
            if len(labelList) == 0:
                continue
            label = 2
            for l in labelList:
                if l  == "Normal":
                    label = 0
                    break
                if l == "Clump":
                    label = 1
                    break
    
            temp = {}
            temp['imageFilename'] = 'dataset/' + key
            #temp['maskFilename'] = os.path.join("dataset/masks_2018.Aug.14", ntpath.basename(key))
            temp['label'] = label
            temp['actual label'] = str(data[key])

            if label == 0:
                    normalCount += 1
                    normalsWithLabel.append(temp)
            elif label == 1:
                    clumpCount += 1
                    clumpsWithLabel.append(temp)
            else:
                    badCount += 1
                    badsWithLabel.append(temp)

        #Resample to deal with the inbalance in the dataset (fewer clumps then bads or normals)
        maxLength = max(max(len(clumpsWithLabel), len(badsWithLabel)), len(normalsWithLabel))
        for i in range(20):
            clumpsWithLabel += list(clumpsWithLabel)
        for i in range(2):
            badsWithLabel += list(badsWithLabel)
        for i in range(2):
            normalsWithLabel += list(normalsWithLabel)
  
        clumpsWithLabel = clumpsWithLabel[0:maxLength]
        badsWithLabel = badsWithLabel[0:maxLength]
        normalsWithLabel = normalsWithLabel[0:maxLength]
        self.imagesWithLabels  = clumpsWithLabel + badsWithLabel + normalsWithLabel
       
        #For each image that has a mask
        for maskPath in glob.glob('dataset/masks_2018.Aug.14/*'):
            imgPath = maskPath.replace('masks_2018.Aug.14', 'raw_images_2018.Aug.14')
            temp = {}
            temp['imageFilename'] = imgPath
            temp['maskFilename'] = maskPath

            self.imagesWithMasks.append(temp)
            
    #For the FCN
    def getRandomImageAndMask(self):
        entry = self.imagesWithMasks[random.randint(0, len(self.imagesWithMasks) - 1)]
        img = cv2.imread(entry['imageFilename'], 1)
        mask = cv2.imread(entry['maskFilename'], 0)
        mask[mask >= 1] = 1

        img = cv2.resize(img,(FCN_IMAGE_SIZE,FCN_IMAGE_SIZE))
        mask = cv2.resize(mask,(FCN_IMAGE_SIZE,FCN_IMAGE_SIZE))

        img = img[None, :, :, :]
        mask = mask[None, :, :, None]

        return img, mask

    #For the classifier
    def getRandomImageAndLabel(self):
        entry = self.imagesWithLabels[random.randint(0, len(self.imagesWithLabels) - 1)]
        img = cv2.imread(entry['imageFilename'], 1)
        img = cv2.resize(img,(FCN_IMAGE_SIZE,FCN_IMAGE_SIZE))
        img = img[None, :, :, :]

        #label = entry['label']
        labelScalar = entry['label']

        if labelScalar == 0:
            label = np.array([1, 0, 0])[None, :]
        elif labelScalar == 1:
            label = np.array([0, 1,0])[None, :]
        elif labelScalar == 2:
            label = np.array([0, 0, 1])[None, :]
        else:
            print "Invalid label found! -- Stopping"
            exit()
        return img, label, entry['actual label']

def flipImage(img):
  #Get random int between in interval (-1,2)
  flipType = random.randint(-1,3)
  #1 in 4 chance we don't flip
  if flipType == 2:
    return img
  return cv2.flip(img, flipType)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("weights", shape,initializer=initializer, dtype=tf.float32)

def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def FCN(image, keep_prob):
    with tf.variable_scope("FCN"):

        #conv1
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([3, 3, 3, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

        # conv2
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([3, 3, 32, 128])
            b_conv2 = bias_variable([128])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(max_pool_2x2(h_conv2))

        # conv3
        with tf.variable_scope('conv3'):
            W_conv3 = weight_variable([3, 3, 128, 256])
            b_conv3 = bias_variable([256])
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)

        # conv4
        with tf.variable_scope('conv4'):
            W_conv4 = weight_variable([3, 3, 256, 512])
            b_conv4 = bias_variable([512])
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4)

        #Upscale
        deconv_shape1 = h_pool3.get_shape()

        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, 512], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(h_pool4, W_t1, b_t1, output_shape=tf.shape(h_pool3))
        fuse_1 = tf.add(conv_t1, h_pool3, name="fuse_1")

        deconv_shape2 = h_pool2.get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(h_pool2))
        fuse_2 = tf.add(conv_t2, h_pool2, name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSES_FCN])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSES_FCN, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSES_FCN], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3

class Classifier(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.y_conv = 0
        self.h_conv2 = 0
        self.h_conv1 = 0
        self.h_conv5 = 0
        self.W_fc1 = 0
        self.b_fc1 = 0
        self.h_pool5 = 0

        # conv1
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([7, 7, 3, 8])
            b_conv1 = bias_variable([8])
            h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)


        # conv2
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 8, 16])
            b_conv2 = bias_variable([16])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

        # conv3
        with tf.variable_scope('conv3'):
            W_conv3 = weight_variable([5, 5, 16, 32])
            b_conv3 = bias_variable([32])
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)

        # conv4
        with tf.variable_scope('conv4'):
            W_conv4 = weight_variable([5, 5, 32, 64])
            b_conv4 = bias_variable([64])
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4)

        # fc1
        with tf.variable_scope("fc1"):
            shape = int(np.prod(h_pool4.get_shape()[1:]))
            W_fc1 = weight_variable([shape, 1024])
            b_fc1 = bias_variable([1024])
            h_pool4_flat = tf.reshape(h_pool4, [-1, shape])
            h_fc1 = tf.matmul(h_pool4_flat, W_fc1) + b_fc1

        # dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # fc2
        with tf.variable_scope("fc2"):
            W_fc2 = weight_variable([1024, 3])
            b_fc2 = bias_variable([3])
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv))
        self.softmax = tf.nn.softmax(y_conv, axis=None, name=None,  dim=None)
        self.pred = tf.argmax(y_conv, 1)

        self.correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop

#Train op for FCN
def train(loss_val, var_list):
    fcn_learning_rate = 0.0001
    optimizer = tf.train.AdamOptimizer(fcn_learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)


def main():
    trainFCN = False
    loadFCNModel = True

    trainClassifier = False
    loadClassifierModel = True

    fcn_keep_probabilty = 0.85
    classifier_keep_probabilty = 0.85

    trainDataset = Dataset(jsonFile = 'dataset/train.json')
    testDataset = Dataset(jsonFile = 'dataset/test.json')
    
    ###################### FCN Part ##############################
    keep_probability_FCN = tf.placeholder(tf.float32, name="keep_probabilty")
    image_FCN = tf.placeholder(tf.float32, shape=[None, FCN_IMAGE_SIZE, FCN_IMAGE_SIZE, 3], name="input_image")
    annotation_FCN  = tf.placeholder(tf.int32, shape=[None, FCN_IMAGE_SIZE, FCN_IMAGE_SIZE, 1], name="annotation")

    pred_annotation_FCN , logits_FCN  = FCN(image_FCN , keep_probability_FCN )
    
    loss_FCN  = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits =logits_FCN ,
                                                                          labels =tf.squeeze(annotation_FCN , squeeze_dims=[3]),
                                                                          name="entropy")))
    trainable_var_FCN  = tf.trainable_variables()
    train_op_FCN  = train(loss_FCN , trainable_var_FCN )

    sess_FCN  = tf.Session()

    print "Setting up Saver"
    saver_FCN = tf.train.Saver()

    sess_FCN.run(tf.global_variables_initializer())
    ckpt_FCN  = tf.train.get_checkpoint_state("logs/")

    if loadFCNModel:
        saver_FCN.restore(sess_FCN , ckpt_FCN .model_checkpoint_path)
        print ("Loaded Model:", ckpt_FCN .model_checkpoint_path)

    if trainFCN:
        print "Training FCN"
        train_losses = []
        for trainStep in range(1000):
            train_images, train_annotations = trainDataset.getRandomImageAndMask()
            feed_dict = {image_FCN : train_images, annotation_FCN : train_annotations, keep_probability_FCN : fcn_keep_probabilty}

            _, train_loss = sess_FCN.run([train_op_FCN , loss_FCN ], feed_dict=feed_dict)

            train_losses.append(train_loss)

            #How often to save a checkpoint
            if trainStep % 100 == 0:
                test_images, test_annotations = testDataset.getRandomImageAndMask()
                test_loss = sess_FCN.run(loss_FCN, feed_dict={image_FCN : test_images, annotation_FCN : test_annotations,
                                                       keep_probability_FCN : 1.0})
                print "test_loss (random batch):", test_loss, "train_loss:", np.mean(train_losses)
                saver_FCN .save(sess_FCN , "logs/" + "FCNModel.ckpt", trainStep)
                train_losses = []

    ###################### Classifier Part ##############################

    keep_probability_classifier = tf.placeholder(tf.float32, name="keep_probabilty")

    num_class = 3
    image_classifier = tf.placeholder(tf.float32, shape=[None, CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE, 3], name="input_image")
    #labels = tf.placeholder(tf.float32, shape=[None, num_class], name="label")
    
    labels_classifier = tf.placeholder(tf.float32, (None, num_class))

    classifier = Classifier(image_classifier, labels_classifier)

    optimizer = tf.train.AdamOptimizer(0.0001).minimize(classifier.loss)
    print "Setting up Saver for classifier"

    trainable_var_classifier  = tf.trainable_variables()

    saver_classifier = tf.train.Saver(tf.trainable_variables())

    sess_classifier = tf.Session()

    sess_classifier.run(tf.global_variables_initializer())
    ckpt_classifier = tf.train.get_checkpoint_state("classifierLogs/")

    if loadClassifierModel:
        saver_classifier.restore(sess_classifier, ckpt_classifier.model_checkpoint_path)
        print ("Loaded Model:", ckpt_classifier.model_checkpoint_path)

    #Get a dummy annotation for the FCN
    _, dummy_annotation = trainDataset.getRandomImageAndMask()

    train_losses = []
    total_accuracy = []
    for trainStep in range(1000000):
        if trainClassifier:
            train_images, train_labels, actualLabel = trainDataset.getRandomImageAndLabel()

            #Segment and crop image
            #Get segmentation from FCN
            segmentation = sess_FCN.run(pred_annotation_FCN, feed_dict={image_FCN : train_images, annotation_FCN : dummy_annotation, keep_probability_FCN : 1.0})
            segmentation = np.squeeze(segmentation) #Get rid of batch and channel dimensions
            segmentation = segmentation.astype(np.uint8)
            orignalImage = np.squeeze(train_images)

            #Get largest connected component
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(segmentation, connectivity=4)
            sizes = stats[:, -1]
            max_label = 1
            max_size = sizes[1]
            for i in range(2, nb_components):
                if sizes[i] > max_size:
                    max_label = i
                    max_size = sizes[i]

            onlyLargestObject = np.zeros(output.shape)
            onlyLargestObject[output == max_label] = 255
            segmentation = onlyLargestObject.astype(np.uint8)

            _, contours, _= cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(contours[0])

            croppedImage = crop_minAreaRect(orignalImage, rect)
            
            #Rotate all cropped potatoes to be landscape
            h, w = croppedImage.shape[:2]
            if h > w:
                #Swap height and width dimensions
                croppedImage = np.transpose(croppedImage, (1, 0, 2))

            plt.title(str(np.squeeze(train_labels)))
            plt.imshow(croppedImage[...,::-1])
            #plt.show()
            labelString = str(np.squeeze(train_labels))
            if labelString == "[1 0 0]":
                plt.savefig("normals/" + actualLabel + "-" + str(trainStep))
            elif labelString == "[0 1 0]":
                plt.savefig("clumps/" + actualLabel + "-" +str(trainStep))
            elif labelString == "[0 0 1]":
                plt.savefig("bads/" + actualLabel + "-" +str(trainStep))

            croppedImage = np.array(cv2.resize(croppedImage,(CLASSIFIER_IMAGE_SIZE,CLASSIFIER_IMAGE_SIZE)))
            train_images = croppedImage[None, :, :] #Add batch dimension back in

            feed_dict = {image_classifier: train_images, 
                             labels_classifier: train_labels,
                             classifier.keep_prob: classifier_keep_probabilty}
            _, train_loss = sess_classifier.run([optimizer, classifier.loss], feed_dict=feed_dict)

            train_losses.append(train_loss)

        #How often to save a checkpoint
        if trainStep % 100 == 0:
            test_images, test_labels, actualLabel = testDataset.getRandomImageAndLabel()

            #Segment and crop image
            #Get segmentation from FCN
            segmentation = sess_FCN.run(pred_annotation_FCN, feed_dict={image_FCN : test_images, annotation_FCN : dummy_annotation, keep_probability_FCN : 1.0})
            segmentation = np.squeeze(segmentation) #Get rid of batch and channel dimensions
            segmentation = segmentation.astype(np.uint8)
            orignalImage = np.squeeze(test_images)

            #Get largest connected component
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(segmentation, connectivity=4)
            sizes = stats[:, -1]
            max_label = 1
            max_size = sizes[1]
            for i in range(2, nb_components):
                if sizes[i] > max_size:
                    max_label = i
                    max_size = sizes[i]

            onlyLargestObject = np.zeros(output.shape)
            onlyLargestObject[output == max_label] = 255
            segmentation = onlyLargestObject.astype(np.uint8)

            _, contours, _= cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(contours[0])

            croppedImage = crop_minAreaRect(orignalImage, rect)
            print croppedImage.shape
            #Rotate all cropped potatoes to be landscape
            h, w = croppedImage.shape[:2]
            if h > w:
                #Swap height and width dimensions
                croppedImage = np.transpose(croppedImage, (1, 0, 2))
            try:
                croppedImage = np.array(cv2.resize(croppedImage,(CLASSIFIER_IMAGE_SIZE,CLASSIFIER_IMAGE_SIZE)))
            except:
                exit()
            test_images = croppedImage[None, :, :] #Add batch dimension back in

            test_loss, pred = sess_classifier.run([classifier.loss, classifier.pred], feed_dict={image_classifier: test_images, 
                                                   labels_classifier: test_labels,
                                                   classifier.keep_prob: 1.0})
            whichClass = pred[0]
            test_labels = test_labels[0]
            if test_labels[whichClass] == 1: #Correct
                print "Correct"
                total_accuracy.append(1)
            else: # Wrong
                print "Wrong"
                total_accuracy.append(0)
            print "test_loss (random batch):", test_loss, "train_loss:", np.mean(train_losses), "mean accuracy", np.mean(total_accuracy)
            saver_classifier.save(sess_classifier, "classifierLogs/" + "ClassifierModel.ckpt", trainStep)
            train_losses = []

main()










    
