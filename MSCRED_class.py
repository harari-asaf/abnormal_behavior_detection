import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import argparse
import pandas as pd
import os, sys
# os.chdir('/data/home/hsaf/pycharm_projects/anomaly_detection')
import timeit
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class MSCRED:
    def __init__(self, step_max = 5,sensor_n=  21,sensor_m=  21, scale_n = 3,learning_rate = 0.0001,scale_down_filters=1):
        self.step_max =step_max
        self.sensor_n =sensor_n
        self.sensor_m =sensor_m
        self.scale_n =scale_n
        self.learning_rate = learning_rate
        self.scale_down_filters =scale_down_filters

    def createMSCRED(self):


        data_input = tf.placeholder(tf.float32, [self.step_max, self.sensor_n, self.sensor_m, self.scale_n])
        data_y = tf.placeholder(tf.float32, [ self.sensor_n, self.sensor_m, self.scale_n])
        conv1drop_rate = tf.placeholder(tf.float32)
        # parameters: adding bias weight get similar performance
        conv1_W = tf.Variable(tf.zeros([3, 3, self.scale_n, int(32*self.learning_rate)]), name="conv1_W")
        conv1_W = tf.get_variable("conv1_W", shape=[3, 3, self.scale_n, int(32*self.learning_rate)], initializer=tf.contrib.layers.xavier_initializer())
        conv2_W = tf.Variable(tf.zeros([3, 3, int(32*self.learning_rate), int(64*self.learning_rate)]), name="conv2_W")
        conv2_W = tf.get_variable("conv2_W", shape=[3, 3, int(32*self.learning_rate), int(64*self.learning_rate)], initializer=tf.contrib.layers.xavier_initializer())
        conv3_W = tf.Variable(tf.zeros([2, 2, int(64*self.learning_rate), int(128*self.learning_rate)]), name="conv3_W")
        conv3_W = tf.get_variable("conv3_W", shape=[2, 2, int(64*self.learning_rate), int(128*self.learning_rate)], initializer=tf.contrib.layers.xavier_initializer())
        conv4_W = tf.Variable(tf.zeros([2, 2, int(128*self.learning_rate), int(256*self.learning_rate)]), name="conv4_W")
        conv4_W = tf.get_variable("conv4_W", shape=[2, 2, int(128*self.learning_rate), int(256*self.learning_rate)], initializer=tf.contrib.layers.xavier_initializer())

        deconv4_W = tf.Variable(tf.zeros([2, 2, int(128*self.learning_rate), int(256*self.learning_rate)]), name="deconv4_W")
        deconv4_W = tf.get_variable("deconv4_W", shape=[2, 2, int(128*self.learning_rate), int(256*self.learning_rate)], initializer=tf.contrib.layers.xavier_initializer())
        deconv3_W = tf.Variable(tf.zeros([2, 2, int(64*self.learning_rate), int(256*self.learning_rate)]), name="deconv3_W")
        deconv3_W = tf.get_variable("deconv3_W", shape=[2, 2, int(64*self.learning_rate), int(256*self.learning_rate)], initializer=tf.contrib.layers.xavier_initializer())
        deconv2_W = tf.Variable(tf.zeros([3, 3, int(32*self.learning_rate), int(128*self.learning_rate)]), name="deconv2_W")
        deconv2_W = tf.get_variable("deconv2_W", shape=[3, 3,  int(32*self.learning_rate), int(128*self.learning_rate)], initializer=tf.contrib.layers.xavier_initializer())
        deconv1_W = tf.Variable(tf.zeros([3, 3, self.scale_n, int(64*self.learning_rate)]), name="deconv1_W")
        deconv1_W = tf.get_variable("deconv1_W", shape=[3, 3, self.scale_n, int(64*self.learning_rate)], initializer=tf.contrib.layers.xavier_initializer())


        def cnn_encoder(input_matrix,conv1drop_rate):
            input_matrix = tf.nn.dropout(input_matrix, rate=conv1drop_rate)
            conv1 = tf.nn.conv2d(
                input=input_matrix,
                filter=conv1_W,
                strides=(1, 1, 1, 1),
                padding="SAME")
            conv1 = tf.nn.selu(conv1)


            conv2 = tf.nn.conv2d(
                input=conv1,
                filter=conv2_W,
                strides=(1, 2, 2, 1),
                padding="SAME")
            conv2 = tf.nn.selu(conv2)

            conv3 = tf.nn.conv2d(
                input=conv2,
                filter=conv3_W,
                strides=(1, 2, 2, 1),
                padding="SAME")
            conv3 = tf.nn.selu(conv3)

            conv4 = tf.nn.conv2d(
                input=conv3,
                filter=conv4_W,
                strides=(1, 2, 2, 1),
                padding="SAME")
            conv4 = tf.nn.selu(conv4)

            # conv5 = tf.nn.conv2d(
            #   input = conv4,
            #   filter = conv5_W,
            #   strides=(1, 2, 2, 1),
            #   padding = "SAME")
            # conv5 = tf.nn.selu(conv5)

            # print conv5.get_shape()

            return conv1, conv2, conv3, conv4


        def conv1_lstm(conv1_out):
            convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=2,
                input_shape=[self.sensor_n, self.sensor_m, int(32*self.learning_rate)],
                output_channels=int(32*self.learning_rate),
                kernel_shape=[2, 2],
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name="conv1_lstm_cell")

            outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv1_out, time_major=False, dtype=conv1_out.dtype)

            # attention based on transformation of feature representation of last step and other steps
            # outputs_mean = tf.reduce_mean(tf.reduce_mean(outputs[0], axis = 1), axis = 1)
            # outputs_mean_W = tf.tanh(tf.matmul(outputs_mean, atten_2_W))
            # outputs_mean_W = tf.matmul(outputs_mean_W, tf.reshape(outputs_mean_W[-1],[32, 1]))/5.0
            # #outputs_mean_W = tf.matmul(outputs_mean_W, atten_2_V)/5.0
            # attention_w = tf.transpose(tf.nn.softmax(outputs_mean_W, dim=0))

            # attention based on inner-product between feature representation of last step and other steps
            attention_w = []
            for k in range(self.step_max):
                attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1])) / self.step_max)
            attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, self.step_max])

            outputs = tf.reshape(outputs[0], [self.step_max, -1])
            outputs = tf.matmul(attention_w, outputs)
            outputs = tf.reshape(outputs, [1, self.sensor_n, self.sensor_m, int(32*self.learning_rate)])

            return outputs, state[0], attention_w


        def conv2_lstm(conv2_out):
            convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=2,
                input_shape=[int(math.ceil(float(self.sensor_n) / 2)), int(math.ceil(float(self.sensor_m) / 2)), int(64*self.learning_rate)],
                output_channels=int(64*self.learning_rate),
                kernel_shape=[2, 2],
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name="conv2_lstm_cell")

            outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv2_out, time_major=False, dtype=conv2_out.dtype)

            # attention based on transformation of feature representation of last step and other steps
            # outputs_mean = tf.reduce_mean(tf.reduce_mean(outputs[0], axis = 1), axis = 1)
            # outputs_mean_W = tf.tanh(tf.matmul(outputs_mean, atten_2_W))
            # outputs_mean_W = tf.matmul(outputs_mean_W, tf.reshape(outputs_mean_W[-1],[32, 1]))/5.0
            # #outputs_mean_W = tf.matmul(outputs_mean_W, atten_2_V)/5.0
            # attention_w = tf.transpose(tf.nn.softmax(outputs_mean_W, dim=0))

            # attention based on inner-product between feature representation of last step and other steps
            attention_w = []
            for k in range(self.step_max):
                attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1])) / self.step_max)
            attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, self.step_max])

            outputs = tf.reshape(outputs[0], [self.step_max, -1])
            outputs = tf.matmul(attention_w, outputs)
            outputs = tf.reshape(outputs, [1, int(math.ceil(float(self.sensor_n) / 2)), int(math.ceil(float(self.sensor_m) / 2)), int(64*self.learning_rate)])

            return outputs, state[0], attention_w


        def conv3_lstm(conv3_out):
            convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=2,
                input_shape=[int(math.ceil(float(self.sensor_n) / 4)), int(math.ceil(float(self.sensor_m) / 4)), int(128*self.learning_rate)],
                output_channels=int(128*self.learning_rate),
                kernel_shape=[2, 2],
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name="conv3_lstm_cell")

            outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv3_out, time_major=False, dtype=conv3_out.dtype)

            # outputs_mean = tf.reduce_mean(tf.reduce_mean(outputs[0], axis = 1), axis = 1)
            # outputs_mean_W = tf.tanh(tf.matmul(outputs_mean, atten_3_W))
            # outputs_mean_W = tf.matmul(outputs_mean_W, tf.reshape(outputs_mean_W[-1],[64, 1]))/5.0
            # #outputs_mean_W = tf.matmul(outputs_mean_W, atten_3_V)/5.0
            # attention_w = tf.transpose(tf.nn.softmax(outputs_mean_W, dim=0))

            attention_w = []
            for k in range(self.step_max):
                attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1])) / self.step_max)
            attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, self.step_max])

            outputs = tf.reshape(outputs[0], [self.step_max, -1])
            outputs = tf.matmul(attention_w, outputs)
            outputs = tf.reshape(outputs, [1, int(math.ceil(float(self.sensor_n) / 4)), int(math.ceil(float(self.sensor_m) / 4)), int(128*self.learning_rate)])

            return outputs, state[0], attention_w


        def conv4_lstm(conv4_out):
            convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=2,
                input_shape=[int(math.ceil(float(self.sensor_n) / 8)), int(math.ceil(float(self.sensor_m) / 8)), int(256*self.learning_rate)],
                output_channels=int(256*self.learning_rate),
                kernel_shape=[2, 2],
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name="conv4_lstm_cell")

            # initial_state = convlstm_layer.zero_state(batch_size, dtype = tf.float32)
            outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv4_out, time_major=False, dtype=conv4_out.dtype)

            # outputs_mean = tf.reduce_mean(tf.reduce_mean(outputs[0], axis = 1), axis = 1)
            # outputs_mean_W = tf.tanh(tf.matmul(outputs_mean, atten_4_W))
            # outputs_mean_W = tf.matmul(outputs_mean_W, tf.reshape(outputs_mean_W[-1],[128, 1]))/5.0
            # #outputs_mean_W = tf.matmul(outputs_mean_W, atten_4_V)/5.0
            # attention_w = tf.transpose(tf.nn.softmax(outputs_mean_W, dim=0))

            attention_w = []
            for k in range(self.step_max):
                attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1])) / self.step_max)
            attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, self.step_max])

            outputs = tf.reshape(outputs[0], [self.step_max, -1])
            outputs = tf.matmul(attention_w, outputs)
            outputs = tf.reshape(outputs, [1, int(math.ceil(float(self.sensor_n) / 8)), int(math.ceil(float(self.sensor_m) / 8)), int(256*self.learning_rate)])

            return outputs, state[0], attention_w


        def cnn_decoder(conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out):
            conv1_lstm_out = tf.reshape(conv1_lstm_out, [1, self.sensor_n, self.sensor_m, int(32*self.learning_rate)])
            conv2_lstm_out = tf.reshape(conv2_lstm_out,
                                        [1, int(math.ceil(float(self.sensor_n) / 2)), int(math.ceil(float(self.sensor_m) / 2)), int(64*self.learning_rate)])
            conv3_lstm_out = tf.reshape(conv3_lstm_out,
                                        [1, int(math.ceil(float(self.sensor_n) / 4)), int(math.ceil(float(self.sensor_m) / 4)), int(128*self.learning_rate)])
            conv4_lstm_out = tf.reshape(conv4_lstm_out,
                                        [1, int(math.ceil(float(self.sensor_n) / 8)), int(math.ceil(float(self.sensor_m) / 8)), int(256*self.learning_rate)])


            deconv4 = tf.nn.conv2d_transpose(
                value=conv4_lstm_out,
                filter=deconv4_W,
                output_shape=[1, int(math.ceil(float(self.sensor_n) / 4)), int(math.ceil(float(self.sensor_m) / 4)), int(128*self.learning_rate)],
                strides=(1, 2, 2, 1),
                padding="SAME")
            deconv4 = tf.nn.selu(deconv4)
            deconv4_concat = tf.concat([deconv4, conv3_lstm_out], axis=3)

            deconv3 = tf.nn.conv2d_transpose(
                value=deconv4_concat,
                filter=deconv3_W,
                output_shape=[1, int(math.ceil(float(self.sensor_n) / 2)), int(math.ceil(float(self.sensor_m) / 2)), int(64*self.learning_rate)],
                strides=(1, 2, 2, 1),
                padding="SAME")
            deconv3 = tf.nn.selu(deconv3)
            deconv3_concat = tf.concat([deconv3, conv2_lstm_out], axis=3)

            deconv2 = tf.nn.conv2d_transpose(
                value=deconv3_concat,
                filter=deconv2_W,
                output_shape=[1, self.sensor_n, self.sensor_m, int(32*self.learning_rate)],
                strides=(1, 2, 2, 1),
                padding="SAME")
            deconv2 = tf.nn.selu(deconv2)

            deconv2_concat = tf.concat([deconv2, conv1_lstm_out], axis=3)

            deconv1 = tf.nn.conv2d_transpose(
                value=deconv2_concat,
                filter=deconv1_W,
                output_shape=[1, self.sensor_n, self.sensor_m, self.scale_n],
                strides=(1, 1, 1, 1),
                padding="SAME")
            deconv1 = tf.nn.selu(deconv1)
            deconv1 = tf.reshape(deconv1, [1, self.sensor_n, self.sensor_m, self.scale_n])
            return deconv1



        conv1_out, conv2_out, conv3_out, conv4_out = cnn_encoder(data_input,conv1drop_rate)
        conv1_out = tf.reshape(conv1_out, [-1, self.step_max, self.sensor_n, self.sensor_m, int(32*self.learning_rate)])
        conv2_out = tf.reshape(conv2_out,
                               [-1, self.step_max, int(math.ceil(float(self.sensor_n) / 2)), int(math.ceil(float(self.sensor_m) / 2)), int(64*self.learning_rate)])
        conv3_out = tf.reshape(conv3_out,
                               [-1, self.step_max, int(math.ceil(float(self.sensor_n) / 4)), int(math.ceil(float(self.sensor_m) / 4)), int(128*self.learning_rate)])
        conv4_out = tf.reshape(conv4_out,
                               [-1, self.step_max, int(math.ceil(float(self.sensor_n) / 8)), int(math.ceil(float(self.sensor_m) / 8)), int(256*self.learning_rate)])

        conv1_lstm_attention_out, conv1_lstm_last_out, atten_weight_1 = conv1_lstm(conv1_out)
        conv2_lstm_attention_out, conv2_lstm_last_out, atten_weight_2 = conv2_lstm(conv2_out)
        conv3_lstm_attention_out, conv3_lstm_last_out, atten_weight_3 = conv3_lstm(conv3_out)
        conv4_lstm_attention_out, conv4_lstm_last_out, atten_weight_4 = conv4_lstm(conv4_out)

        deconv_out = cnn_decoder(conv1_lstm_attention_out, conv2_lstm_attention_out, conv3_lstm_attention_out,
                                 conv4_lstm_attention_out)


        import tensorflow.contrib.slim as slim

        def model_summary():
            model_vars = tf.trainable_variables()
            slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        model_summary()

        # loss function: reconstruction error of last step matrix
        loss = tf.reduce_mean(tf.square(data_y - deconv_out))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=5)

        return init, loss, optimizer, saver, data_y,data_input, deconv_out, conv1drop_rate


        init, loss, optimizer, saver, data_y,data_input, deconv_out, conv1drop_rate = createMSCRED()