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



def createMSCRED(step_max = 5,sensor_n=  21,sensor_m=  21, scale_n = 3,learning_rate = 0.0001,scale_down_filters=1):

    # 4d data
    # step_max = 5
    # sensor_n, scale_n =  21, 3
    data_input = tf.placeholder(tf.float32, [step_max, sensor_n, sensor_m, scale_n])
    data_y = tf.placeholder(tf.float32, [ sensor_n, sensor_m, scale_n])
    conv1drop_rate = tf.placeholder(tf.float32)
    # parameters: adding bias weight get similar performance
    conv1_W = tf.Variable(tf.zeros([3, 3, scale_n, int(32*scale_down_filters)]), name="conv1_W")
    conv1_W = tf.get_variable("conv1_W", shape=[3, 3, scale_n, int(32*scale_down_filters)], initializer=tf.contrib.layers.xavier_initializer())
    conv2_W = tf.Variable(tf.zeros([3, 3, int(32*scale_down_filters), int(64*scale_down_filters)]), name="conv2_W")
    conv2_W = tf.get_variable("conv2_W", shape=[3, 3, int(32*scale_down_filters), int(64*scale_down_filters)], initializer=tf.contrib.layers.xavier_initializer())
    conv3_W = tf.Variable(tf.zeros([2, 2, int(64*scale_down_filters), int(128*scale_down_filters)]), name="conv3_W")
    conv3_W = tf.get_variable("conv3_W", shape=[2, 2, int(64*scale_down_filters), int(128*scale_down_filters)], initializer=tf.contrib.layers.xavier_initializer())
    conv4_W = tf.Variable(tf.zeros([2, 2, int(128*scale_down_filters), int(256*scale_down_filters)]), name="conv4_W")
    conv4_W = tf.get_variable("conv4_W", shape=[2, 2, int(128*scale_down_filters), int(256*scale_down_filters)], initializer=tf.contrib.layers.xavier_initializer())

    deconv4_W = tf.Variable(tf.zeros([2, 2, int(128*scale_down_filters), int(256*scale_down_filters)]), name="deconv4_W")
    deconv4_W = tf.get_variable("deconv4_W", shape=[2, 2, int(128*scale_down_filters), int(256*scale_down_filters)], initializer=tf.contrib.layers.xavier_initializer())
    deconv3_W = tf.Variable(tf.zeros([2, 2, int(64*scale_down_filters), int(256*scale_down_filters)]), name="deconv3_W")
    deconv3_W = tf.get_variable("deconv3_W", shape=[2, 2, int(64*scale_down_filters), int(256*scale_down_filters)], initializer=tf.contrib.layers.xavier_initializer())
    deconv2_W = tf.Variable(tf.zeros([3, 3, int(32*scale_down_filters), int(128*scale_down_filters)]), name="deconv2_W")
    deconv2_W = tf.get_variable("deconv2_W", shape=[3, 3,  int(32*scale_down_filters), int(128*scale_down_filters)], initializer=tf.contrib.layers.xavier_initializer())
    deconv1_W = tf.Variable(tf.zeros([3, 3, scale_n, int(64*scale_down_filters)]), name="deconv1_W")
    deconv1_W = tf.get_variable("deconv1_W", shape=[3, 3, scale_n, int(64*scale_down_filters)], initializer=tf.contrib.layers.xavier_initializer())


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
            input_shape=[sensor_n, sensor_m, int(32*scale_down_filters)],
            output_channels=int(32*scale_down_filters),
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
        for k in range(step_max):
            attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1])) / step_max)
        attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

        outputs = tf.reshape(outputs[0], [step_max, -1])
        outputs = tf.matmul(attention_w, outputs)
        outputs = tf.reshape(outputs, [1, sensor_n, sensor_m, int(32*scale_down_filters)])

        return outputs, state[0], attention_w


    def conv2_lstm(conv2_out):
        convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
            conv_ndims=2,
            input_shape=[int(math.ceil(float(sensor_n) / 2)), int(math.ceil(float(sensor_m) / 2)), int(64*scale_down_filters)],
            output_channels=int(64*scale_down_filters),
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
        for k in range(step_max):
            attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1])) / step_max)
        attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

        outputs = tf.reshape(outputs[0], [step_max, -1])
        outputs = tf.matmul(attention_w, outputs)
        outputs = tf.reshape(outputs, [1, int(math.ceil(float(sensor_n) / 2)), int(math.ceil(float(sensor_m) / 2)), int(64*scale_down_filters)])

        return outputs, state[0], attention_w


    def conv3_lstm(conv3_out):
        convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
            conv_ndims=2,
            input_shape=[int(math.ceil(float(sensor_n) / 4)), int(math.ceil(float(sensor_m) / 4)), int(128*scale_down_filters)],
            output_channels=int(128*scale_down_filters),
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
        for k in range(step_max):
            attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1])) / step_max)
        attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

        outputs = tf.reshape(outputs[0], [step_max, -1])
        outputs = tf.matmul(attention_w, outputs)
        outputs = tf.reshape(outputs, [1, int(math.ceil(float(sensor_n) / 4)), int(math.ceil(float(sensor_m) / 4)), int(128*scale_down_filters)])

        return outputs, state[0], attention_w


    def conv4_lstm(conv4_out):
        convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
            conv_ndims=2,
            input_shape=[int(math.ceil(float(sensor_n) / 8)), int(math.ceil(float(sensor_m) / 8)), int(256*scale_down_filters)],
            output_channels=int(256*scale_down_filters),
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
        for k in range(step_max):
            attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1])) / step_max)
        attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

        outputs = tf.reshape(outputs[0], [step_max, -1])
        outputs = tf.matmul(attention_w, outputs)
        outputs = tf.reshape(outputs, [1, int(math.ceil(float(sensor_n) / 8)), int(math.ceil(float(sensor_m) / 8)), int(256*scale_down_filters)])

        return outputs, state[0], attention_w


    def cnn_decoder(conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out):
        conv1_lstm_out = tf.reshape(conv1_lstm_out, [1, sensor_n, sensor_m, int(32*scale_down_filters)])
        conv2_lstm_out = tf.reshape(conv2_lstm_out,
                                    [1, int(math.ceil(float(sensor_n) / 2)), int(math.ceil(float(sensor_m) / 2)), int(64*scale_down_filters)])
        conv3_lstm_out = tf.reshape(conv3_lstm_out,
                                    [1, int(math.ceil(float(sensor_n) / 4)), int(math.ceil(float(sensor_m) / 4)), int(128*scale_down_filters)])
        conv4_lstm_out = tf.reshape(conv4_lstm_out,
                                    [1, int(math.ceil(float(sensor_n) / 8)), int(math.ceil(float(sensor_m) / 8)), int(256*scale_down_filters)])


        deconv4 = tf.nn.conv2d_transpose(
            value=conv4_lstm_out,
            filter=deconv4_W,
            output_shape=[1, int(math.ceil(float(sensor_n) / 4)), int(math.ceil(float(sensor_m) / 4)), int(128*scale_down_filters)],
            strides=(1, 2, 2, 1),
            padding="SAME")
        deconv4 = tf.nn.selu(deconv4)
        deconv4_concat = tf.concat([deconv4, conv3_lstm_out], axis=3)

        deconv3 = tf.nn.conv2d_transpose(
            value=deconv4_concat,
            filter=deconv3_W,
            output_shape=[1, int(math.ceil(float(sensor_n) / 2)), int(math.ceil(float(sensor_m) / 2)), int(64*scale_down_filters)],
            strides=(1, 2, 2, 1),
            padding="SAME")
        deconv3 = tf.nn.selu(deconv3)
        deconv3_concat = tf.concat([deconv3, conv2_lstm_out], axis=3)

        deconv2 = tf.nn.conv2d_transpose(
            value=deconv3_concat,
            filter=deconv2_W,
            output_shape=[1, sensor_n, sensor_m, int(32*scale_down_filters)],
            strides=(1, 2, 2, 1),
            padding="SAME")
        deconv2 = tf.nn.selu(deconv2)

        deconv2_concat = tf.concat([deconv2, conv1_lstm_out], axis=3)

        deconv1 = tf.nn.conv2d_transpose(
            value=deconv2_concat,
            filter=deconv1_W,
            output_shape=[1, sensor_n, sensor_m, scale_n],
            strides=(1, 1, 1, 1),
            padding="SAME")
        deconv1 = tf.nn.selu(deconv1)
        deconv1 = tf.reshape(deconv1, [1, sensor_n, sensor_m, scale_n])
        return deconv1



    conv1_out, conv2_out, conv3_out, conv4_out = cnn_encoder(data_input,conv1drop_rate)
    conv1_out = tf.reshape(conv1_out, [-1, step_max, sensor_n, sensor_m, int(32*scale_down_filters)])
    conv2_out = tf.reshape(conv2_out,
                           [-1, step_max, int(math.ceil(float(sensor_n) / 2)), int(math.ceil(float(sensor_m) / 2)), int(64*scale_down_filters)])
    conv3_out = tf.reshape(conv3_out,
                           [-1, step_max, int(math.ceil(float(sensor_n) / 4)), int(math.ceil(float(sensor_m) / 4)), int(128*scale_down_filters)])
    conv4_out = tf.reshape(conv4_out,
                           [-1, step_max, int(math.ceil(float(sensor_n) / 8)), int(math.ceil(float(sensor_m) / 8)), int(256*scale_down_filters)])

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





    #
    model_summary()
    #

    # matrix_gt = drone_sig_mats_np[:5]
    #
    # with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=80, intra_op_parallelism_threads=80)) as sess:
    # 	sess.run(init)
    #
    # 	matrix_gt = np.ones((step_max,sensor_n,sensor_m, scale_n))
    #
    # 	feed_dict = {data_input: np.asarray(matrix_gt)}
    # 	a, loss_value = sess.run([optimizer, loss], feed_dict)

    # data = drone_sig_mats_np



    # loss function: reconstruction error of last step matrix
    loss = tf.reduce_mean(tf.square(data_y - deconv_out))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5)

    return init, loss, optimizer, saver, data_y,data_input, deconv_out, conv1drop_rate

def update_learning_rate(loss,learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return optimizer


# model_path = 'models/MSCRED/CA/CA_LR_'+str(learning_rate)#mscredStright'
# saved_files_name = 'CA_MSCRED'# 'SL_MSCRED'
# input_data_path = 'model_input_data/'+saved_files_name
# #
# # final_train_X = np.load(input_data_path + '_final_train_X' + '.npy')
# # final_train_Y = np.load(input_data_path + '_final_train_Y' + '.npy')
# #
# # final_val_X = np.load(input_data_path + '_final_val_X' + '.npy')
# # final_val_Y = np.load(input_data_path + '_final_val_Y' + '.npy')
# #
# # final_train_positive_X = np.load(input_data_path + '_final_positive_X' + '.npy')
# # final_train_positive_Y = np.load(input_data_path + '_final_positive_Y' + '.npy')
# #
#
#
# final_train_X = final_train_X[:,:,:,-1]
# final_val_X = final_val_X[:,:,:,-1]
# final_train_positive_X = final_train_positive_X[:,:,:,-1]
#
# ephoces_num = 10
# with tf.Session() as sess:  # config=tf.ConfigProto() inter_op_parallelism_threads=80, intra_op_parallelism_threads=80,
#     devices = sess.list_devices()
#     print('==============devices:============================\n', devices)
#     sess.run(init)
#     # saver.restore(sess, model_path + str(0) + str(0.0) + ".ckpt")
#     # saver.restore(sess, model_path + str(9) + str(0.02675) + ".ckpt")
#     ephocs_loss = []
#     ephocs_val_loss = []
#     mean_loss, mean_val_loss = np.inf, np.inf
#
#     # """Compute initial loss"""
#     # loss_ls = run_on_dataset(step_max, data_scaled = final_train_X, operations = [loss],data_y_scaled=final_train_Y)
#     # val_loss_ls = run_on_dataset(step_max, data_scaled = final_val_X, operations = [loss],data_y_scaled=final_val_Y)
#     #
#     #
#     # mean_loss, mean_val_loss = np.mean(loss_ls), np.mean(val_loss_ls)
#     # print('initail loss: {}, initial val loss : {}'.format(mean_loss, mean_val_loss))
#     # ephocs_loss.append(mean_loss)
#     # ephocs_val_loss.append(mean_val_loss)
#
#     for ephoc in range(ephoces_num):
#         start = time.time()
#         print('ephoce: ', ephoc)
#         # run computations
#         loss_ls = run_on_dataset(step_max, data_scaled=final_train_X, operations=[optimizer, loss],data_y_scaled=final_train_Y)
#         val_loss_ls = run_on_dataset(step_max, data_scaled=final_val_X, operations=[loss],data_y_scaled=final_val_Y)
#         # comp;ute mean, print and save
#         mean_loss = np.mean(loss_ls) ; mean_val_loss = np.mean(val_loss_ls)
#         print(mean_loss); print(mean_val_loss)
#         ephocs_loss.append(mean_loss); ephocs_val_loss.append(mean_val_loss)
#
#         saver.save(sess, model_path + str(ephoc) + str(np.round(mean_val_loss, 20)) + ".ckpt")
#
# # # plt.figure()
# # # plt.plot(loss_ls)
# # # plt.savefig('plots/mscredLoss.png')
# #
# # positive_final_train_X = scale_matrix(positive_train_data, mat_min, mat_max)
# # val_data_scaled = scale_matrix(val_data, mat_min, mat_max)
#
# # final_train_positive_X, final_train_positive_Y
#
# # detect anomalies
# with tf.Session() as sess:  # config=tf.ConfigProto() inter_op_parallelism_threads=80, intra_op_parallelism_threads=80,
#     devices = sess.list_devices()
#     print('==============devices:============================\n', devices)
#     sess.run(init)
#     # saver.restore(sess, model_path + str(9) + str(0.02675) + ".ckpt")
#     # saver.restore(sess, model_path + str(0) + str(0.00032143138) + ".ckpt")
#     # saver.restore(sess, model_path + str(0) + str(0.0306747) + ".ckpt") # CA 5
#     # saver.restore(sess, model_path + str(0) + str(0.028982567) + ".ckpt") # CA 5
#     saver.restore(sess, model_path + str(0) + str(0.04997124) + ".ckpt") # CA 2
#     recunstract_val_ls = run_on_dataset(step_max, data_scaled=final_val_X, operations = [deconv_out],data_y_scaled=final_val_Y)
#     recunstract_positive_ls = run_on_dataset(step_max, data_scaled=final_train_positive_X, operations = [deconv_out],data_y_scaled=final_train_positive_Y)
#
#
#
# recunstract_val_np = np.array(recunstract_val_ls).reshape(final_val_X[step_max:].shape)
# recunstract_positive_np = np.array(recunstract_positive_ls).reshape(final_train_positive_X[step_max:].shape)
#
# # np.save('recunstract_val_np'+'.npy',recunstract_val_np)
# # np.save('recunstract_positive_np'+'.npy',recunstract_positive_np)
#
# positive_mse = np.mean(np.abs(recunstract_positive_np - final_train_positive_Y[step_max:]))
# val_mse = np.mean(np.abs(recunstract_val_np - final_val_Y[step_max:]))
# np.mean(np.mean(np.abs(recunstract_positive_np - final_train_positive_Y[step_max:]),axis=(1,2,3))[train_positive_label[step_max:]==0])
# np.mean(np.mean(np.abs(recunstract_positive_np - final_train_positive_Y[step_max:]),axis=(1,2,3))[train_positive_label[step_max:]==1])
#
# positive_mse_by_rec = np.mean(np.abs(recunstract_positive_np - final_train_positive_Y[step_max:]),axis=(1,2,3))
# val_mse_by_rec = np.mean(np.abs(recunstract_val_np - final_val_Y[step_max:]),axis=(1,2,3))
#
# plt.figure()
# mse_vs_label = pd.DataFrame({'label': list(train_positive_label[step_max:]), 'mse': list(positive_mse_by_rec)}, columns=['label', 'mse'])
# sns.boxplot(x='label', y='mse',data = mse_vs_label.loc[mse_vs_label.mse<30,:])
# plt.savefig('plots/mse_vs_labelMsCRED')
#
# def gaussian_anomaly_score(Y_val,Y_test):
#     mean_without_model = np.mean(Y_val,axis = 0)
#     std_without_model = np.std(Y_val,axis = 0)
#     cil_per_pixel = mean_without_model-std_without_model*3
#     ciu_per_pixel = mean_without_model+std_without_model*3
#
#     pre_per_pixel = (Y_test>ciu_per_pixel) | (Y_test<cil_per_pixel)
#
#     dims_except_first = tuple(range(1, len(Y_test.shape)))
#     anomaly_score = np.sum(pre_per_pixel,dims_except_first)
#     return anomaly_score
#
# recall = 0.85
# beta = 1
# # from metrics_and_scores import compute_thr_and_metrics_on_drones
# anomaly_score = compute_anomaly_score_per_pixel(Y_val=final_val_Y[step_max:], pre_val=recunstract_val_np, Y_test=final_train_positive_Y[step_max:], pre_test=recunstract_positive_np, beta=beta, method='MSCRED')
# # compute_thr_and_metrics_on_drones(anomaly_score, train_positive_label[step_max:],beta=beta, recall=recall, plot_scoreVSanomaly=True, plot_name = 'scoreVSlabelMSCRED_stright.png',maxScoreForPlot=5)
#
# train_positive_keys.drone = train_positive_keys.drone.astype('object')
# res, seqs_with_pre = compute_detection_matrics(anomaly_score = anomaly_score, labels = train_positive_label[step_max:],beta = beta,
#                                                recall = recall, original_df=positive_train,
#                           keys =train_positive_keys[step_max:] ,down_sampled_records = '', plot_scoreVSanomaly=True, plot_name = 'plots/scoreVSlabelMSCRED_CA.png')
#
#
#
# res_all_true, seqs_with_pre = compute_detection_matrics(anomaly_score = np.ones(anomaly_score.shape), labels = train_positive_label[step_max:],beta = beta,
#                                                recall = recall, original_df=positive_train,
#                           keys =train_positive_keys[step_max:] ,down_sampled_records = '', plot_scoreVSanomaly=False, plot_name = 'plots/scoreVSlabelMSCRED_CA.png')
#
#
#
#
# res_random, seqs_with_pre = compute_detection_matrics(anomaly_score = np.random.randint(0, 2, anomaly_score.shape[0]), labels = train_positive_label[step_max:],beta = beta,
#                                                recall = recall, original_df=positive_train,
#                           keys =train_positive_keys[step_max:] ,down_sampled_records = '', plot_scoreVSanomaly=False, plot_name = 'plots/scoreVSlabelMSCRED_CA.png')
#
#
#
#
# res_df = pd.DataFrame([res,res_all_true,res_random])
#
# anomaly_in_swarm = seqs_with_pre[['iter', 'update_step', 'label', 'pre']].groupby(['iter', 'update_step'],
#                                                                                            as_index=False).max().dropna()
#
# score_in_swarm = seqs_with_pre[['iter', 'update_step',  'anomaly_score']].groupby(['iter', 'update_step'],
#                                                                                            as_index=False).mean().dropna()
#
# anomaly_in_swarm_and_score = anomaly_in_swarm.merge(score_in_swarm, on = ['iter', 'update_step'])
#
# """Plot of TRUE/FALSE distribution against num of prediction errors """
# plt.figure()
# sns.boxplot(y='anomaly_score', x='label', data=anomaly_in_swarm_and_score.loc[:, :]) # scoreVSanomaly['score'] < 5
# plt.savefig('plots/scoreVSlabelMSCRED_CA_swarm.png')
# print('save plot')
#
#
#
# gaus_anomaly_score = gaussian_anomaly_score(final_val_X[step_max:],final_train_positive_X[step_max:])
# gaus_anomaly_score_val = gaussian_anomaly_score(final_val_X[step_max:],final_val_X[step_max:])
#
# compute_thr_and_metrics_on_drones(gaus_anomaly_score, positive_train_labels[step_max:],beta=beta, recall=recall, plot_scoreVSanomaly=True, plot_name = 'plots/GAUSscoreVSlabel.png')
#
#
#
# compute_thr_and_metrics_on_drones(np.ones(anomaly_score.shape[0]), positive_train_labels[step_max:],beta=beta, recall=recall, plot_scoreVSanomaly=False, plot_name = 'plots/scoreVSlabel.png')
# compute_thr_and_metrics_on_drones(np.random.randint(0,2,anomaly_score.shape[0]), positive_train_labels[step_max:],beta=1, recall=recall, plot_scoreVSanomaly=False, plot_name = 'plots/scoreVSlabel.png')
#
#
#
#
#
#
#
#
# anomaly_score = compute_anomaly_score_per_pixel(Y_val=final_val_X[step_max:], pre_val = recunstract_val_np,
#                                                 Y_test = final_train_positive_X[step_max:], pre_test = recunstract_positive_np,
#                                                 beta = 1, method = 'CNN')
#
#
#
# gaus_anomaly_score = gaussian_anomaly_score(Y_val=final_val_X,Y_test=final_train_positive_X)
#
# results_dict, sequence_with_pred = compute_detection_matrics(anomaly_score=anomaly_score, labels=positive_train_labels[step_max:],beta=beta,
#                           recall= 0.9, original_df=positive_train[step_max:], keys=positive_train_keys[step_max:],
#                           down_sampled_records='',
#                           plot_scoreVSanomaly=False, plot_name='plots/scoreVSlabelAvoidanceMSCRED.png')
#
# results_dict, sequence_with_pred = compute_detection_matrics(anomaly_score=gaus_anomaly_score, labels=positive_train_labels,beta=beta,
#                           recall= 0.9, original_df=positive_train, keys=positive_train_keys,
#                           down_sampled_records='',
#                           plot_scoreVSanomaly=True, plot_name='plots/scoreVSlabelAvoidancegaus.png')
#
#
#
#
#
#
#
#
#
#
#
# lossVSanomaly =   pd.DataFrame(np.array([loss_ls, positive_train_labels[step_max:(final_train_positive_X.shape[0] - 1) ]]).T,
#                                   columns=['score', 'label'])
#
# lossVSanomaly.groupby('label').describe()
# plot_name = 'plots/MSCREDlossVSanomaly.png'
# """Plot of TRUE/FALSE distribution against num of prediction errors """
# plt.figure()
# sns.boxplot(y='score', x='label', data=lossVSanomaly.loc[lossVSanomaly.score<0.00001, :])
# plt.savefig(plot_name)
# print('save plot')
#
#
# # anomaly_score = compute_anomaly_score_per_pixel(Y_val=final_val_Y, pre_val = pre_val, Y_test = final_train_positive_Y, pre_test = pre_positive, beta = beta, method = method)
# #
# #
# #
# # results_dict, sequence_with_pred = compute_detection_matrics(anomaly_score=anomaly_score, labels=train_positive_label_flattern,beta=beta,
# #                           recall= recall, original_df=positive_train, keys=train_positive_keys,
# #                           down_sampled_records=down_sampled_records,
# #                           plot_scoreVSanomaly=False, plot_name='plots/scoreVSlabelAvoidance.png')
