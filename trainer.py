import numpy as np
import tensorflow as tf
import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)
import os, sys
import time
from tqdm import tqdm

from MSCRED import createMSCRED
import gc

class Trainer:
    """"config (dict"""
    def __init__(self, train_X, val_X, test_X, config):
        self.config = config
        self.train_X = train_X
        self.val_X = val_X
        self.test_X = test_X


        """Open dir for results"""
        model_path = config['model_results_output_folder']
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # train
        tf.reset_default_graph()
        self.init, self.loss, self.optimizer, self.saver, self.data_y, self.data_input, self.deconv_out, self.conv1drop_rate = createMSCRED(
            step_max=config['step_max'],
            sensor_n=config['sensor_n'],
            scale_n=config['scale_n'],
            sensor_m=config['sensor_m'],
            learning_rate=config['learning_rate'],
            scale_down_filters=config['scale_down_filter'])




    def create_batch_in_step_max_size(self,first_step, step_max, data_scaled, data_y_scaled='', drop_rate=0.0):
        steps = range(first_step - step_max, first_step)
        # print('steps: ', steps, ' out of: ', final_train_X.shape[0])
        matrix_gt = data_scaled[steps]

        if data_y_scaled == '':
            matrix_gt_y = data_scaled[steps][-1]
        else:
            matrix_gt_y = data_y_scaled[steps][-1]

        feed_dict = {self.data_input: matrix_gt, self.data_y: matrix_gt_y, self.conv1drop_rate: drop_rate}
        return feed_dict


    def run_on_singal_batch(self,first_step, step_max, data_scaled, operations, data_y_scaled='', drop_rate=0.0):
        """execute the computation graph with the given operations on one chunk of data X, from first_step to first_step+step_max.
            recive: first_step: index to start the chunk from (int)
                    step max: number of time steps (int)
                    data_scaled: X of the dataset (numpy array (num_of_sumples, mat_first_dim, mat_second_dim,num of win sizes ) )
                    data_y_scaled: Y of the dataset
                    operations: which operations to run (list)
            return: value: the output of the computations graph
                    """

        feed_dict = self.create_batch_in_step_max_size(first_step, step_max, data_scaled, data_y_scaled=data_y_scaled,
                                                  drop_rate=drop_rate)

        if len(operations) == 2:
            a, value = self.sess.run(operations, feed_dict)
        else:
            value = self.sess.run(operations, feed_dict)[0]

        return value


    def run_on_dataset(self, step_max, data_scaled, operations, ephoc=0, data_y_scaled='', drop_rate=0.0, random_samplig=False):
        """Iterates over the data, each iteration pass chunk in size of step_max to function that execute the computation graph
          recive: step max: number of time steps (int)
                  data_scaled: X of the datasets (numpy array (num_of_sumples, mat_first_dim, mat_second_dim,num of win sizes ) )
                  operations: which operations to run (list)
          return: results_ls: the output of the computations (list)
                  """
        print('dropout rate is: ', drop_rate)
        start = time.time()
        steps_results_ls = []
        iter_idx_ls = list(range(len(data_scaled)))

        if random_samplig:
            print('START RANDOM SAMPLING')
            """ shuffle iterationes and samples, feed into the model """
            feed_dicts_ls = []
            # iterate over iters
            for sim_iter_idx in iter_idx_ls:
                # get iter data
                sim_iter_data = data_scaled[sim_iter_idx]
                sim_iter_data_y = data_y_scaled[sim_iter_idx]

                step_idx_ls = list(range(step_max, sim_iter_data.shape[0]))

                for first_step in step_idx_ls:
                    feed_dict = self.create_batch_in_step_max_size(first_step, step_max, sim_iter_data,
                                                              data_y_scaled=sim_iter_data_y, drop_rate=drop_rate)
                    feed_dicts_ls.append(feed_dict)

            print('samples per ephoc:', len(feed_dicts_ls))
            np.random.seed(ephoc)
            np.random.shuffle(feed_dicts_ls)
            for feed_dict in tqdm(feed_dicts_ls):
                # only in train mode their are two inputs
                a, value = self.sess.run(operations, feed_dict)
                steps_results_ls.append(value)


        else:  # feed samples by order
            iter_results_ls = []
            for sim_iter_idx in iter_idx_ls:
                sim_iter_data = data_scaled[sim_iter_idx]
                sim_iter_data_y = data_y_scaled[sim_iter_idx]

                step_idx_ls = list(range(step_max, sim_iter_data.shape[0]))
                steps_results_ls = []
                for first_step in step_idx_ls:
                    # print(first_step)
                    output = self.run_on_singal_batch(first_step, step_max, sim_iter_data, operations=operations,
                                                 data_y_scaled=sim_iter_data_y, drop_rate=drop_rate)
                    steps_results_ls.append(np.squeeze(output))
                iter_results_ls.append(np.array(steps_results_ls))
            return iter_results_ls

        print('time of ephoc: ', (time.time() - start))
        # print(steps_results_ls[:20])
        return steps_results_ls


    def train_MSCRED(self, random_samplig=False,no_improvment_till_stop=1):
        with tf.Session() as self.sess:
            ephocs_loss = []
            ephocs_val_loss = []
            model_ephocs_loss = {'loss': [], 'val_loss': [], 'LR': [], 'ephoc': [], 'model_path_and_name': []}
            exp_id = str(np.round(time.time(), 4))
            # config=tf.ConfigProto() inter_op_parallelism_threads=80, intra_op_parallelism_threads=80,
            devices = self.sess.list_devices()
            print('==============devices:============================\n', devices)
            self.sess.run(self.init)
            no_improvment_counter = 0
            print('START TRAINING')
            for ephoc in range(self.config['ephoces_num']):

                print('ephoce: ', ephoc)

                start_tarin = time.time()
                # run computations
                loss_ls = self.run_on_dataset(self.config['step_max'], data_scaled=self.train_X, operations=[self.optimizer, self.loss], ephoc=ephoc,
                                         data_y_scaled=self.train_X, drop_rate=self.config['drop_rate'], random_samplig=random_samplig)
                mean_loss = np.mean(loss_ls);
                print('Time took to epoc {} : {}, loss: {}'.format(ephoc, time.time() - start_tarin, mean_loss))

                start_val = time.time()
                val_loss_ls = self.run_on_dataset(self.config['step_max'], data_scaled=self.val_X, operations=[self.loss],
                                             data_y_scaled=self.val_X, drop_rate=0.0)

                # comp;ute mean, print and save

                mean_val_loss = np.mean(val_loss_ls)
                print('Time took to epoc {} : {}, loss: {}'.format(ephoc, time.time() - mean_val_loss, mean_val_loss))

                ephocs_loss.append(mean_loss);
                ephocs_val_loss.append(mean_val_loss)
                model_name = exp_id +str(ephoc)+ ".ckpt"  # mscredStright'


                model_wights_file = self.config['model_results_output_folder'] + model_name
                # os.mkdir(model_wights_folder)
                # save models with emprovement, stop after no improvnemt == 2
                if ephoc != 0:
                    if ephocs_val_loss[ephoc] < ephocs_val_loss[ephoc - 1]:
                        print('improvement!')
                        self.saver.save(self.sess, model_wights_file)
                    else:
                        no_improvment_counter += 1
                        print('no improvement')
                else:
                    self.saver.save(self.sess, model_wights_file)  # always_save first ep
                if no_improvment_counter > no_improvment_till_stop:
                    print('break')
                    break

                model_ephocs_loss['loss'].append(mean_loss);
                model_ephocs_loss['val_loss'].append(mean_val_loss);
                model_ephocs_loss['LR'].append(self.config['learning_rate'])
                model_ephocs_loss['model_path_and_name'].append(model_wights_file), model_ephocs_loss['ephoc'].append(ephoc)

            print('\n============iteration summary : ' + str(self.config['learning_rate']))
            print('loss: ', ephocs_loss)
            print('Val_loss: ', ephocs_val_loss)
            model_ephocs_loss_df = pd.DataFrame.from_dict(model_ephocs_loss)
            self.final_model_path = model_wights_file
        return model_ephocs_loss_df


    def recunstract_val_and_test(self):
        with tf.Session() as self.sess:  # config=tf.ConfigProto() inter_op_parallelism_threads=80, intra_op_parallelism_threads=80,
            devices = self.sess.list_devices()
            print('==============devices:============================\n', devices)
            self.sess.run(self.init)
            self.saver.restore(self.sess, self.final_model_path)  #
            print('num of val iters: {} num of test iters {}'.format(len(self.val_X),len(self.test_X)))

            recunstract_val_ls = self.run_on_dataset(self.config['step_max'], data_scaled=self.val_X, operations=[self.deconv_out],
                                                data_y_scaled=self.val_X)

            recunstract_test_ls = self.run_on_dataset(self.config['step_max'], data_scaled=self.test_X, operations=[self.deconv_out],
                                                data_y_scaled=self.test_X)



            return recunstract_val_ls, recunstract_test_ls







"""Reconstruct valdiation and test"""







