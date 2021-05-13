# import numpy as np
# import pandas as pd
# import os
#
# def fix_update_step_col(data):
#    #data/4drones_0.1_CT.csv' pd.read_csv('data/6drones.csv')#pd.read_csv('data/4drones_0.1_CT.csv')#pd.read_csv('data/4drones_path_tracking_0.05.csv')#
#     # parse drone num
#     data.drone = data.drone.apply(lambda x: x.split('Drone')[1]).astype('int')
#     # sort
#     data = data.sort_values(['drone','iter','sim_time'],ascending=True)
#     # arrange and aggrigate drone by new update step
#     drones_ls = data.drone.unique()
#     iters_ls = data.iter.unique()
#
#
#     drones_iters = [(drone,iter) for drone in drones_ls for iter in iters_ls]
#     update_step = [list(range(((data.drone == drone) & (data.iter == iter)).sum())) for drone,iter in  drones_iters]
#
#     update_step = [step for drones in update_step for step in drones]
#     data.update_step = update_step
#
#     # remove invalid steps (not all drones participated)
#     data = data.sort_values(['iter','update_step']).reset_index(drop = True)
#     num_of_drone_per_step = data.groupby(['iter','update_step'],as_index=False).count()[['drone' , 'iter' ,'update_step']]
#     num_of_drone_per_step['valid'] = num_of_drone_per_step.drone == len(drones_ls)
#     invalid_steps_df = num_of_drone_per_step.loc[num_of_drone_per_step.valid==False,:]
#     # print('ratio of invalid steps (not all drones participated): ',np.round(invalid_steps_df.shape[0]/ num_of_drone_per_step.shape[0],3))
#
#     invalid_steps_iter_ls = list(zip(invalid_steps_df.iter.to_list(), invalid_steps_df.update_step.to_list()))
#     steps_iter_ls = list(zip(data.iter.to_list(), data.update_step.to_list()))
#     v = [step not in  invalid_steps_iter_ls for step in steps_iter_ls ]
#     data_with_update_step = data.loc[v,:]
#     print('data after removing invalid updatesteps: ',np.round(data_with_update_step.shape[0]/ data.shape[0],3))
#
#
#     data.drone = data.drone.astype('str')
#
#     # remove time columns
#     # data_with_update_step = data_with_update_step.drop(['script_time','sim_time'],1)
#     return data_with_update_step
#
# def import_process(dir_path,scenario,num_of_comp=0, anomaly_type=''):
#     files_list = os.listdir(dir_path)
#     if anomaly_type == '':
#         file_id = '{}_{}{}'.format(scenario,num_of_comp,anomaly_type)
#     else:
#         # if isinstance(anomaly_type, list):
#         #     for i in anomaly_type:
#         #         file = 'e_{}_{}{}_30.csv'.format(scenario, num_of_comp, anomaly_type)
#         file_id = '{}_{}_{}'.format(scenario, num_of_comp, anomaly_type)
#
#     files = [file for file in files_list if file_id in file]
#     print('files to upload: ', files)
#     if len(files)==1:
#         path = dir_path+files[0]
#         data = pd.read_csv(path)
#     else:
#         df_list = [pd.read_csv(dir_path+file) for file in files]
#         # update iters number to start where iters in 1 ends
#         max_iter_in_0 = df_list[0].iter.max()
#         df_list[1]['iter'] = df_list[1]['iter'] + max_iter_in_0 + 1
#
#         data = pd.concat(df_list)
#
#
#     data = fix_update_step_col(data)
#     # save anomaly type
#     data['anomaly_type'] = anomaly_type
#     data['scenario'] = scenario
#     # add anomaly type to iter field and create another field with original iter
#     data['original_iter']  = data.iter
#     data['iter'] = data.iter.astype('str') +data.anomaly_type + data.scenario
#     return data
#
# def down_samplimg(data,sample_factor = 2,num_of_drones=20,chose_random_drones = True):
#     """Recives data with freqncy X and number of drones Y
#     return data with frequnct X/sample_factor and disired number of drone
#     fix drone number"""
#     data = data.loc[((data['update_step'] % sample_factor) == 0), :]
#     # if there are anomalis chose at list one drone with anomaly
#
#     if (data.label.sum() == 0) | (chose_random_drones == False):
#         data = data.loc[data['drone'].isin(list(range(1,num_of_drones+1))), :]
#     else: # in case of animalies - chose one comp drone and normal others
#
#         iters_ls = data.iter.unique()
#         label_drones_by_iter = data[['iter','drone','label']].groupby(['iter','drone'], as_index=False).max()
#         comp_drones_by_iter_df = label_drones_by_iter.loc[label_drones_by_iter.label==1,:]
#         noraml_drones_by_iter = label_drones_by_iter.loc[label_drones_by_iter.label==0,:]
#
#         # iterate over iterations and pick random sample of normal drones (num_of_drones-1)
#         top_num_of_drones_df = pd.concat([noraml_drones_by_iter.loc[noraml_drones_by_iter.iter == iter,].sample(num_of_drones-1) for iter in iters_ls])
#         # concat comp and normal
#         piked_drones_df = pd.concat([comp_drones_by_iter_df,top_num_of_drones_df])
#         piked_drones_df = piked_drones_df.sort_values(['iter','drone'])
#         # change drones number to start from 1
#         piked_drones_df['scaled_drone_num'] = list(range(1,num_of_drones+1))*int(piked_drones_df.shape[0]/num_of_drones)
#         data_after_down_samp = pd.merge(piked_drones_df[['iter','drone','scaled_drone_num']],data,on = ['iter','drone'])
#         data_after_down_samp = data_after_down_samp.drop('drone',1).rename(columns={"scaled_drone_num": "drone"})
#         data_after_down_samp = data_after_down_samp.sort_values(['iter','update_step'])
#
#         print('down sample to: {}, in practice down sampled to: {}'.format((data.shape[0]/(20))*num_of_drones,data_after_down_samp.shape[0]))
#         return data_after_down_samp.reset_index(drop=True)
#     return data.reset_index(drop=True)
#
# def import_combine_and_downsampeld_data(dir_path, scenario,sample_factor=2, num_of_drones=4,chose_random_drones = True,num_of_comp =1):
#     """Import and preprocees data"""
#     negative_data = import_process(dir_path, scenario, num_of_comp=0, anomaly_type='')
#     positive_data_random = import_process(dir_path, scenario, num_of_comp=num_of_comp, anomaly_type='random')
#     positive_data_path = import_process(dir_path, scenario, num_of_comp=num_of_comp, anomaly_type='path')
#     positive_data_shift = import_process(dir_path, scenario, num_of_comp=num_of_comp, anomaly_type='shift')
#     """Down sample"""
#     negative_data = down_samplimg(negative_data, sample_factor=sample_factor, num_of_drones=num_of_drones,chose_random_drones = chose_random_drones)
#     positive_data_random = down_samplimg(positive_data_random, sample_factor=sample_factor, num_of_drones=num_of_drones,chose_random_drones = chose_random_drones)
#     positive_data_path = down_samplimg(positive_data_path, sample_factor=sample_factor, num_of_drones=num_of_drones,chose_random_drones = chose_random_drones)
#     positive_data_shift = down_samplimg(positive_data_shift, sample_factor=sample_factor, num_of_drones=num_of_drones,chose_random_drones = chose_random_drones)
#     # concat
#     positive_data = pd.concat([positive_data_random, positive_data_path, positive_data_shift])
#     del positive_data_random, positive_data_path, positive_data_shift
#
#     return negative_data, positive_data
