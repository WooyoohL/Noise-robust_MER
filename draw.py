# import matplotlib as mpl
#
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# from utils.logger import get_logger, ResultRecorder, LossRecorder
# import pickle
# import numpy as np
# import random
# from sklearn.manifold import TSNE
# import os
# import warnings
#
# warnings.filterwarnings("ignore")  # 忽略警告
#
# TSNE_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=3000, random_state=23)
#
#
# def read_pkl(path, num_list, basename):
#     data = []
#     for num in num_list:
#         filename = num + '_' + basename
#         with open((os.path.join(path, filename)), 'rb') as tf:
#             _ = pickle.load(tf)
#             data.append(_)
#         tf.close()
#     return data
#     pass
#
#
# def build_array(shared_lists):
#     shared_array = []
#     for shared_list in shared_lists:
#         # for shared in shared_list:
#         #     shared_array.append(shared)
#         shared_array.append(shared_list)
#     lists = np.array(shared_array)
#
#     mins = lists.min(0)
#     maxs = lists.max(0)
#     ranges = maxs - mins
#     normList = np.zeros(np.shape(lists))
#     row = lists.shape[0]
#     normList = lists - np.tile(mins, (row, 1))
#     normList = normList / np.tile(ranges, (row, 1))
#
#     return normList
#     # return (lists - np.mean(lists)) / np.std(lists)
#
#
# def random_index(random_range, random_num=0):
#     index = [i for i in range(0, random_range)]
#     random.shuffle(index)
#     index = index[:random_num]
#     return index
#
#
# def visualization_loss(loss, loss_num, filename):
#     x_axis = range(0, loss_num + 1, 8)
#     plt.figure(figsize=(12, 12))
#     plt.plot(loss, color='red', label=r'$\mathcal{L}_{inv}$', linewidth=5)
#     plt.yticks(fontsize=45)
#     plt.xticks(x_axis, fontsize=45)
#     # plt.xlabel('epoch', fontsize=45)
#     plt.rcParams.update({'font.size': 50})
#     plt.legend()
#     plt.savefig(filename)
#     plt.clf()
#
#
# def draw_loss():
#     recorder_loss = LossRecorder((r'result_loss.tsv'), total_cv=1, total_epoch=40)
#     loss_mean_list = recorder_loss.read_result_from_tsv()
#     visualization_loss(loss_mean_list, 40, 'common_loss.jpg')
#
#
# def visualization_consistency(key_list, data, filename):
#     color = ['g', 'b', 'c', 'r', 'm', 'y']
#     plt.figure(figsize=(12, 12))
#     p = []
#
#     for i, key in enumerate(key_list):
#         # plt.scatter(x=data[key][:, 0], y=data[key][:, 1], c=color[i], marker='o', s=90, label=translate(key))
#         p.append(plt.scatter(x=data[key][:, 0], y=data[key][:, 1], c=color[i], marker='o', s=90))
#
#     plt.yticks(np.arange(0, 0.5 + 1, 0.5), fontsize=45)
#     plt.xticks(np.arange(0, 0.5 + 1, 0.5), fontsize=45)
#     plt.rcParams.update({'font.size': 25})
#
#     # plt.legend(loc='best')
#     a = plt.legend(p[:(len(p) // 2)], [translate(key) for key in key_list[:(len(p) // 2)]], loc=1)
#     plt.legend(p[(len(p) // 2):], [translate(key) for key in key_list[(len(p) // 2):]], loc=2)
#     plt.gca().add_artist(a)
#
#     plt.savefig(filename)
#     plt.clf()
#
#
# def translate(key):
#     if key == 'azz':
#         return '$H\'${a}'
#     elif key == 'avz':
#         return '$H\'${a,v}'
#     elif key == 'azl':
#         return '$H\'${a,t}'
#     elif key == 'zvz':
#         return '$H\'${v}'
#     elif key == 'zvl':
#         return '$H\'${v,t}'
#     elif key == 'zzl':
#         return '$H\'${t}'
#     if key == '0':
#         return 'happy'
#     if key == '1':
#         return 'angry'
#     if key == '2':
#         return 'sad'
#     if key == '3':
#         return 'neutral'
#
#
# def draw_consistency_feature(path, num_list):
#     consistent_dict_condition = {
#         "azz": [],
#         "zvz": [],
#         "zzl": [],
#         "avz": [],
#         "azl": [],
#         "zvl": [],
#     }
#     num_len = len(num_list)
#     part_names = ['azz', 'avz', 'azl', 'zvl', 'zvz', 'zzl']
#
#     consistent_feat_list = read_pkl(path=path, num_list=num_list[:num_len], basename='consistent_feat.pkl')
#     miss_type_list = read_pkl(path=path, num_list=num_list[:num_len], basename='miss_type.pkl')
#
#     consistent_feats = []
#     miss_types = []
#
#     for i in range(0, num_len):
#         for item in consistent_feat_list[i].cpu().detach().numpy():
#             consistent_feats.append(item)
#         for item in miss_type_list[i]:
#             miss_types.append(item)
#
#     consistent_feats = np.array(consistent_feats)
#     miss_types = np.array(miss_types)
#
#     consistent_feats = TSNE_model.fit_transform(consistent_feats)
#     for part_name in part_names:
#         index = np.where(miss_types == part_name)
#         consistent_dict_condition[part_name] = consistent_feats[index]
#     for part_name in part_names:
#         # normalization
#         consistent_dict_condition[part_name] = build_array(consistent_dict_condition[part_name])
#
#         # remove points which are over limited
#         if part_name in ['avz', 'zvl', 'zvz']:
#             p = 0
#             for i, item in enumerate(consistent_dict_condition[part_name]):
#                 if (part_name == 'avz' and item[1] > 0.5) or (
#                         part_name == 'zvl' and item[0] > 0.5):  # or (part_name == 'zvz' and item[1] < 0.5):
#                     consistent_dict_condition[part_name] = np.delete(consistent_dict_condition[part_name], i - p, 0)
#                     p += 1
#
#         # take 100 points for each condition
#         length = len(consistent_dict_condition[part_name]) if len(consistent_dict_condition[part_name]) < 100 else 100
#         consistent_dict_condition[part_name] = consistent_dict_condition[part_name][[i for i in range(0, length)]]
#         # print(part_name, consistent_dict_condition[part_name].size // 2)
#
#     visualization_consistency(key_list=part_names, data=consistent_dict_condition, filename='consistent_feature.jpg')
#
#
# def draw_consistency_label(path, num_list):
#     consistent_dict_lable = {
#         "0": [],
#         "1": [],
#         "2": [],
#         "3": []
#     }
#     # # num_list = ['8', '10', '18', '19', '20']
#     # num_list = ['10', '18', '19', '20']
#     num_len = len(num_list)
#     part_names = ['0', '1', '2', '3']
#
#     consistent_feat_list = read_pkl(path=path, num_list=num_list[:num_len], basename='consistent_feat.pkl')
#     label_list = read_pkl(path=path, num_list=num_list[:num_len], basename='label.pkl')
#
#     consistent_feats = []
#     labels = []
#
#     for i in range(0, num_len):
#         for item in consistent_feat_list[i].cpu().detach().numpy():
#             consistent_feats.append(item)
#         for item in label_list[i].cpu().detach().numpy():
#             labels.append(str(item))
#
#     consistent_feats = np.array(consistent_feats)
#     labels = np.array(labels)
#
#     consistent_feats = TSNE_model.fit_transform(consistent_feats)
#     for part_name in part_names:
#         index = np.where(labels == part_name)
#         consistent_dict_lable[part_name] = consistent_feats[index]
#         # consistent_dict_lable[part_name] = consistent_dict_lable[part_name][[i for i in range(0, 10)]]
#     for part_name in part_names:
#         consistent_dict_lable[part_name] = build_array(consistent_dict_lable[part_name])
#     visualization_consistency(key_list=part_names, data=consistent_dict_lable, filename='consistent_label.jpg')
#
#

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FormatStrFormatter

from Domiss import NoiseScheduler
import numpy as np
from sklearn.manifold import TSNE
import pickle
import json





def collect_and_noise_feats(miss, origin):
    device = 'cuda:6'
    # noise_scheduler_100 = NoiseScheduler(noise_type='Gaussian', num_time_steps=100)
    # noise_scheduler_80 = NoiseScheduler(noise_type='Gaussian', num_time_steps=80)
    # noise_scheduler_60 = NoiseScheduler(noise_type='Gaussian', num_time_steps=60)
    # noise_scheduler_40 = NoiseScheduler(noise_type='Gaussian', num_time_steps=40)
    # noise_scheduler_20 = NoiseScheduler(noise_type='Gaussian', num_time_steps=20)
    print('begin to draw.')
    tsne = TSNE(n_components=3, random_state=58, perplexity=3, n_iter=1000)
    # acoustic = input['A_feat'].float()
    # lexical = input['L_feat'].float()
    # visual = input['V_feat'].float()

    # a_list = [acoustic[i] for i in range(0, 30)]
    # v_list = [visual[i] for i in range(0, 30)]
    # l_list = [lexical[i] for i in range(0, 30)]
    a = origin[0]
    l = origin[1]
    v = origin[2]
    a_miss = miss[0]
    l_miss = miss[1]
    v_miss = miss[2]
    a = a.cpu().detach().numpy()
    v = v.cpu().detach().numpy()
    l = l.cpu().detach().numpy()
    a_miss = a_miss.cpu().detach().numpy()
    v_miss = v_miss.cpu().detach().numpy()
    l_miss = l_miss.cpu().detach().numpy()



    a = tsne.fit_transform(a)
    v = tsne.fit_transform(v)
    l = tsne.fit_transform(l)
    a_miss = tsne.fit_transform(a_miss)
    v_miss = tsne.fit_transform(v_miss)
    l_miss = tsne.fit_transform(l_miss)

    plt.figure()
    #
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)

    # plt.scatter(a[:, 0], a[:, 1], c='blue', label='a', s=15)
    plt.scatter(v[:, 0], v[:, 1], c='black', label='v', s=15)
    # plt.scatter(l[:, 0], l[:, 1], c='red', label='l', s=15)

    # plt.scatter(a_miss[:, 0], a_miss[:, 1], c='yellow', label='a_noisy', s=15)
    plt.scatter(v_miss[:, 0], v_miss[:, 1], c='green', label='v_noisy', s=15)
    # plt.scatter(l_miss[:, 0], l_miss[:, 1], c='#00CED1', label='l_noisy', s=15)
    # plt.scatter(l[:, 0], l[:, 1], c='black', label='l', s=15)


    # 添加图例
    plt.legend()

    import time

    plt.savefig(f'./fig/fig{time.time()}.png')
    plt.show()
    print('OK')

if __name__ == '__main__':
#     import re
#
# # 定义正则表达式模式
# pattern = r"Cur epoch (\d+).*vae:([\d.]+)"
#
# # 跟踪已计数过的数字
# counted_nums = {}
#
# # 打开日志文件
# with open(r"C:\Users\Administrator\Desktop\20.log", "r") as file:
#     # 逐行读取文件内容
#     for line in file:
#         # 在每一行中匹配正则表达式
#         match = re.search(pattern, line)
#         if match:
#             # 获取匹配到的数字
#             cur_epoch = int(match.group(1))
#             vae_num = float(match.group(2))
#
#             # 如果cur_epoch已经计数过，则跳过这一行
#             if cur_epoch in counted_nums:
#                 continue
#
#             # 记录当前cur_epoch的计数
#             counted_nums[cur_epoch] = True
#
#             # 输出并保存结果
#             result = f"{cur_epoch},{vae_num}"
#             # print(result)
#             with open(r"C:\Users\Administrator\Desktop\20out.log", "a") as output_file:
#                 output_file.write(result + "\n")

    # x = [20, 40, 60, 80, 100]
    # plt.xticks([20, 40, 60, 80, 100], fontsize=18)  # x 轴数据
    # plt.yticks(fontsize=14)
    #
    # # men = [0.5949, 0.5663, 0.5563, 0.5458, 0.5325]
    # mmin = [0.7031, 0.6791, 0.6711, 0.6630, 0.6615]
    # ifmmin = [0.6977, 0.6812, 0.6703, 0.6702, 0.6641]
    # mrcn = [0.7060, 0.6831, 0.6773, 0.6706, 0.6673]
    #
    # # tick_spacing = 0.8
    # # plt.xticks(x, labels=x[::int(1 / tick_spacing)])
    # # plt.plot(x, men, label='MEN')  # 绘制第一条线，并设置 label 参数
    # plt.plot(x, mmin, label='MMIN', linewidth=3.5, color='blue')  # 绘制第二条线，并设置 label 参数
    # plt.plot(x, ifmmin, label='IF-MMIN', linewidth=3.5, color='black')
    # plt.plot(x, mrcn, label='NMER(ours)', linewidth=3.5, color='red')
    #
    # # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # plt.subplots_adjust(top=0.98, right=0.98, bottom=0.16, left=0.18)
    # plt.xlabel("Noise Intensity",fontsize=22, labelpad=1)  # 添加 X 轴标签
    # plt.ylabel("Weighted Accuracy", fontsize=22, labelpad=1)  # 添加 Y 轴标签
    # plt.yticks(size=12, fontsize=22)
    # plt.xticks(size=12, fontsize=22)
    # plt.legend(fontsize=22)  # 添加图例，使用默认的位置和标签
    # plt.savefig('D:/WA.png', dpi=600)  # 保存图表，并指定 DPI 为 300
    # plt.show()  # 显示图表


    #
    import numpy as np
    import matplotlib.pyplot as plt

    # 读取 log 文件
    log_file100 = r'C:\Users\Administrator\Desktop\100out.log'
    log_file80 = r'C:\Users\Administrator\Desktop\80out.log'
    log_file60 = r'C:\Users\Administrator\Desktop\60out.log'
    log_file40 = r'C:\Users\Administrator\Desktop\40out.log'
    log_file20 = r'C:\Users\Administrator\Desktop\20out.log'
    data100 = np.genfromtxt(log_file100, delimiter=',')
    data80 = np.genfromtxt(log_file80, delimiter=',')
    data60 = np.genfromtxt(log_file60, delimiter=',')
    data40 = np.genfromtxt(log_file40, delimiter=',')
    data20 = np.genfromtxt(log_file20, delimiter=',')

    # 创建折线图
    plt.subplots_adjust(top=0.98, right=0.98)
    plt.plot(data100[:, 0], data100[:, 1], label='Intensity 100', color='blue', linewidth=2.5)
    plt.plot(data80[:, 0], data80[:, 1], label='Intensity 80', color='black', linewidth=2.5)
    plt.plot(data60[:, 0], data60[:, 1], label='Intensity 60', color='red', linewidth=2.5)
    plt.plot(data40[:, 0], data40[:, 1], label='Intensity 40', color='pink', linewidth=2.5)
    plt.plot(data20[:, 0], data20[:, 1], label='Intensity 20', color='green', linewidth=2.5)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    #
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16, left=0.18)
    plt.xlabel('Epoch', fontsize=22, labelpad=0)
    plt.ylabel('VAE Loss', fontsize=22, labelpad=-3)
    plt.legend(fontsize=22)
    plt.savefig('D:/vae_loss_1.png', dpi=600)
    plt.show()