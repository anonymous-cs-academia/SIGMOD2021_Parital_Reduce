import numpy as np
import pickle
import os
import sys
import random
CIFAR_DIR = "./data/cifar-10-batches-py"
splited_CIFAR_DIR = "./data/splited_cifar"
# print(os.listdir(CIFAR_DIR))
file_list = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5"
]
# dict_keys(['batch_label', 'labels', 'data', 'filenames'])
# type        str,           list,     numpy.ndarray  list
# with open(os.path.join(CIFAR_DIR, "data_batch_1"), 'rb') as f:
#     if sys.version_info[0] == 2:
#         data = pickle.load(f)
#     else:
#         data = pickle.load(f, encoding='latin1')
#     print(type(data))
#     print(data.keys())
#     print(type(data['data']), data['data'].shape)
#     print(type(data['labels']), len(data['labels']))
#     print(type(data['batch_label']), len(data['batch_label']))
    
#     print(type(data['filenames']), len(data['filenames']))
#     print(data['data'].shape)
#     print(data['data'][2:4])
#     print(data['batch_label'])
#     print(data['filenames'][2:4])

if __name__ == "__main__":
    data = []
    label = []
    file_names = []
    for file_path in file_list:
        with open(os.path.join(CIFAR_DIR, file_path), 'rb') as f:
            if sys.version_info[0] == 2:
                input = pickle.load(f)
            else:
                input = pickle.load(f, encoding='latin1')
            data.append(input['data'])
            label.extend(input['labels'])
            file_names.extend(input['filenames'])
    # print(len(data), type(data[0]), data[0].shape)
    # print(len(label))
    data = np.vstack(data)
    shuffle_list = [i for i in range(len(label))]
    random.shuffle(shuffle_list)

    shuffled_data = np.empty_like(data)
    shuffled_label = []
    shuffled_filenames = []
    # print(type(label))
    for i in range(len(label)):
        idx = shuffle_list[i]
        shuffled_data[i] = data[idx]
        shuffled_label.append(label[idx])
        shuffled_filenames.append(file_names[idx])
    print(len(shuffled_label))
    print(len(shuffled_filenames))

    for k_split in range(2, 9):
        split_length = len(label) // k_split
        split_folder = splited_CIFAR_DIR + '/worker_num_' + str(k_split) + '/'

        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        
        for idx in range(k_split):
            output_dict = dict()
            file_name = split_folder + 'worker_id_' + str(idx)
            output_file = open(file_name, 'wb')

            start_index = idx * split_length
            end_index = (idx + 1) * split_length
            output_dict['data'] = shuffled_data[start_index : end_index]
            output_dict['labels'] = shuffled_label[start_index : end_index]
            output_dict['filenames'] = shuffled_filenames[start_index : end_index]

            pickle.dump(output_dict, output_file)
            output_file.close()
        
    