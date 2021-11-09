import os
import torch

def get_all_filenames(root='/home/kangsc/wiping_vae_dataset/line_vae/'):
    filenames = []
    cnt = 0
    for file in os.listdir(root):
        if '.jpg' in file:
            filenames.append(root + file)
            cnt += 1
    print(f'There are {cnt} pictures in {root}.')
    return filenames, cnt

def divide_filenames(cnt, p_train = 0.8, p_val = 0.1, p_test = 0.1):
    all_file = open('./dataset_divide/all.txt', 'r')
    train_file = open('./dataset_divide/train.txt', 'w')
    val_file = open('./dataset_divide/val.txt', 'w')
    test_file = open('./dataset_divide/test.txt', 'w')
    weight = torch.tensor([p_train, p_val, p_test])
    random_nums = torch.multinomial(weight, cnt, replacement=True)
    for random_num, filename in zip(random_nums, all_file.readlines()):
        if random_num == 0:
            train_file.write(filename)
        elif random_num == 1:
            val_file.write(filename)
        elif random_num == 2:
            test_file.write(filename)
        else:
            raise ValueError('No such random number!')
    test_file.close()
    val_file.close()
    train_file.close()
    all_file.close()    
    
if __name__ == '__main__':
    filenames, cnt = get_all_filenames()
    file = open('./dataset_divide/all.txt', 'w')
    for tmp in filenames:
        file.write(tmp + '\n')
    file.close()
    divide_filenames(cnt)
    