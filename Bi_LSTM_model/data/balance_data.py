# -*-coding:utf-8-*-
# 创建时间：2021/10/30 12:53
import os


tr_pos_len = 34999
test_pos_len = 8999

def get_origin_path():
    train_x_path = 'train_x.txt'
    train_y_path = 'train_y.txt'
    test_x_path = 'test_x.txt'
    test_y_path = 'test_y.txt'
    return train_x_path, train_y_path, test_x_path, test_y_path


def get_save_path():
    train_x_path = 'train_x_new.txt'
    train_y_path = 'train_y_new.txt'
    test_x_path = 'test_x_new.txt'
    test_y_path = 'test_y_new.txt'
    return train_x_path, train_y_path, test_x_path, test_y_path


def mk_new_data(label_file, comfile, new_lable_file, new_comfile, train = True):
    with open(label_file, 'r', encoding='utf-8') as reader1, open(new_lable_file, 'w', encoding='utf-8') as save1,\
         open(comfile, 'r', encoding='utf-8') as reader2, open(new_comfile, 'w', encoding='utf-8') as save2:
        pos_count = 0
        neg_count = 0
        if train:
            data_len = tr_pos_len
        else:
            data_len = test_pos_len
        labels = reader1.readlines()
        lines = reader2.readlines()
        for label, line in zip(labels, lines):
            if int(label[0]) == 0 and neg_count <= data_len:
                save1.write(label)
                save2.write(line)
                neg_count += 1
            elif int(label[0]) == 1 and pos_count <= data_len:
                save1.write(label)
                save2.write(line)
                pos_count += 1


if __name__ == '__main__':
    tr_x, tr_y, te_x, te_y  = get_origin_path()
    tr_x1, tr_y1, te_x1, te_y1 = get_save_path()
    mk_new_data(tr_y, tr_x, tr_y1, tr_x1, train=True)
    mk_new_data(te_y, te_x, te_y1, te_x1, train=False)
