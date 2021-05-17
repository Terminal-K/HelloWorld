'''
    将数据集和对应的标签放进txt里，并进行划分
'''

# --------------------------------准备数据格式(将csv里的标签转化到txt) --------------------------------
import os
import random

data_dir = './'
label_file = 'label/csvfile/train.csv'
test_file = 'label/csvfile/test.csv'

def generate_txt(data_dir, label_file, test_file):
    with open('label/csvfile/CuisineList.txt', 'r') as txt:
        with open(os.path.join(data_dir, label_file), 'r') as f:
            lines = f.readlines()[1:]
            for i,per_label in enumerate(txt):
                datas = lines
                listText = open('all.txt', 'a+')
                for l in datas:
                    tokens = l.rstrip().split(',')
                    idx, label = tokens
                    if label == per_label[:-1]:
                        name = './dataset/train/' + idx + ' ' + str(i) + '\n'
                        listText.write(name)

            listText.close()

        with open(os.path.join(data_dir, test_file), 'r') as f1:
            lines1 = f1.readlines()[1:]  # 表示从第1行，下标为0的数据行开始

            print('input :', os.path.join(data_dir, test_file))
            print('start...')
            listText1 = open('test.txt', 'a+')  # 创建并打开test1.txt文件，a+表示打开一个文件并追加内容
            for l1 in lines1:
                name1 = './dataset/test/' + l1.rstrip()  + '\n'  # rstrip()为了把右边的换行符删掉
                listText1.write(name1)
            listText1.close()
            print('down!')
generate_txt(data_dir, label_file, test_file)

#  --------------------------------划分数据集--------------------------------
div = 0.05  #(可调)训练集:验证集 = 19:1     div = 0.05 = 1 / (19+1)
val_list = set()
all_list = set()
train_list = set()

with open('all.txt','r') as all_txt:
    with open('train.txt', 'w') as train_txt:
        with open('val.txt', 'w') as val_txt:
            index = 0
            lines = all_txt.readlines()
            for line in lines:
                all_list.add(line)

            while(len(val_list) < div*len(lines)):
                i = random.randint(0,len(lines) - 1)
                val_list.add(lines[i])
                index += 1

            for line in val_list:
                val_txt.write(line)

            train_list = all_list - val_list
            for train in train_list:
                train_txt.write(train)
print("训练集与验证集划分完成！")