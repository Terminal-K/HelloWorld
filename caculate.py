'''
    计算训练集各类的数量
'''

from tqdm import tqdm
label_dic = {}

num = []
with open('./label/csvfile/CuisineList.txt') as f:
    with open('./train.txt','r') as txt:
        for i, per_laebl in enumerate(f.readlines()):
            label_dic[str(i)] = per_laebl[:-1]

        lines = txt.readlines()
        for line in lines:
            nums = (line.split())[1]
            num.append(nums)

total_num = [0 for i in range(172)]

with tqdm(total = 172) as pbar:
    with open('./total.txt','a+') as total:
        for i in range(172):
            for x in num:
                if x == str(i):
                    total_num[i] += 1
            total.write('第{}类数量为:'.format(i) + str(total_num[i]) + '\n')
            pbar.update(1)

x = 0
for i in range(len(total_num)):
    x += int(total_num[i])
print('图片总数量为:{}'.format(x))

total.close()