from dataloader import train_iter, test_iter, train_loader, val_loader, test_loader
from model import LossFunction,optimizer, model, scheduler
import torch
import pandas as pd
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

# GPU
torch.cuda.set_device(0)
gpu_index = torch.cuda.current_device()

epochs = 100
scale = train_iter

# 日志文件存放地址
writer = SummaryWriter('./resnet50_summary')

#--------------------------------定义训练函数--------------------------------
def train(epoch):
    running_loss = 0
    correct = 0
    total = 0

    for iter, datas in enumerate(train_loader,0):
        data, label = datas
        data = data.cuda(gpu_index)
        label = label.cuda(gpu_index)

        pred = model(data)

        optimizer.zero_grad()
        loss = LossFunction(pred,label)
        loss.backward()
        running_loss += loss

        _, predicted = torch.max(pred, dim=1)

        correct += (predicted == label).sum().item()
        total += label.size(0)

        optimizer.step()
        scheduler.step()

        # 进度条
        a = "*" * int(iter/80)
        b = "." * int((scale - iter)/80)
        c = (iter / scale) * 100

        acc = correct / total
        avg_loss = running_loss / iter

        writer.add_scalar('Loss/train', avg_loss, epoch*scale + iter)
        writer.add_scalar('Accuracy/train', acc, epoch*scale + iter)
        writer.add_scalar('lr/train', optimizer.state_dict()['param_groups'][0]['lr'], epoch*scale + iter)

        print("time:{}, epoch:{}, iter:{}, average_loss:{}, acc:{:.5f}, lr:{} {:^3.0f}%[{}->{}]".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                                                                                                    epoch + 1,
                                                                                                    iter,
                                                                                                    avg_loss,
                                                                                                    acc,
                                                                                                    optimizer.state_dict()['param_groups'][0]['lr'],    #打印学习率
                                                                                                    c,
                                                                                                    a,
                                                                                                    b),
              end="")
        print('\n')
        time.sleep(0.1)

#--------------------------------定义验证函数--------------------------------
def val(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for iter, datas in enumerate(val_loader,0):
            data, label = datas
            data = data.cuda(gpu_index)
            label = label.cuda(gpu_index)

            pred = model(data)

            _,predicted = torch.max(pred, dim=1)

            correct += (predicted == label).sum().item()
            total += label.size(0)

        acc = correct / total
        writer.add_scalar('Accuracy/val', acc, epoch)

    return correct, total

#--------------------------------定义测试函数--------------------------------
def test():
    results = list()
    label_dic = {}

    # 标签地址
    label_path = './label/csvfile/'

    test_csv = os.path.join(label_path, 'test.csv')
    with open(os.path.join(label_path,'CuisineList.txt')) as f:
        for i, per_laebl in enumerate(f.readlines()):
            label_dic[str(i)] = per_laebl[:-1]
    with torch.no_grad():
        with tqdm(total = test_iter) as pbar:
            for datas in test_loader:
                data = datas
                data = data.cuda(gpu_index)

                pred = model(data)
                _,predicted = torch.max(pred, dim=1)
                result = predicted.cpu().numpy().tolist()
                results += label_dic[result]

                pbar.update(1)

            testfile = pd.read_csv(test_csv)
            id = testfile['ID']
            submission = pd.DataFrame({'ID': id, 'label': results})
            submission.to_csv('./submission.csv', index=False)


# #--------------------------------加载模型--------------------------------
def load_model():
    checkpoint = torch.load('./models/model_0.7722140402552774.pth')
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# 开始炼丹
def main(mode, keepon):
    if keepon == True:      # 加载模型
        load_model()

    if mode == 'train':     # 模式选择:训练
        best_result = 0
        for epoch in range(epochs):
            train(epoch)

            if (epoch + 1) % 1 == 0:
                print("Start to predict:\n")
                correct, total = val(epoch)
                result = correct / total
                print("The accurary of test datasets is :{}[correct:{},total:{}]".format(result, correct, total))

                if result > best_result:
                    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                    torch.save(state, './models/model_{}.pth'.format(result))
                    print("\nThe model with best accurary {} in test datasets was saved\n".format(result))
                    best_result = result

    elif mode == 'test':    # 模式选择:预测
        test()

main(mode = 'train', keepon=False)
