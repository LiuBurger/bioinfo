# add your source codes regarding the model training and evaluation here
import numpy as np
from data import load, ProteinDataset, collate_fn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import Convformer
import torch as pt
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


data_ = load('../cath/hdf5/seq1024.hdf5')
dataset = ProteinDataset(data_)
datamap = np.arange(len(dataset), dtype=np.int64)  # 32不够
trainmap, validmap = train_test_split(datamap, test_size=1024*75, random_state=7)
trainset = ProteinDataset(dataset, trainmap)
validset = ProteinDataset(dataset, validmap)
batchsize = 1024
trainloader = DataLoader(trainset, batch_size=batchsize,
                         shuffle=True, drop_last=True, num_workers=6, collate_fn=collate_fn)
validloader = DataLoader(validset, batch_size=batchsize,
                         shuffle=False, drop_last=False, num_workers=6, collate_fn=collate_fn)

prot_model = Convformer().cuda()
criterion = pt.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = pt.optim.Adam(prot_model.parameters(), lr=learning_rate)
num_epochs = 10
logging.basicConfig(level=logging.INFO)


def main():
    for epoch in range(num_epochs):
        prot_model.train()
        running_loss = []
        for seq, ptr, lab in trainloader:
            lab = pt.from_numpy(lab).float().cuda() 
            output = prot_model((seq, ptr))
            loss = criterion(output, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss)

        prot_model.eval()  # 设置为评估模式，不使用dropout等技巧
        running_acc = []
        with pt.no_grad():  # 不计算梯度，节省内存
            for seq, ptr, lab in validloader:
                lab = pt.from_numpy(lab).float().cuda()
                output = prot_model((seq, ptr))  # [batchsize, num_class]
                # [batchsize, 1]
                label = pt.argmax(output, axis=1, keepdim=True)
                acc = float(pt.sum(label == pt.argmax(
                    lab, axis=1, keepdim=True)))/batchsize
                running_acc.append(acc)

        logging.info("epoch: %d\n loss: %.4f \n acc: %.2f%%"
                     % (epoch+1,
                        pt.mean(pt.tensor(running_loss)).detach().numpy(),
                        pt.mean(pt.tensor(running_acc)).detach().numpy()*100))

        # 保存模型参数
        # pt.save(prot_model.state_dict(), 'model'+str(epoch)+'.ckpt')


if __name__ == '__main__':
    import time
    logging.info(time.strftime("%Y-%m-%d %H:%M:%S",
                 time.localtime(time.time())))
    main()
    logging.info(time.strftime("%Y-%m-%d %H:%M:%S",
                 time.localtime(time.time())))


# ps -aux | grep Burger
