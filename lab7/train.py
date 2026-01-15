# add your source codes regarding the model training and evaluation here
import torch as pt
from data import load, ProteinDataset
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from model import ProteinGCN
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


data_ = load('../../cath/hdf5/struct256.hdf5')
dataset = ProteinDataset(data_)
datamap=pt.arange(len(dataset),dtype=pt.int32) 
batchsize = 256
# all: 21,5799
trainmap,validmap=train_test_split(datamap,test_size=20480,random_state=7)
validmap, testmap=train_test_split(validmap,test_size=10240,random_state=7) 
trainset=ProteinDataset(dataset,trainmap)
validset=ProteinDataset(dataset,validmap)
testset=ProteinDataset(dataset,testmap)

trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=6)
validloader = DataLoader(validset, batch_size=batchsize, shuffle=False, drop_last=False, num_workers=6)
testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, drop_last=False, num_workers=6)

prot_model = ProteinGCN(in_channels=21, batch_size=batchsize).cuda()
criterion = pt.nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = pt.optim.AdamW(prot_model.parameters(), lr=learning_rate)
num_epochs = 20
logging.basicConfig(level=logging.INFO)


def main():
    for epoch in range(num_epochs):
        train_loss = []
        prot_model.train()
        for g in trainloader:
            g = g.cuda()
            lab = g.y.view(batchsize, -1)
            output = prot_model(g)
            loss = criterion(output, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss)
        # scheduler.step()

        valid_acc = []
        prot_model.eval()  # 设置为评估模式，不使用dropout等技巧 
        with pt.no_grad():  # 不计算梯度，节省内存
            for g in validloader:
                g = g.cuda()
                lab = g.y.view(batchsize, -1)
                output = prot_model(g)  # [batchsize, num_class]
                # [batchsize, 1]
                output = pt.argmax(output, axis=1, keepdim=True)
                acc = float(pt.sum(output == pt.argmax(lab, axis=1, keepdim=True)))/batchsize
                valid_acc.append(acc)

        logging.info("epoch: %d\n train loss: %.4f \n valid acc: %.2f%%" % (epoch+1,
                        pt.mean(pt.tensor(train_loss)).detach().numpy(),
                        pt.mean(pt.tensor(valid_acc)).detach().numpy()*100))

    test_acc = []
    prot_model.eval()
    with pt.no_grad():
        for g in testloader:
            g = g.cuda()
            lab = g.y.view(batchsize, -1)
            output = prot_model(g)
            output = pt.argmax(output, axis=1, keepdim=True)
            acc = float(pt.sum(output == pt.argmax(lab, axis=1, keepdim=True)))/batchsize
            test_acc.append(acc)
        logging.info("test acc: %.2f%%" % (pt.mean(pt.tensor(valid_acc)).detach().numpy()*100))
        # 保存模型参数
        # pt.save(prot_model.state_dict(), 'model'+str(epoch)+'.ckpt')


if __name__ == '__main__':
    import time
    logging.info(time.strftime("%Y-%m-%d %H:%M:%S",
                 time.localtime(time.time())))
    main()
    logging.info(time.strftime("%Y-%m-%d %H:%M:%S",
                 time.localtime(time.time())))


# ps -aux | grep burger
