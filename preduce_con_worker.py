'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import socket
import time

import os
import argparse
import sys
from models import *
from logger import Logger
from asyncDataSet import MyTrainDataset

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
time_stamp = 0

def prepare_data(dir = None, train_batch_size = 128, num_of_worker = -1, worker_id = -1):
    # Data
    mylogger.info("==> Preparing data..")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # trainset = torchvision.datasets.CIFAR10(root=dir, train=True, download=True, transform=transform_train)
    data_path = dir + '/splited_cifar/worker_num_' + str(num_of_worker) + '/worker_id_' + str(worker_id)
    mylogger.info("Loading dataset in {}".format(data_path))
    trainset = MyTrainDataset(path = data_path, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=3)

    testset = torchvision.datasets.CIFAR10(root=dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=3)

    return trainloader, testloader

def update_model(net = None, model_recv_buffer = None):
    """write model fetched from parameter server to local model"""
    new_state_dict = {}
    model_counter_ = 0
    for param_idx,(key_name, param) in enumerate(net.state_dict().items()):
        # handle the case that `running_mean` and `running_var` contained in `BatchNorm` layer
        if "running_mean" in key_name or "running_var" in key_name or "num_batches_tracked" in key_name:
            tmp_dict={key_name: param}
        else:
            assert param.size() == model_recv_buffer[model_counter_].size()
            tmp_dict = {key_name: model_recv_buffer[model_counter_]}
            model_counter_ += 1
        new_state_dict.update(tmp_dict)
    net.load_state_dict(new_state_dict)    

# Training
def train(comm = None, epoch = 0, myrank = -1, net = None, criterion = None, optimizer = None, trainloader = None):
    # print('\nEpoch: %d' % epoch)
    global time_stamp
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    cnt = 0


    updates = torch.zeros(size = [1], dtype = torch.int)
    updates_gpu = torch.zeros(size = [1], dtype = torch.int, device = device)
    model_buffer = []
    gpu_model_buffer = []
    for param in net.parameters():
        model_buffer.append(torch.zeros(param.size()))
        gpu_model_buffer.append(torch.zeros(param.size(), device = device))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # mylogger.info("batch id %d" %(batch_idx))
        # mylogger.info("time stamp for worker %d is %d" %(myrank, time_stamp)) 
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        time_stamp = time_stamp + 1

        # mylogger.info("send ready signal") 
        ''' send ready signal '''
        ready_signal = '-1' + str(myrank) + "%06d"%(time_stamp)
        ready_signal = ready_signal.encode('utf-8')
        comm.send(ready_signal)
        
        # mylogger.info("get group") 
        ''' get group '''
        group_signal = comm.recv(20)
        group_signal = group_signal.decode('utf-8')
        kgpu = int(group_signal[0])
        src_rank = int(group_signal[1])
        max_time_stamp_info = int(group_signal[kgpu+1:kgpu+7])
        # mylogger.info("group signal"+group_signal)
        # mylogger.info("divide %d" %(max_time_stamp_info-time_stamp+1))
        # if not max_time_stamp_info == time_stamp:
            # decay = float(max_time_stamp_info-time_stamp+1)
            # for param in net.parameters():
                # param.grad = param.grad / decay
        
        # get group
        if kgpu != 1:
            group_hash_id = 0
            for i in range(kgpu):
                group_hash_id = group_hash_id + (int)(2**(int(group_signal[i+1])))
            allreduce_group = group_map[group_hash_id]

        # mylogger.info("allreduce ok %d" %(myrank))
        if(myrank == src_rank):
            # reduce gradient
            if kgpu != 1:
                for param in net.parameters():
                    if param.requires_grad:
                        dist.all_reduce(param.data, group = allreduce_group)
        else:
            for param in net.parameters():
                if param.requires_grad:
                    dist.all_reduce(param.data, group = allreduce_group)            
        # mylogger.info("after allreduce")
        for param in net.parameters():
            if param.requires_grad:
                param.data = param.data / kgpu
        cnt = cnt + 1
        train_loss += loss.item()
        # mylogger.info("loss: %.3f" %(loss.item()))
        # mylogger.info("after get loss")
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # mylogger.info("after loss")
    
    updates.copy_(updates_gpu)
    mylogger.info("Update Num: %d" %(updates.item()))
    mylogger.info("Training Loss: %.3f | Acc: %.3f%%" %(train_loss/(cnt), 100.*correct/total))
        


def test(net = None, criterion = None, optimizer = None, testloader = None, checkpoint_name = None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    cnt = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            cnt = cnt + 1
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    mylogger.info("Testing Loss: %.3f | Acc: %.3f%%" %(test_loss/(cnt), 100.*correct/total))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+ " Testing Loss: %.3f | Acc: %.3f%%" %(test_loss/(cnt), 100.*correct/total))
    # Save checkpoint.
    acc = 100.*correct/total
    '''
    if acc > best_acc:
        # print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + checkpoint_name)
        best_acc = acc
     '''

def adjust_learning_rate(optimizer, epoch, args):
    lr_decay = 0.1
    lr_decay_freq = 130
    lr = args.lr * (lr_decay ** (epoch // lr_decay_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":

    global mylogger
    global device
    global group_map

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--rank', '-r', default = -1, type = int)
    parser.add_argument('--partial-reduce-num', '-k', default = -1, type = int)
    parser.add_argument('--world-size', '-s', default = -1, type = int)
    parser.add_argument('--gpu-id', '-g', default = -1, type = int)
    # parser.add_argument('--check_point', '-c', default = None, type = str)
    parser.add_argument('--resume', '-resume', action='store_true', help='resume from checkpoint')
    
    args = parser.parse_args()
    args.lr = args.lr / args.partial_reduce_num
    log_dir = "./combine"
    log_name = "/worker_" + str(args.gpu_id) + ".log"
    checkpoint_name = 'partial_reduce_num_' + str(args.partial_reduce_num) + "/worker_" + str(args.rank - 1) + '.pth'
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # elif os.path.exists(log_dir + log_name):
    #     print("Log file has already exist!")
    #     sys.exit(1)
    # init logging
    L = Logger()
    L.set_log_name(log_dir + log_name)
    mylogger = L.get_logger()

    # init device
    if args.gpu_id is not -1:
        device = torch.device('cuda', args.gpu_id)
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    
    mylogger.info("Worker %d of %d"%(args.rank - 1, args.world_size - 1))
    # init Model
    mylogger.info("==> Building ResNet-34 model..")
    # net = VGG('VGG19')
    net = ResNet34()
    # net = DenseNet121()

    # init loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    mylogger.info("==> Init socket.")
    tcp_sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    HOST = socket.gethostname()
    PORT = 10000

    time.sleep(2)
    mylogger.info("==> Connect Server.")
    tcp_sender.connect((HOST, PORT))
    worker_id_str = str(args.rank)
    tcp_sender.send(worker_id_str.encode('utf-8'))   
    time.sleep(2)

    # init communicator
    mylogger.info("==> Init torch distributed.")
    dist.init_process_group(backend="gloo",
                            init_method = "tcp://localhost:5000",
                            world_size=args.world_size,
                            rank=args.rank)

    mylogger.info("Receive model paramter from server.")
    # ---------------Param Init----------------------
    # recv parameter from server
    # for param in net.parameters():
    #    dist.broadcast(param.data, src = 0)
    # -----------------------------------------------

    mylogger.info("Init all_reduce worker group.") 

    group_map = {}
    for i in range((int)(2**(args.world_size - 1))):
        tmp = i
        base = 2
        worker_id = 1
        hashid = 0
        ranks = list()
        while(tmp > 0):
            if(tmp % 2 == 1):
                hashid += base
                ranks.append(worker_id)
            tmp = tmp//2
            base = base * 2
            worker_id = worker_id + 1
        if(len(ranks) > 1):
            group_map[hashid] = dist.new_group(ranks, backend = 'gloo')
            # mylogger.info("hashid: {}".format(hashid))
            # mylogger.info("ranks: {}".format(ranks)) 
 
    mylogger.info("==> Prepare train/test dataloader.")
    # init train/test loader
    basefolder_path = os.getcwd()
    trainloader, testloader = prepare_data(dir = basefolder_path + '/data', train_batch_size = 256 , 
                                           num_of_worker = args.world_size - 1, 
                                           worker_id = args.rank - 1)  


    net.to(device)
    for epoch in range(start_epoch, start_epoch+200):
        # print("Epoch %d:" %(epoch + 1))
        mylogger.info("Epoch %d:" %(epoch + 1))
        train(comm = tcp_sender, epoch = epoch + 1, myrank = args.rank, net = net, criterion = criterion, optimizer = optimizer, trainloader = trainloader)
        test(net = net, criterion = criterion, optimizer = optimizer, testloader = testloader, checkpoint_name = checkpoint_name)
        
        # break
        # adjust_learning_rate(optimizer, epoch, args)
