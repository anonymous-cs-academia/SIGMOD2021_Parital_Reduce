import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from multiprocessing import Process, Queue
import socket
import time
import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
from logger import Logger
from models import *

best_acc = 0 # best test accuracy

def adjust_learning_rate(optimizer, epoch, _lr):
    # lr_decay = 0.1
    # lr_decay_freq = 130
    # lr = _lr * (lr_decay ** (epoch // lr_decay_freq))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # lr_decay = 0.1
    # lr_decay_freq = 130
    # lr = _lr * (lr_decay ** (epoch // lr_decay_freq))
    if epoch < 100:
        lr = 0.001 * 8
    elif epoch < 200:
        lr = 0.001 * 4
    elif epoch < 300:
        lr = 0.001 * 2
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def test(net = None, criterion = None, testloader = None, checkpoint_name = None):
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
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        # print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + checkpoint_name)
        best_acc = acc

def recv_signal(worker_id = None, comm = None):
    # mylogger.info("==>Build connection with worker: %d"%(worker_id))

    while True:
        '''
            signal type:
            len = 6
            type(-1, -2, -3) + worker_id(1 - 8) + epoch(000 - 999)
            ready signal: -1
            push  signal: -2
            pull  signal: -3
        '''
        signal = comm.recv(10)
        if(len(signal) != 0):
            signal = signal.decode("utf-8")
            if signal == 'end':
                break
            if signal[0:2] == '-1': # ready signal
                ready_queue.put(signal[2:9])
            else:
                ''' wrong signal '''
                mylogger.info("Signal type error")
                sys.exit(-1)

def send_group(partial_reduce_num = None , comm_list = None):
    # mylogger.info("Building send AllReduce group...")
    cnt = 0
    while True:
        if(ready_queue.qsize() >= partial_reduce_num):
            group_list = []
            send_str = str(partial_reduce_num)
            send_str = send_str.encode('utf-8')
            max_time_stamp = -1
            for i in range(partial_reduce_num):
                cnt = cnt + 1
                worker_info = ready_queue.get()
                worker_id = worker_info[0]
                worker_time_stamp = int(worker_info[1:])
                max_time_stamp = max(max_time_stamp, worker_time_stamp)
                send_str = send_str + worker_id.encode('utf-8')
                group_list.append(int(worker_id))
            max_time_stamp_info = "%06d"%(max_time_stamp)
            max_time_stamp_info = max_time_stamp_info.encode('utf-8')
            for worker_id in group_list:
                comm_list[worker_id].send(send_str+max_time_stamp_info)
            mylogger.info("group:{}".format(send_str.decode('utf-8')))
            

if __name__ == "__main__":
    
    global mylogger
    global device
    global mutex
    global net
    # global optimizer
    global args
    global p2p_group_map
    global ready_queue
    global update_queue

    parser = argparse.ArgumentParser()
    # mode: server, worker
    # parser.add_argument('--mode', default = None, type = str)

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--rank', '-r', default = -1, type = int)
    parser.add_argument('--world-size', '-s', default = -1, type = int)
    parser.add_argument('--gpu-id', '-g', default = -1, type = int)
    parser.add_argument('--partial-reduce-num', '-k', default = -1, type = int)
    # parser.add_argument('--check_point', '-c', default = None, type = str)

    args = parser.parse_args()
    log_dir = "./combine"
    log_name = "/server.log"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    L = Logger()
    L.set_log_name(log_dir + log_name)
    mylogger = L.get_logger()

    device = torch.device('cpu')
    # init Model
    mylogger.info("==> Building ResNet-34 model...")
    # net = VGG('VGG19')
    net = ResNet34()
    # net = LeNet()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    # multi thread queue
    ready_queue = Queue()
    update_queue = Queue()

    mylogger.info("==> Init socket.")
    tcp_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    HOST = socket.gethostname()
    PORT = 10000
    num_workers = 8
    
    tcp_listener.bind((HOST, PORT))
    tcp_listener.listen(num_workers)
    init_comm_list = [0] * (num_workers + 1)
    comm_list = [0] * (num_workers + 1)
    mylogger.info("tcp begin listen")
    for i in range(num_workers):
        comm, addr = tcp_listener.accept()
        init_comm_list[i+1] = comm
    mylogger.info("tcp end listen")
    for i in range(num_workers):
        worker_id = init_comm_list[i+1].recv(2)
        worker_id = int(worker_id.decode("utf-8"))
        print("worker_id for communicator:", worker_id)
        comm_list[worker_id] = init_comm_list[i+1]

    mylogger.info("==> Init listener process.")
    listener_process = []
    for i in range(num_workers):
        p = Process(target = recv_signal, args = (i+1, comm_list[i+1]))
        listener_process.append(p)
        
    mylogger.info("==> Init torch distributed.")
    dist.init_process_group(backend="gloo",
                init_method="tcp://localhost:5000",
                world_size= args.world_size,
                rank=args.rank)
    
    mylogger.info("==> Broadcast model parameters to all workers")
    # ---------------Param Init----------------------
    # broadcast parameter to worker
    # for param in net.parameters():
    #    dist.broadcast(param.data, src = 0)
    # -----------------------------------------------
    for param in net.parameters():
        param.grad = torch.zeros_like(param)

    mylogger.info("==> Start Listener Process.")
    for p in listener_process:
        p.start()
    
    
    mylogger.info("==> Start Divide Process.")
    groupDiv_process = Process(target = send_group, args = (args.partial_reduce_num, comm_list))
    groupDiv_process.start()

    mylogger.info("==> End Listener Process.")
    for p in listener_process:
        p.join()

    mylogger.info("==> End Divide Process.")
    groupDiv_process.join()
