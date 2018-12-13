## Wrapper for execution in PySpark - First block
def model_wrapper():
    
    from hops import hdfs
    from tempfile import TemporaryFile
    import numpy as np
    import os
    import time
    #import tensorflow as tf
    #from keras.models import Sequential
    #from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    #from keras.utils import to_categorical
    #from sklearn.preprocessing import OneHotEncoder
    from PIL import Image
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    
    np.set_printoptions(threshold=np.inf)
    np.random.seed(23333)
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512*block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out


    def ResNet18():
        return ResNet(BasicBlock, [2,2,2,2])
    
    def ResNet34():
        return ResNet(BasicBlock, [3,4,6,3])

    def ResNet50():
        return ResNet(Bottleneck, [3,4,6,3])

    def ResNet101():
        return ResNet(Bottleneck, [3,4,23,3])

    def ResNet152():
        return ResNet(Bottleneck, [3,8,36,3])
    
    ## Early christmas gift from the mentors :)
    def open_hdfs(path):
        temp_file = TemporaryFile()
        with hdfs.get_fs().open_file(path, 'r') as data_file:
            temp_file.write(data_file.read())
            temp_file.seek(0)
        return temp_file

    def loadnp(path):
        temp_file = open_hdfs(path)
        data = np.load(temp_file)
        return data

    def save(data, path):
        with hdfs.get_fs().open_file(path, 'w') as data_file:
            data_file.write(data)

    def savetf(temp_file, path, close=True):
        with hdfs.get_fs().open_file(path, 'w') as data_file:
            temp_file.seek(0)
            data_file.write(temp_file.read())
        if close:
            temp_file.close()
            
    def savenp(data, path):
        temp_file = TemporaryFile()
        np.save(temp_file, data)
        savetf(temp_file, path)
    
    def log(message):
        hdfs.log(str(message))
    
    def vectorize(sequences,dimension=10):
        results = np.zeros((len(sequences),dimension))
        for i,sequence in enumerate(sequences):
            results[int(i),int(sequence)] = 1
        return results
    
    ## Load datasets
    data_path = 'hdfs://10.0.104.196:8020/Projects/Mentors/hackathon_AB/'
    
    # Unlabled images
    #unlabled = loadnp(data_path + 'A.npy')
    
    # Full labled training set
    with loadnp(data_path + 'B_train_full.npz') as train:
        X_train, y_train = train['X'], train['y']
    
    # Labled trainig set provided in the same format as the final evaluation data set
    # A dict of python tuples {num_samples0: (x0, y0), ..., num_samplesN: (xN, yN)}
    with loadnp(data_path + 'B_train.npz') as train:
        train_eval = train['data'].item()
    
    # Test set
    with loadnp(data_path + 'B_test.npz') as test:
        X_test, y_test = test['X'], test['y']
    
    ##reshape data into use
    #data_unlabled = unlabled.reshape(100000, 32, 32, 3)#unlabled data
    
    data_labled_x = X_train.reshape(4000, 32, 32, 3)#labled data image
    #data_labled_y = to_categorical(y_train)
    data_labled_y = y_train.reshape(4000,1)

    test_x = X_test.reshape(1000,32, 32, 3)#labled data image
    #test_y = to_categorical(y_test)
    test_y = y_test
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    #net = ShuffleNetV2(1)
    net = ResNet18()
    net.cuda()
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_loss = 0.
    correct = 0.
    epoch = 100
    
    batch_size = 100
    iter_num = int(4000 / batch_size)
    val_iter_num = int(1000 / batch_size)
    idxs = np.arange(4000)
    val_idxs = np.arange(1000)
    
    batch_x = np.zeros((batch_size, 3, 32, 32), np.float32)
    #batch_y = np.zeros((batch_size, ), np.float32)
    
    for e in range(epoch):
        net.train()
        np.random.shuffle(idxs)
        start = 0
        end = start + batch_size
        
        training_loss = 0.
        correct = 0.
        total = 0.
        
        if e >= 3:
            for p in optimizer.param_groups:
                p['lr'] = 0.5 * (1 + np.cos(e * np.pi/100)) * 0.001
        
        
        for i in range(iter_num):
            batch_idx = idxs[start:end]
            start = end
            end = start + batch_size
            
            for j in range(batch_size):
                img = data_labled_x[batch_idx[j], :, :, :]
                img = transform_train(Image.fromarray(img))
                batch_x[j, :, :, :] = img
            
            #batch_x = data_labled_x[batch_idx, :, :, :]
            batch_y = data_labled_y[batch_idx].reshape((batch_size,))
            
            input_x = torch.from_numpy(batch_x).cuda()
            input_y = torch.from_numpy(batch_y).cuda().long()
            
            outputs = net(input_x)
            loss = criterion(outputs, input_y)
            _, predicted = outputs.max(1)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            total += batch_size
            correct += predicted.eq(input_y).sum().item()
            
            if i % 10 == 0:
                log('epoch %d  iter %d  loss %f  acc %f'%(e, i, loss.item(), correct / total))
        
        log('epoch %d  training acc %f'%(e, correct / total))
        
        net.eval()
        correct = 0.
        total = 0.
        start = 0
        end = start + batch_size
        for i in range(val_iter_num):
            batch_idx = val_idxs[start:end]
            start = end
            end = start + batch_size
            
            for j in range(batch_size):
                img = test_x[batch_idx[j], :, :, :]
                img = transform_test(Image.fromarray(img))
                batch_x[j, :, :, :] = img
            
            #batch_x = data_labled_x[batch_idx, :, :, :]
            batch_y = test_y[batch_idx].reshape((batch_size,))
            
            input_x = torch.from_numpy(batch_x).cuda()
            input_y = torch.from_numpy(batch_y).cuda().long()
            
            outputs = net(input_x)
            _, predicted = outputs.max(1)
            
            total += batch_size
            correct += predicted.eq(input_y).sum().item()
        
        log('epoch %d  val acc %f'%(e, correct / total))


## Launch experiment - Second block
from hops import experiment
experiment.launch(model_wrapper, name='Our winning model', local_logdir=True)
