import torch
import torch.nn as nn
import os
from torch import optim
from model import VisionTransformer, Encoder
from sklearn.metrics import confusion_matrix, accuracy_score
from data_loader import get_loader

import shutil
import torchvision.utils as vutil

class Solver(object):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.test_loader = get_loader(args)

        self.model = VisionTransformer(args).cuda()
        self.ce = nn.CrossEntropyLoss()

        print('--------Network--------')
        print(self.model)

        if args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'Transformer.pt')))

    def test_dataset(self, db='test'):
        self.model.eval()

        actual = []
        pred = []
        total_loss = 0
        if db.lower() == 'train':
            loader = self.train_loader
        elif db.lower() == 'test':
            loader = self.test_loader

        for (imgs, labels) in loader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                class_out = self.model(imgs)
            _, predicted = torch.max(class_out.data, 1)
            clf_loss = self.ce(class_out, labels)
            total_loss += clf_loss

            actual += labels.tolist()
            pred += predicted.tolist()

        acc = accuracy_score(y_true=actual, y_pred=pred) * 100
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.n_classes))

        return acc, cm, total_loss/len(loader)

    def test(self):
        train_acc, cm, train_loss = self.test_dataset('train')
        # self.result_write.write('Tr Acc: {}, Tr Loss: {:.4f}'.format(train_acc, train_loss) + '\n')
        print("Tr Acc: %.2f" % (train_acc))
        print(cm)

        test_acc, cm, test_loss = self.test_dataset('test')
        # self.result_write.write('Te Acc: {}, Te Loss: {:.4f}'.format(test_acc, test_loss) + '\n')
        print("Te Acc: %.2f" % (test_acc))
        print(cm)
    
        return train_acc, test_acc

    def train(self):
        best_test_acc = 0
        best_test_loss = float('inf')
        iter_per_epoch = len(self.train_loader)

        optimizer = optim.AdamW(self.model.parameters(), self.args.lr, weight_decay=1e-3)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)
        
        for epoch in range(self.args.epochs):

            self.model.train()

            for i, (imgs, labels) in enumerate(self.train_loader):

                imgs, labels = imgs.cuda(), labels.cuda()

                logits = self.model(imgs)
                clf_loss = self.ce(logits, labels)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                if i % 50 == 0 or i == (iter_per_epoch - 1):
                    print('Ep: %d/%d, it: %d/%d, err: %.4f' % (epoch + 1, self.args.epochs, i + 1, iter_per_epoch, clf_loss))

            test_acc, cm, test_loss = self.test_dataset('test')
            # self.result_write.write('Ep: {}, Test acc: {:.4f}'.format(epoch + 1, test_acc) + '\n')
            print("Test acc: %0.2f" % (test_acc))
            print(cm,"\n")

            cos_decay.step()

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), 'model/mnist/Transformer.pt')
            if test_loss < best_test_loss:
                best_test_loss = test_loss

        # self.result_write.write('Best Test acc: {:.4f}, Best Test loss: {:.4f}'.format(best_test_acc, best_test_loss) + '\n')




