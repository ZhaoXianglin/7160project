import time

import torch.nn as nn
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


class DeepFM(nn.Module):
    def __init__(self, feature_size, int_features, cate_features, cate_each_features_size, emb_size=8,
                 dnn_hidden_size=[256, 128],
                 dropout_rate=[0.2, 0.2], device='cuda'):
        super(DeepFM, self).__init__()
        self.feature_size = feature_size

        self.int_features = int_features
        self.int_feature_size = len(int_features)
        self.cate_features = cate_features
        self.cate_feature_size = len(cate_features)
        self.cate_each_features_size = cate_each_features_size
        self.embedding_size = emb_size

        self.dnn_hidden = dnn_hidden_size

        self.device = device

        # init input
        if self.int_feature_size != 0:
            self.fm_1st_int_vec = nn.Linear(self.int_feature_size, 1)

        self.fm_1st_cate_emb = self.create_emb_matrix(1, self.cate_each_features_size)
        self.fm_2nd_cate_emb = self.create_emb_matrix(self.embedding_size, self.cate_each_features_size)

        self.dnn_each_size = [self.cate_feature_size * self.embedding_size] + self.dnn_hidden

        self.dense_linear = nn.Linear(self.int_feature_size, self.cate_feature_size * self.embedding_size)

        # DNN init 3 layers
        for i in range(1, len(self.dnn_each_size)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.dnn_each_size[i - 1], self.dnn_each_size[i]))
            setattr(self, 'Norm_' + str(i), nn.BatchNorm1d(self.dnn_each_size[i]))
            setattr(self, 'act_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout_rate[i - 1]))

        self.relu = nn.ReLU()

        # Output
        self.sigmoid = nn.Sigmoid()
        self.out_linear = nn.Linear(self.dnn_each_size[-1], 1)

    def create_emb_matrix(self, emb_size, feature_size):
        return nn.ModuleList([nn.Embedding(vec_size, emb_size) for vec_size in feature_size])

    def forward(self, cate_data, int_data):
        """
        dataloader [cate_data, int_data, label]
        :return: y
        """
        # print(cate_data)
        ## FM 1
        fm_1st_cate_res = []
        for i, emb in enumerate(self.fm_1st_cate_emb):
            fm_1st_cate_res.append(emb(cate_data[:, i].unsqueeze(1)).view(-1, 1))
        fm_1st_cate_res = torch.cat(fm_1st_cate_res, dim=1)
        fm_1st_cate_res = torch.sum(fm_1st_cate_res, 1, keepdim=True)

        fm_1st_int_res = self.fm_1st_int_vec(int_data)

        # fm 1st
        fm_1st_all = fm_1st_cate_res + fm_1st_int_res

        ## FM 2
        fm_2nd_res = []
        for i, emb in enumerate(self.fm_2nd_cate_emb):
            fm_2nd_res.append(emb(cate_data[:, i].unsqueeze(1)))

        fm_2nd_concat = torch.cat(fm_2nd_res, dim=1)

        # FM PART
        sum_embed = torch.sum(fm_2nd_concat, 1)
        square_sum_embed = sum_embed * sum_embed

        square_embed = fm_2nd_concat * fm_2nd_concat
        sum_square_embed = torch.sum(square_embed, 1)

        # 1/2 {}
        fm_res = (square_sum_embed - sum_square_embed) / 2

        fm_2nd_all = torch.sum(fm_res, 1, keepdim=True)

        ### DNN ###
        dense_out = self.relu(self.dense_linear(int_data))
        dnn_out = torch.flatten(fm_2nd_concat, 1) + dense_out

        for i in range(1, len(self.dnn_each_size)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'Norm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'act_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        dnn_out = self.out_linear(dnn_out)
        y = self.sigmoid(fm_1st_all + fm_2nd_all + dnn_out)
        return y

    def fit(self, train_loader, eval_loader, epochs=30):
        best_AUC = 0.0
        model = self.train().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

        loss_func = nn.BCELoss().to(self.device)

        for epoch in range(epochs):
            train_loss_total = 0.0
            print(time.time())
            for i, (cate_data, int_data, label) in enumerate(train_loader):
                cate_data = cate_data.to(self.device)
                int_data = int_data.to(self.device)
                label = label.to(self.device)
                predict_y = model(cate_data, int_data).view(-1)
                loss = loss_func(predict_y, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_total += loss.cpu().item()
                if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
                    print("Epoch:{0}, Step:{1}/{2}, Loss:{3}".format(epoch + 1, i + 1, len(train_loader),
                                                                     train_loss_total / (i + 1)))
            current_AUC = self.check_acc(eval_loader, model)
            if current_AUC > best_AUC:
                best_AUC = current_AUC
            print("Best AUC:{0}".format(best_AUC))

        scheduler.step()

    def check_acc(self, eval_loader, model):
        best_AUC = 0.0
        model.eval()
        with torch.no_grad():
            valid_labels, valid_preds = [], []
            for i, (cate_data, int_data, label) in enumerate(eval_loader):
                cate_data = cate_data.to(self.device)
                int_data = int_data.to(self.device)
                label = label.to(self.device)
                predict_y = model(cate_data, int_data).reshape(-1).data.cpu().numpy().tolist()
                valid_preds.extend(predict_y)
                valid_labels.extend(label.cpu().numpy().tolist())

        currect_AUC = roc_auc_score(valid_labels, valid_preds)
        if currect_AUC > best_AUC:
            best_AUC = currect_AUC
        torch.save(model.state_dict(), "model/deepfm_best.pth")
        # print('Current AUC: {0}'.format(currect_AUC))
        return best_AUC


if __name__ == '__main__':
    data = pd.read_csv("data/criteo_sample_50w.csv")
