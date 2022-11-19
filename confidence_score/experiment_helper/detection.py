import numpy as np
from sklearn import svm
import copy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
use step 1 to initialize the training data set
loop step 2 to step 3 until all the responses are tagged
"""


class Detector:

    def __init__(self, data, var_cri, cri_ab):
        self.crt, self.crt_ppl, self.crt_item, self.rt, self.crt_full = data['rt'], data['crt_ppl'], data['crt_item'], data['rt'], data['crt_full']
        self.resp = data['resp']
        self.test_len = data['test_len']
        self.cri_ab, self.cri_nor = cri_ab, cri_ab
        self.var_cri = var_cri

        [self.ppl_num, self.item_num] = np.shape(data['crt'])

        self.abnormal_res = pd.DataFrame(columns=['ppl', 'item'])
        self.prob_ci = np.zeros(self.item_num)
        self.prob_ewp = np.zeros(self.ppl_num)

        self.ques_ewp = list()
        self.ques_ci = list()


    def process(self):

        while True:
            self.get_training_sample()
            self.classify()
            self.iden_que() # identify suspicious
            self.prepare_next() # update response time and unlabeled set
            if self.check_stop():
                break


    def get_training_sample(self):

        crt = copy.deepcopy(self.crt)
        tmp_ind = np.where(~np.isnan(crt))

        # generate a N*3 matrix, N is the number of correct responses
        # 0 col: crt; 1-2 col: response id
        tmp = crt[tmp_ind]
        tmp_ind = np.array(tmp_ind).transpose()
        tmp = np.concatenate([tmp[:, np.newaxis], tmp_ind], axis=1)
        tmp = tmp[np.argsort(tmp[:, 0]), :]

        training_ab = np.transpose(tmp[0:self.cri_ab, :])
        training_nor = np.transpose(tmp[-self.cri_nor:, :])
        ab_id = list(np.array([training_ab[1, :], training_ab[2, :]]).astype(int))
        nor_id = list(np.array([training_nor[1, :], training_nor[2, :]]).astype(int))
        self.abnormal_rt = np.array([self.crt_ppl[ab_id], self.crt_item[ab_id]]).transpose()
        self.normal_rt = np.array([self.crt_ppl[nor_id], self.crt_item[nor_id]]).transpose()

        # test responses and response IDs.
        self.test_rt, self.test_id = list(), list()
        for i in set(ab_id[0]):
            for j in set(ab_id[1]):
                if ~np.isnan(crt[i, j]):
                    self.test_rt.append([self.crt_ppl[i, j], self.crt_item[i, j]])
                    self.test_id.append([i, j])

    def classify(self):
        if len(self.abnormal_rt) == 0 or len(self.normal_rt) == 0:
            return
        x = np.concatenate([self.abnormal_rt, self.normal_rt],axis=0)
        y = np.concatenate([np.ones(len(self.abnormal_rt)), np.zeros(len(self.normal_rt))])
        scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x)
        x = scaling.transform(x)
        self.test_rt = scaling.transform(self.test_rt)
        self.clf = svm.SVC(kernel='poly', degree=3)
        self.clf.fit(x, y)
        cur_id = np.array(self.test_id)
        self.abnormal_pred = pd.DataFrame(cur_id[self.clf.predict(self.test_rt) == 1], columns=['ppl', 'item'])
        self.abnormal_res = self.abnormal_res.append(self.abnormal_pred)

    def iden_que(self):
        self.ques_ci = list()
        dete_item = list(self.abnormal_res['item'].unique())
        dete_ppl = list(self.abnormal_res['ppl'].unique()) if len(self.ques_ewp) == 0 else list(self.ques_ewp)
        for dete_item_ele in dete_item: # filter item
            dif = np.nanmean(self.crt_full[dete_ppl, dete_item_ele]) - \
                  np.nanmean(np.delete(self.crt_full[:, dete_item_ele], dete_ppl))
            if ~np.isnan(dif) and dif < 0:
                self.ques_ci.append(dete_item_ele)

        self.ques_ewp = list()
        dete_ppl = list(self.abnormal_res['ppl'].unique())
        dete_item = list(self.ques_ci)
        for dete_ppl_ele in dete_ppl: # filter ppl
            dif = np.nanmean(self.crt_full[dete_ppl_ele, dete_item]) - \
                  np.nanmean(np.delete(self.crt_full[dete_ppl_ele, :], dete_item))
            if ~np.isnan(dif) and dif < 0:
                self.ques_ewp.append(dete_ppl_ele)

    def check_stop(self):
        temp_unc = self.rt[np.where(~np.isnan(self.rt))]
        self.cur_var = np.nanvar(temp_unc)
        if self.cur_var<self.var_cri or len(temp_unc)<=self.cri_ab or len(self.abnormal_pred['ppl'])==0: # stop rule
            return True
        return False

    def prepare_next(self):

        group_item = self.abnormal_res.groupby(['item']).count()
        for item, count in group_item.iterrows():
            if item in self.ques_ci:
                nonnan_num = np.sum(1*(~np.isnan(self.crt_full[:, item])))
                self.prob_ci[item] = (count / nonnan_num)
                self.crt[:, item] -= self.prob_ci[item] * np.nanstd(self.crt[:, item])
                self.crt_item[:, item] -= self.prob_ci[item] * np.nanstd(self.crt_item[:, item])
                self.crt_ppl[:, item] -= self.prob_ci[item] * np.nanstd(self.crt_ppl[:, item])


        group_ppl = self.abnormal_res.groupby(['ppl']).count()
        for ppl, count in group_ppl.iterrows():
            if ppl in self.ques_ewp:
                self.prob_ewp[ppl] = (count / self.test_len)
                self.crt[ppl, :] -= self.prob_ewp[ppl] * np.nanstd(self.crt[ppl, :])
                self.crt_item[ppl, :] -= self.prob_ewp[ppl] * np.nanstd(self.crt_item[ppl, :])
                self.crt_ppl[ppl, :] -= self.prob_ewp[ppl] * np.nanstd(self.crt_ppl[ppl, :])

        for index, row in self.abnormal_pred.iterrows():
            self.crt[row['ppl'], row['item']] = np.nan
            self.rt[row['ppl'], row['item']] = np.nan
            self.crt_item[row['ppl'], row['item']] = np.nan
            self.crt_ppl[row['ppl'], row['item']] = np.nan
