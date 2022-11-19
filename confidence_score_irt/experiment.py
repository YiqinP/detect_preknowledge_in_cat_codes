

import warnings
import numpy as np
import copy
from experiment_helper.evaluation import evaluate
from experiment_helper.detection import Detector
from parameter import exp_default_para
import scipy.stats as stats

warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)



class Experiment:

    def __init__(self, ppl_num, item_num, test_len, ewp_rate, ci_rate, iterat_times, ab_cri, data):
        self.ppl_num, self.item_num, self.test_len, self.ewp_rate, self.ci_rate = ppl_num, item_num, test_len, ewp_rate, ci_rate
        self.ab_cri = ab_cri
        self.iterat_times, self.cur_times = iterat_times, 0
        self.data = data
        self.count_ppl, self.count_item = np.zeros(shape=(self.ppl_num)), np.zeros(shape=(self.item_num))
        self.res = {'comp_item_name': [], 'comp_ppl_id':[]}
        self.rt = None
        self.iterat_prop = exp_default_para['iterat_prop']

    def process(self):

        self.prepare_data()

        while self.cur_times < self.iterat_times:

            # sampling (ensemble learning)
            tmp_data = {'crt': copy.deepcopy(self.crt), 'crt_ppl': copy.deepcopy(self.crt_ppl),
                        'crt_item': copy.deepcopy(self.crt_item), 'rt': copy.deepcopy(self.rt)}
            select_id = self.valid_id[np.random.choice(len(self.valid_id), int(len(self.valid_id) * (1-self.iterat_prop)),
                                         replace=False), :].transpose()
            for name, content in tmp_data.items():
                content[select_id[0, :], select_id[1, :]] = np.nan
            tmp_data['crt_full'], tmp_data['resp'], tmp_data['test_len'] = self.crt, self.resp, self.test_len

            # conduct detection
            detection = Detector(tmp_data, self.var_cri, self.ab_cri)
            detection.process()

            # flag counts
            self.count_ppl[detection.ques_ewp] += 1
            self.count_item[detection.ques_ci] += 1

            self.cur_times += 1

        # combine results from all repetitions
        self.merge_eval()


    def prepare_data(self):

        if self.rt is None:
            self.rt, self.resp = copy.deepcopy(self.data.rt), self.data.resp
            rt_ppl, rt_item = np.nanmedian(self.rt, axis=1), np.nanmedian(self.rt, axis=0)   # hang, lie
            self.crt_ppl = self.rt - np.repeat([rt_ppl], self.item_num, axis=0).transpose()
            self.crt_item = self.rt - np.repeat([rt_item], self.ppl_num, axis=0)
            self.crt = self.rt - np.repeat([rt_ppl], self.item_num, axis=0).transpose() - np.repeat([rt_item], self.ppl_num, axis=0)

            cen = np.nanmedian(self.rt)
            right = self.rt[np.where(self.rt > cen)]
            self.var_cri = np.nanmean(np.square(right - cen))

            self.rt[np.where(self.resp == 0)] = np.nan
            self.crt[np.where(self.resp == 0)] = np.nan
            self.crt_item[np.where(self.resp == 0)] = np.nan
            self.crt_ppl[np.where(self.resp == 0)] = np.nan

            self.valid_id = np.array(np.where(~np.isnan(self.rt))).transpose()


            cen = stats.mode(np.round(self.rt.flatten(), 3))[0][0]
            right = self.rt[np.where(self.rt > cen)]
            self.var_cri = np.nanmean(np.square(right - cen))

            self.std = np.nanstd(self.crt)


    def merge_eval(self):

        self.nn_ppl = np.where(self.count_ppl!=0)
        self.nn_item = np.where(self.count_item!=0)

        step1 = np.multiply(np.linspace(0.8, 1, 21), np.max(self.count_item))
        step2 = np.multiply(np.linspace(0.8, 1, 21), np.max(self.count_ppl))
        cur_max = 0
        for i in range(len(step1)):
            for ii in range(len(step2)):
                group_ab_item = np.where(self.count_item >= step1[i])[0]
                group_mem_id = np.where(self.count_ppl >= step2[ii])[0]
                cur_ci_ewp = np.nanmean(self.crt[group_mem_id, :][:, group_ab_item])
                cur_nci_ewp = np.nanmean(np.delete(self.crt[group_mem_id, :], group_ab_item, axis=1))
                cur_ci_newp = np.nanmean(np.delete(self.crt[:, group_ab_item], group_mem_id, axis=0))
                cur = ((cur_nci_ewp - cur_ci_ewp) + (cur_ci_newp - cur_ci_ewp))/self.std
                if cur > cur_max:
                    self.res['comp_item_name'] = group_ab_item
                    self.res['comp_ppl_id'] = group_mem_id
                    cur_max = cur
        self.res_mid = self.res
        self.eva_mid = evaluate(self.data, self)



class AfterExperiment:

    def __init__(self, experiment):
        self.count_ppl, self.count_item = experiment.count_ppl, experiment.count_item
        self.res_mid = experiment.res_mid
        self.eva_mid = experiment.eva_mid
