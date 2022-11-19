import numpy as np
import pandas as pd
from girth.three_pl import ability_3pl_eap as ability_eap
from parameter import simu_default_para as default_para

class Data_Simulartor_CAT:

    def __init__(self, ppl_num, item_num, test_len, ewp_rate, ci_rate):
        self.item_num, self.ppl_num, self.test_len, self.ci_rate, self.ewp_rate = item_num, ppl_num, test_len, ci_rate, ewp_rate
        self.resp, self.rt = np.full(shape=(ppl_num, item_num), fill_value=np.nan), np.full(shape=(ppl_num, item_num), fill_value=np.nan)
        self.resp_org, self.rt_org = np.full(shape=(ppl_num, item_num), fill_value=np.nan), np.full(shape=(ppl_num, item_num), fill_value=np.nan)
        self.item_pre, self.aberrant = list(), list()
        self.stage_num, self.test_len = 5, test_len
        self.item_num_each_stage = self.test_len//self.stage_num
        self.rand_guess_para = default_para['rand_guess_para']
        self.prek_acc = default_para['prek_acc']
        self.mu_theta, self.cov_theta = default_para['mu_theta'], default_para['cov_theta']


    def simulate(self):
        self.para_simulator()  # simulate item and ppl parameters for IRT model
        self.ip_selector()  # select the EWP, CI and rapid guessing ppl and items
        self.ab_stratum()  # this is prepared for item selection
        self.process()  # simulate responses

    def process(self):
        for cur_ppl in range(self.ppl_num):
            self.process_single(cur_ppl)

    def process_single(self, cur_ppl):

        cur_tested_item_order = list()
        cur_resp_full = list()

        while len(cur_tested_item_order) < self.test_len:
            if len(cur_tested_item_order) == 0:
                cur_est_theta = np.random.normal(0, 1)

            cur_item = self.item_selector(cur_est_theta, cur_tested_item_order)
            cur_tested_item_order.append(cur_item)
            self.resp[cur_ppl,cur_item], self.rt[cur_ppl,cur_item] = last_resp, last_rt = self.resp_rt_generator(cur_item, cur_ppl, cur_tested_item_order)
            cur_resp_full.append(last_resp)
            cur_est_theta = ability_eap(np.array(cur_resp_full).reshape(len(cur_tested_item_order), 1),
                                        difficulty=np.array(self.item_para['b'][cur_tested_item_order]),
                                        discrimination=np.array(self.item_para['a'][cur_tested_item_order]),
                                        guessing=np.array(self.item_para['c'][cur_tested_item_order]))

        return

    def item_selector(self, cur_theta, cur_tested_item_order):
        cur_bs = self.stratum_b[:, len(cur_tested_item_order) // self.item_num_each_stage]
        cur_items = self.stratum[:, len(cur_tested_item_order) // self.item_num_each_stage]
        cur_bs = abs(cur_bs - cur_theta)
        select_item = np.nan
        while np.isnan(select_item):
            ind = np.where(cur_bs == np.nanmin(cur_bs))[0][0]
            tmp_select_item = cur_items[ind]
            if tmp_select_item not in cur_tested_item_order:
                select_item = tmp_select_item
            else:
                cur_bs[ind] = np.nan
        return select_item

    def resp_rt_generator(self, cur_item, cur_ppl, cur_tested_item_order):
        prob = self.item_para['c'][cur_item] + (1 - self.item_para['c'][cur_item]) / \
               (1 + np.exp(-self.item_para['a'][cur_item] * (
                           self.ppl_para['ability'][cur_ppl] - self.item_para['b'][cur_item])))
        cur_resp = np.random.binomial(p=prob, n=1)
        cur_rt = np.random.normal(loc=self.item_para['speed_b'][cur_item] - self.ppl_para['speed'][cur_ppl],
                                  scale=1 / self.item_para['speed_a'][cur_item], size=1)
        self.resp_org[cur_ppl, cur_item], self.rt_org[cur_ppl, cur_item] = cur_resp, cur_rt

        if cur_item in self.ci_id and cur_ppl in self.ewp_id:
            cur_resp = np.random.binomial(n=1, p=self.prek_acc, size=1)[0]
            cur_rt = np.random.normal(loc=-2, scale=1/3.5, size=1)
            self.item_pre.append([cur_ppl, cur_item])
        if str(cur_ppl) in self.aberrant_ind and len(cur_tested_item_order)+self.aberrant_ind[str(cur_ppl)]>self.test_len:
            cur_resp = np.random.binomial(n=1, p=self.rand_guess_para['acc'], size=1)
            cur_rt = np.random.normal(loc=-2, scale=1 / 3.5, size=1)
            self.aberrant.append([cur_ppl, cur_item])

        return cur_resp, cur_rt

    def para_simulator(self):

        theta_speed = np.random.multivariate_normal(mean=self.mu_theta, cov=self.cov_theta, size=self.ppl_num)
        self.ppl_para = pd.DataFrame(theta_speed, columns=['ability', 'speed'])

        item_ind = np.array(range(self.item_num)) #random.sample(set(np.arange(540)), self.item_num)
        f = np.loadtxt(open('abcAB.txt', 'r'))[item_ind,1:]
        self.item_para =pd.DataFrame(f, columns=['a', 'b', 'c', 'speed_a', 'speed_b'])


    def ip_selector(self):
        ewp_num = int(self.ewp_rate * self.ppl_num)
        self.ewp_id = np.sort(np.random.choice(a = np.arange(self.ppl_num), size=ewp_num, replace=False))
        ci_num = int(self.ci_rate * self.item_num)
        self.ci_id = np.sort(np.random.choice(a = np.arange(self.item_num), size=ci_num, replace=False))
        guess_ppl_num = int(self.ppl_num*self.rand_guess_para['rate_ppl'])
        guess_ppl = np.random.choice(a = np.delete(np.arange(self.ppl_num), self.ewp_id), size=guess_ppl_num, replace=False)
        self.aberrant_ind = {}
        for gp in guess_ppl:
            guess_num = int(np.round(np.random.gamma(self.rand_guess_para['gamma'][0], self.rand_guess_para['gamma'][1], 1)[0],0))
            self.aberrant_ind[str(gp)] = guess_num


    def ab_stratum(self):
        self.stratum = list()  # item_ID
        self.stratum_b = list()  # exact b value
        b_sort = np.array(np.argsort(self.item_para['b'])).reshape((self.item_num // self.stage_num, self.stage_num))
        for i in range(self.item_num // self.stage_num):
            tmp = b_sort[i, :]
            c = np.array(self.item_para['a'][tmp])
            self.stratum_b.append(np.array(self.item_para['b'][tmp[np.argsort(c)]]))
            self.stratum.append(tmp[np.argsort(c)])
        self.stratum = np.array(self.stratum)
        self.stratum_b = np.array(self.stratum_b)
