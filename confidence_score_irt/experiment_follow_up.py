
import numpy as np
import tensorflow as tf
import copy
from girth.three_pl import ability_3pl_eap as ability_eap
from girth.three_pl.three_pl_mml import threepl_mml

class ExperimentFollowUp:

    def __init__(self, data, detect_res):

        self.flagged = detect_res.res_mid
        self.nn_ppl = np.where(detect_res.count_item!=0)

        self.resp, self.rt = data.resp.astype('float'), data.rt
        self.resp_org = data.resp_org
        self.ppl_num, self.item_num = self.resp.shape

        self.reduce_node_num, self.middle_node_num = 50, 250


    def process(self):
        self.data_prepare()
        self.nn_model()
        self.confidence_score()

    def data_prepare(self):

        "delete detected"
        self.resp_no_dete = copy.deepcopy(self.resp)
        self.rt_no_dete = copy.deepcopy(self.rt)
        for item in self.flagged['comp_item_name']:
            for ppl in self.flagged['comp_ppl_id']:
                self.resp_no_dete[ppl, item] = np.nan
                self.rt_no_dete[ppl, item] = np.nan


    def nn_model(self):

        def my_leaky_relu(x):
            return tf.nn.leaky_relu(x, alpha=0.2)

        "training"
        ac, cur_node_num, nodes_nums = my_leaky_relu, self.item_num - self.reduce_node_num, [self.item_num]
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(units=cur_node_num, input_dim=self.item_num, activation=ac))
        while cur_node_num > self.middle_node_num:
            nodes_nums.append(cur_node_num)
            cur_node_num -= self.reduce_node_num
            self.model.add(tf.keras.layers.Dense(units=cur_node_num, activation=ac))
        nodes_nums.reverse()
        for cur_node_num in nodes_nums:
            self.model.add(tf.keras.layers.Dense(units=cur_node_num, activation=ac))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        # IRT part
        # prepare training data
        cur_resp = copy.deepcopy(self.resp_no_dete)
        # estimate parameter
        self.item_para = pd.DataFrame()
        item_para_est = threepl_mml(np.transpose(cur_resp))
        self.item_para['a'] = item_para_est['Discrimination']
        self.item_para['b'] = item_para_est['Difficulty']
        self.item_para['c'] = item_para_est['Guessing']
        cur_thetas = ability_eap(np.transpose(cur_resp),
                                    difficulty=np.array(self.item_para['b']),
                                    discrimination=np.array(self.item_para['a']),
                                    guessing=np.array(self.item_para['c']))
        resp_for_replace = list()
        for cur_theta in cur_thetas:
            resp_for_replace.append(np.array(self.item_para['c'] + (1 - self.item_para['c']) /
                                        (1 + np.exp(-self.item_para['a'] * (cur_theta - self.item_para['b'])))))
        cur_resp = np.random.binomial(p=resp_for_replace, n=1)

        # predict missing
        row_mean, col_mean = np.nanmean(cur_resp,axis=1), np.nanmean(cur_resp,axis=0)
        col_mean[np.where(np.isnan(col_mean))] = np.nanmean(cur_resp)
        loc = np.where(np.isnan(cur_resp))
        cur_resp[loc] = (row_mean[loc[0]]+col_mean[loc[1]])/2
        cur_resp = np.random.binomial(1, cur_resp, cur_resp.shape) #as observed p=0/1, obs info also included
        x_train, y_train= cur_resp, cur_resp
        # conduct training
        self.model.fit(x_train, y_train, epochs=300, verbose=0)

        "prediction"
        resp_for_pred = copy.deepcopy(self.resp)
        loc = np.where(np.isnan(resp_for_pred))
        resp_for_pred[loc] = cur_resp[loc]
        self.nn_pred_prob = self.model.predict(resp_for_pred)

    def confidence_score(self):

        loc = np.where(~np.isnan(self.resp))

        "difference with original response data"
        self.nn_pred_resp_05 = 1 * (self.nn_pred_prob > 0.5)
        self.dif_nn = np.round(np.nanmean(abs(self.resp_org[loc] - self.nn_pred_resp_05[loc])), 3)

        "calculate degree of unexpectancy at examinee and item level"
        "note, only the examinee level will be used"
        tmp = 1.0*(self.nn_pred_resp_05<self.resp)
        tmp[np.where(np.isnan(self.resp))] = np.nan

        count_nn_1 = np.nanmean(tmp[self.flagged['comp_ppl_id'], :])
        count_nn_2 = np.nanmean(np.delete(tmp, self.flagged['comp_ppl_id'], 0))
        self.ppl_ratio = ((count_nn_1+0.2) / (count_nn_2+0.2)) if count_nn_2<0.01 else count_nn_1/count_nn_2

        count_nn_1 = np.nanmean(tmp[:, self.flagged['comp_item_name']])
        count_nn_2 = np.nanmean(np.delete(tmp, self.flagged['comp_item_name'], 1))
        self.item_ratio = ((count_nn_1+0.2) / (count_nn_2+0.2)) if count_nn_2<0.01 else count_nn_1/count_nn_2

        # self.score = (self.ppl_ratio+self.item_ratio)/2



class AfterExperimentFollowUp:

    def __init__(self, experiment):

        self.nn_pred_prob = experiment.nn_pred_prob
        self.dif_nn = experiment.dif_nn

        self.ppl_ratio = experiment.ppl_ratio
        self.item_ratio = experiment.item_ratio
        #self.score = experiment.score
