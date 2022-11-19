from parameter import para, para_null, exp_default_para
import numpy as np
import pickle
import pandas as pd
from experiment_helper.evaluation import evaluate
from data_simulation_cat import Data_Simulartor_CAT


def main(simu_ids, para):


    for simu_id in simu_ids:
        print(simu_id)
        for ppl_num in para['ppl_num']:
            for item_num in para['item_num']:
                for test_len in para['test_len']:
                    for ewp_rate in para['ewp_rate']:
                        for ci_rate in para['ci_rate']:

                            name = str(ppl_num) + '_' + str(item_num) + '_' + str(test_len)+ '_'+str(ewp_rate) + '_' + str(ci_rate) + '_' + str(simu_id)
                            with open('../CAT_data/' + name, 'rb') as f:
                                data = self = pickle.loads(f.read())
                            f.close()

                            # calculate the standardized log response time
                            ltimes = data.rt
                            alpha = data.item_para['speed_a']
                            beta = data.item_para['speed_b']
                            sigmatausqInv = 1 / (0.02)
                            n = ltimes.shape[0]
                            resid = np.full(fill_value=np.nan, shape=ltimes.shape)
                            smasqb = sum(alpha * alpha * beta)
                            sumasq = sum(alpha * alpha)
                            for i in range(n):
                                wtdav = np.nansum(alpha * alpha * ltimes[i,])
                                mean = beta - (smasqb - wtdav - alpha * alpha * beta + alpha * alpha * ltimes[i,]) / (
                                        sigmatausqInv + sumasq - alpha * alpha)
                                var = 1 / (alpha * alpha) + 1 / (sigmatausqInv + sumasq - alpha * alpha)
                                resid[i,] = (ltimes[i,] - mean) / np.sqrt(var)

                            # flag examinees and items
                            self.res = {}
                            loc = np.where((resid<-1.96)|(resid>1.96))
                            ab_resp = pd.DataFrame(loc).transpose()
                            ab_resp.columns = ['ppl','item']
                            ab_ppl = ab_resp.groupby(['ppl']).count().reset_index()
                            self.res['comp_ppl_id'] = np.array(ab_ppl['ppl'][ab_ppl['item']>=np.percentile(np.array(ab_ppl['item']), 95)])
                            ab_item = ab_resp.groupby(['item']).count().reset_index()
                            self.res['comp_item_name']= np.array(ab_item['item'][ab_item['ppl'] >= np.percentile(np.array(ab_item['ppl']), 95)])

                            # detection result evaluation
                            eva_mid = evaluate(data, self)
                            for false_type in exp_default_para['false_type']:
                                cur_res = [simu_id, ppl_num, item_num, test_len, ewp_rate, ci_rate,
                                           0, 0, eva_mid[false_type]]
                                with open('result_compare/' + false_type + '.csv', 'a') as f:
                                    np.savetxt(f, np.array(cur_res).reshape(1, 9), delimiter=',')
                                f.close()


    return




if __name__ == '__main__':

    simu_ids = np.arange(30)
    main(simu_ids, para)
    main(simu_ids, para_null)
