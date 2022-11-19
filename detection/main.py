from parameter import para, para_null, exp_default_para
import numpy as np
import pickle
from experiment import Experiment,AfterExperiment
from os.path import exists
from data_simulation_cat import Data_Simulartor_CAT
import pandas as pd


def main(simu_ids, para):

    result_mid = {'false_posi_item': list(), 'false_neg_item': list(), 'false_posi_ppl': list(), 'false_neg_ppl': list(),
              'precision_item': list(), 'precision_ppl':list()}

    for simu_id in simu_ids:
        print(simu_id)
        for ppl_num in para['ppl_num']:
            for item_num in para['item_num']:
                for test_len in para['test_len']:
                    for ewp_rate in para['ewp_rate']:
                        for ci_rate in para['ci_rate']:
                            name = '../CAT_data/' + str(ppl_num) + '_' + str(item_num) + '_' + str(
                                test_len) + '_' + str(
                                ewp_rate) + '_' + str(ci_rate)  + '_' + str(simu_id)
                            with open(name, 'rb') as f:
                                data = pickle.loads(f.read())
                            f.close()


                            for ab_cri in para['ab_cri']:
                                experiment = Experiment(ppl_num=ppl_num, item_num=item_num, test_len=test_len,
                                                        ewp_rate=ewp_rate, ci_rate=ci_rate, iterat_times=0, ab_cri=ab_cri, data=data)
                                for iterat_times in para['iterat_times']:
                                    experiment.iterat_times = iterat_times

                                    # conduct detection
                                    experiment.process()
                                    after_experiment = AfterExperiment(experiment)

                                    loc = 'object/' + str(ppl_num) + '_' + str(item_num) + '_' + str(
                                        test_len) + '_' + str(ewp_rate) + '_' + str(ci_rate)  + '_' +str(
                                        ab_cri)+'_'+str(iterat_times)+'_'+str(simu_id)
                                    with open(loc, 'wb') as f:
                                        str_ = pickle.dumps(after_experiment)
                                        f.write(str_)
                                    f.close()

                                    for false_type in exp_default_para['false_type']:
                                        cur_res = [simu_id, ppl_num, item_num, test_len, ewp_rate, ci_rate,
                                                   iterat_times, ab_cri, experiment.eva_mid[false_type]]
                                        result_mid[false_type].append(cur_res)
                                        with open('result_mid/' + false_type + '.csv', 'a') as f:
                                            np.savetxt(f, np.array(cur_res).reshape(1, 9), delimiter=',')
                                        f.close()


    return




if __name__ == '__main__':

    simu_ids = np.arange(30)
    main(simu_ids, para_null)
    main(simu_ids, para)
