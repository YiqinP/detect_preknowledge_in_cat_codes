from parameter import para, para_null,exp_default_para
import numpy as np
import pickle
from experiment_follow_up import ExperimentFollowUp, AfterExperimentFollowUp
from experiment import AfterExperiment
from data_simulation_cat import Data_Simulartor_CAT

import pandas as pd
from os.path import exists

"""
This function impose the confidence score criterion on the detection result.
Change line 40 for adding the number of detected item criterion
"""

def main(simu_ids, para):

    for simu_id in simu_ids:
        print(simu_id)
        for ppl_num in para['ppl_num']:
            for item_num in para['item_num']:
                for test_len in para['test_len']:
                    for ewp_rate in para['ewp_rate']:
                        for ci_rate in para['ci_rate']:
                            for ab_cri in para['ab_cri']:
                                for iterat_times in para['iterat_times']:

                                    name = str(ppl_num) + '_' + str(item_num) + '_' + str(test_len) + '_' + str(
                                        ewp_rate) + '_' + str(ci_rate) + '_' + str(
                                        ab_cri) + '_' + str(iterat_times) + '_' + str(simu_id)
                                    with open('../study2/object/' + name, 'rb') as f:
                                        detect_res = pickle.loads(f.read())
                                    f.close()
                                    with open('object/' + name, 'rb') as f:
                                        after_experiment = pickle.loads(f.read())
                                    f.close()

                                    # exp_default_para['index_cri']=1.15
                                    if after_experiment.ppl_ratio<exp_default_para['index_cri']:
                                            # and len(detect_res.res_mid_1['comp_ppl_id'])>exp_default_para['num_cri']*ppl_num:
                                        detect_res.eva_mid['false_posi_ppl'] = 0
                                        detect_res.eva_mid['false_neg_ppl'] = 1 if ewp_rate!=0 else np.nan
                                        detect_res.eva_mid['precision_ppl'] = np.nan
                                        detect_res.eva_mid['false_posi_item'] = 0
                                        detect_res.eva_mid['false_neg_item'] = 1 if ci_rate!=0 else np.nan
                                        detect_res.eva_mid['precision_item'] = np.nan

                                    for false_type in exp_default_para['false_type']:
                                        cur_res = [simu_id, ppl_num, item_num, test_len, ewp_rate, ci_rate,
                                                   iterat_times, ab_cri, detect_res.eva_mid[false_type]]
                                        with open('result/' + false_type + '.csv', 'a') as f:
                                            np.savetxt(f, np.array(cur_res).reshape(1, 9), delimiter=',')
                                        f.close()
    return



if __name__ == '__main__':

    simu_ids = np.arange(30)
    main(simu_ids, para)
    main(simu_ids, para_null)
