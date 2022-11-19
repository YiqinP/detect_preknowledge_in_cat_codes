from parameter import para, para_null
import numpy as np
import pickle
from experiment_follow_up import ExperimentFollowUp, AfterExperimentFollowUp
from experiment import AfterExperiment
from os.path import exists
from data_simulation_cat import Data_Simulartor_CAT
import pandas as pd
import os

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

                                    name = str(ppl_num) + '_' + str(item_num) + '_' + str(test_len)+ '_'+str(ewp_rate) + '_' + str(ci_rate) + '_' + str(simu_id)
                                    with open('../CAT_data/' + name, 'rb') as f:
                                        data = pickle.loads(f.read())
                                    f.close()

                                    name = str(ppl_num) + '_' + str(item_num) + '_' + str(test_len) + '_' + str(ewp_rate) + '_' + str(ci_rate)  + '_' +str(
                                        ab_cri)+'_'+str(iterat_times)+'_'+str(simu_id)
                                    with open('../detection/object/' + name, 'rb') as f:
                                        detect_res = pickle.loads(f.read())
                                    f.close()

                                    experiment = ExperimentFollowUp(data, detect_res)
                                    experiment.process()
                                    after_experiment = AfterExperimentFollowUp(experiment)

                                    with open('object/' + name, 'wb') as f:
                                        str_ = pickle.dumps(after_experiment)
                                        f.write(str_)
                                    f.close()

    return



if __name__ == '__main__':

    simu_ids = np.arange(30)
    main(simu_ids, para)
    main(simu_ids, para_null)
