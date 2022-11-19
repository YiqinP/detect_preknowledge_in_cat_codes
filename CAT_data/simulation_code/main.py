from parameter import para, para_null
import numpy as np
import pickle
from data_simulation_cat import Data_Simulartor_CAT

def main(simu_ids, para):
    for simu_id in simu_ids:
        for ppl_num in para['ppl_num']:
            for item_num in para['item_num']:
                for test_len in para['test_len']:
                    for ewp_rate in para['ewp_rate']:
                        for ci_rate in para['ci_rate']:
                            data = Data_Simulartor_CAT(ppl_num=ppl_num, item_num=item_num, test_len=test_len, ewp_rate=ewp_rate, ci_rate=ci_rate)
                            data.simulate()
                            name = '../'+str(ppl_num) + '_' + str(item_num) + '_' + str(test_len)+ '_'+str(ewp_rate) + '_' + str(ci_rate) + '_' + str(simu_id)
                            with open(name, 'wb') as f:
                                str_ = pickle.dumps(data)
                                f.write(str_)
                            f.close()


if __name__ == '__main__':

    simu_ids = np.arange(30)
    main(simu_ids, para)
    main(simu_ids, para_null)