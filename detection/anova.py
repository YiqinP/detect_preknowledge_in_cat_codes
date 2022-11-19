import pandas as pd
import statsmodels.api as sm # version 0.12.2
from statsmodels.formula.api import ols
import scipy.stats as stats

def moedl_fit(type):

    tmp = pd.read_csv('result_mid/'+type+'.csv', header=None)
    tmp.columns=["simu_id", "ppl_num", "item_num",  "test_len", "ewp_rate", "ci_rate", "iterat_times", "ab_cri", "val"]
    d_m= tmp[(tmp['iterat_times']==30) | (tmp['iterat_times']==60) ]
    model = ols('val ~ C(ewp_rate)*C(ci_rate)*C(ab_cri)*C(iterat_times)', data=d_m).fit()
    anova_table = sm.stats.anova_lm(model,typ=3)
    anova_table['partial_eta_sqr']=anova_table['sum_sq']/(anova_table['sum_sq']['Residual']+anova_table['sum_sq'])
    anova_table.to_csv('anova/'+type+".csv",float_format='%.3f')

types = {'false_neg_item','false_posi_item','false_neg_ppl','false_posi_ppl','precision_ppl','precision_item'}
for type in types:
    moedl_fit(type)

type = 'false_posi_item'

# fig = sm.qqplot(data, line='45')
# plt.show()
#
# fig, ax = plt.subplots(figsize=(10,4))
# import matplotlib.pyplot as plt
# for key, grp in d_m.groupby(by=["ewp_rate", "ci_rate", "iterat_times", "ab_cri"]):
#     a=1
#     plt.hist(grp['val'])
#
# plt.hist([1,2,3,3,2,1])
#     plt.show()
#
#     sm.qqplot(grp['val'], line='45')
#
# ax.legend()
# plt.show()
#
#
#
# import pandas as pd
# import pingouin as pg
# df = pd.DataFrame({
#    'white': {0: 10, 1: 8, 2: 7, 3: 9, 4: 7, 5: 4, 6: 5, 7: 6, 8: 5, 9: 10, 10: 4, 11: 7},
#    'red': {0: 7, 1: 5, 2: 8, 3: 6, 4: 5, 5: 7, 6: 9, 7: 6, 8: 4, 9: 6, 10: 7, 11: 3},
#    'rose': {0: 8, 1: 5, 2: 6, 3: 4, 4: 7, 5: 5, 6: 3, 7: 7, 8: 6, 9: 4, 10: 4, 11: 3}})
# pg.friedman(df)
#
# import scipy.stats as ss
# ss.friedmanchisquare(group1,group2,group3)