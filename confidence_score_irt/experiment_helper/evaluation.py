import pandas as pd
import numpy as np



def evaluate(data, detect_res):
    comp = pd.DataFrame(data.item_pre, columns=['ppl', 'item'])
    data.comp= {'comp_item_name':np.array(list(set(comp['item']))), 'comp_ppl_id':np.array(list(set(comp['ppl'])))}

    eval_res = {}

    detect_ewp = len(detect_res.res['comp_ppl_id'])
    ewp = len(data.comp['comp_ppl_id'])
    corr_ewp = len(np.intersect1d(detect_res.res['comp_ppl_id'], data.comp['comp_ppl_id']))
    eval_res['false_posi_ppl'] = (detect_ewp-corr_ewp)/(data.ppl_num - ewp)
    eval_res['false_neg_ppl'] =1-corr_ewp/ewp if ewp!=0 else np.nan
    eval_res['precision_ppl'] = corr_ewp/detect_ewp if detect_ewp!=0 else np.nan


    detect_ci = len(detect_res.res['comp_item_name'])
    ci = len(data.comp['comp_item_name'])
    corr_ci = len(np.intersect1d(detect_res.res['comp_item_name'], data.comp['comp_item_name']))
    eval_res['false_posi_item'] = (detect_ci-corr_ci)/(data.item_num-ci)
    eval_res['false_neg_item'] = 1-corr_ci/ci if ci!=0 else np.nan
    eval_res['precision_item'] = corr_ci/detect_ci if detect_ci!=0 else np.nan

    print( np.round([eval_res['false_neg_item'], eval_res['false_neg_ppl'], eval_res['false_posi_item'], eval_res['false_posi_ppl']], 3))
    return eval_res
