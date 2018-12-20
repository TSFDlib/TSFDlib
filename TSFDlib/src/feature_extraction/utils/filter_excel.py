import openpyxl as xl
import json
import pickle
import numpy as np
import read_json as rj
import ast


def extract_sheet():
    wb = xl.load_workbook('../../data/Configuration Manager.xlsx')
    sheet = wb['Complete']
    FEATURES_JSON = '../../data/features.json'
    DEFAULT = {'use': 'yes', 'metric': 'euclidean', 'free parameters': '', 'number of features': 1, 'parameters': ''}
    DICTIONARY = rj.compute_dictionary(FEATURES_JSON, DEFAULT)

    list_of_features = [str(i.value) for i in sheet['B']][4:-6]
    use_or_not = [True for i in list_of_features]
    filter = sheet.auto_filter.filterColumn
    if filter:
        init_feat_idx = str(filter[0]).index('filter=[')
        end_feat_idx = str(filter[0])[init_feat_idx:].index(']')
        feat = str(filter[0])[init_feat_idx+7:init_feat_idx+end_feat_idx].split(",")
        _f = []
        for f in feat:
            _f.append(f[3:-1])

        use_or_not = []
        KEYS = ['Statistical', 'Temporal', 'Spectral']
        if _f[0] in DICTIONARY.keys():
            for _key in KEYS:
                for count in DICTIONARY[_key].keys():
                    use_or_not.append(True if str(_key) in _f else False)
        else:
            use_or_not = [True if i in _f else False for i in list_of_features]

    len_stat = len(DICTIONARY['Statistical'].keys())
    len_temp = len(DICTIONARY['Temporal'].keys())
    len_spec = len(DICTIONARY['Spectral'].keys())

    for i in range(len_stat):
        if use_or_not[i]:
            if list_of_features[i] == 'Histogram':
                val = sheet['E' + str(5+i)].value
                DICTIONARY['Statistical'][list_of_features[i]]['free parameters'] = {'nbins': [ast.literal_eval(val)['nbins']], "r": [ast.literal_eval(val)['r']]}
            DICTIONARY['Statistical'][list_of_features[i]]['use'] = 'yes'
            print list_of_features[i]

    for i in range(len_temp):
        if use_or_not[i+len_stat]:
            DICTIONARY['Temporal'][list_of_features[i+len_stat]]['use'] = 'yes'
            print list_of_features[i+len_stat]

    for i in range(len_spec):
        if use_or_not[i+len_stat+len_temp]:
            val = sheet['E' + str(5 + i+len_stat+len_temp)].value
            if not val:
                DICTIONARY['Spectral'][list_of_features[i+len_stat+len_temp]]['parameters'] = ''
            else:
                DICTIONARY['Spectral'][list_of_features[i+len_stat+len_temp]]['parameters'] = str(ast.literal_eval(val)['fs'])
            DICTIONARY['Spectral'][list_of_features[i+len_stat+len_temp]]['use'] = 'yes'
            print list_of_features[i+len_stat+len_temp]

    return DICTIONARY

#extract_sheet()
