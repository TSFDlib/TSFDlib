import pandas as pd
import numpy as np
import pandas_profiling
#import src.feature_extraction.utils.read_json as rj
import TSFDlib as tslib

def extract_features(sig, size, cfg):
    feat_val = None
    labels = None

    header = np.array(pd.read_csv('TSFdlib-master/tests/input_signal/Signal.txt', delimiter=',', header=None))[0, 1:]
    windows = [sig[i:i + size] for i in range(0, len(sig), size)]
    for wind_idx, wind_sig in enumerate(windows):
        row_idx, labels = tslib.feat_extract(cfg, wind_sig, str(header[0]))
        if wind_idx == 0:
            feat_val = row_idx
        else:
            feat_val = np.vstack((feat_val, row_idx))
    feat_val = np.array(feat_val)
    d = {str(lab): feat_val[:,idx] for idx, lab in enumerate(labels)}
    df = pd.DataFrame(data=d)
    df.to_csv('TSFdlib-master/tsfel/utils/Features.csv', sep=',', encoding='utf-8', index_label="Sample")

    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(outputfile="CorrelationReport.html")
    inp = str(input('Do you wish to remove correlated features? Enter y/n: '))
    if inp == 'y':
        reject = profile.get_rejected_variables(threshold=0.9)
        if reject == []:
            print('No features to remove')
        for rej in reject:
            print('Removing ' + str(rej))
            df = df.drop(rej, axis=1)

    return df
