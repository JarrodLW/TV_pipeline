import json
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

date = '24052021'
outputfile = '/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_' + date + '/TV_complex_reg_recons_32768.json'

with open(outputfile, 'r') as f:
    d = json.load(f)
print("Loaded previous datafile at " + dt.datetime.now().isoformat())

f.close()

d_40 = d['output_size=40']['lambda=' + '{:.1e}'.format(0.)]
d_80 = d['output_size=80']['lambda=' + '{:.1e}'.format(0.)]
d_120 = d['output_size=120']['lambda=' + '{:.1e}'.format(0.)]

rec_40 = np.asarray(d_40['recon'])
diff_40 = np.asarray(d_40['fourier_diff'])

rec_80 = np.asarray(d_80['recon'])
diff_80 = np.asarray(d_80['fourier_diff'])

rec_120 = np.asarray(d_120['recon'])
diff_120 = np.asarray(d_120['fourier_diff'])
