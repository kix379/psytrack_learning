import os
#import re
#from IPython.display import clear_output
import numpy as np
import pandas as pd
import scipy.io
import psytrack as psy

from construct_input import *
from plotting_functions import *


def perform_cross_validation(outData,length,hyper_guess,weights,optList,k,img_filename):
	#trim the data if you're performing cross validation
	new_D = psy.trim(outData, END=length)
	hyp, evd, wMode, hess_info = psy.hyperOpt(new_D, hyper_guess, weights, optList)

	#cross validation 
	xval_logli, xval_pL = psy.crossValidate(new_D, hyper_guess, weights, optList, F=k, seed=41)

	W_std=hess_info['W_std']

	#plotting the weights
	sep_plot(wMode, W_std, weights,[0,800], [-3.5,5], SPATH+img_filename+'.png')

	fig_perf_xval = psy.plot_performance(new_D, xval_pL=xval_pL)
	fig_bias_xval = psy.plot_bias(new_D, xval_pL=xval_pL)

	fig_perf_xval.savefig(SPATH+img_filename+'_'+str(k)+'fold_performance.png')
	fig_bias_xval.savefig(SPATH+img_filename+'_'+str(k)+'fold_bias.png')


# Set save path for all figures, decide whether to save permanently
SPATH = "C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Code_python\\Figures\\subject001\\"
datapath = 'C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Data\\BhartiData\\Priya'

#Load data
Bharti_data = scipy.io.loadmat(datapath + '\\sub_001_freeform.mat')

if not os.path.exists(SPATH):
    os.makedirs(SPATH)

CData = np.array(Bharti_data['CData'])
sub_data = pd.DataFrame()

sub_data['probe']        = CData[:,10] # -1: L, 1: R
sub_data['response']     = CData[:,11] #  1: Ch, 0: N-Ch, 5: response not recorded
sub_data['RT']           = CData[:,12] #  response time
# Additional fields :
sub_data["correct"] =   CData[:,13]
sub_data["answer"] =   CData[:,14]
sub_data["probed_Ch"] =  CData[:,17] + CData[:,18]
sub_data["unprobed_Ch"] =  CData[:,15] + CData[:,16]
sub_data["unprobed_Ch_L"] = CData[:,15]
sub_data["unprobed_Ch_R"] = CData[:,16]
sub_data["probed_Ch_L"] = CData[:,17]
sub_data["probed_Ch_R"] = CData[:,18]
sub_data["probe_L"] = CData[:,19]
sub_data["probe_R"] = CData[:,20]
sub_data["probL"] = CData[:,21] #probability of left side cue

# dump mistrials
indices = np.where(sub_data['response'] != 5)[0].tolist()
sub_data=sub_data.iloc[indices,:]        
sub_data["trial_num"]=indices
sub_data=sub_data.reset_index(drop=True)

# add previous response as a variable, comment this out if not used
"""
sub_data["prev_resp"]=np.nan
sub_data.loc[1:,"prev_resp"]=sub_data["response"][:-1].to_numpy()
# trim off the first trial
sub_data=sub_data.iloc[1:,:]
sub_data=sub_data.reset_index(drop=True)
"""

# Modify weights based on which function is used for constructing the input
outData, weights = getData_4(sub_data)
K = np.sum([weights[i] for i in weights.keys()])
print(outData, weights)

#Modify the initial sigma as required
hyper_guess = {
 'sigma'   : [2**5]*K,
 'sigInit' : 2**5,
 'sigDay'  : None
  }
optList = ['sigma']

#main function to get the weights
hyp, evd, wMode, hess_info = psy.hyperOpt(outData, hyper_guess, weights, optList)
W_std=hess_info['W_std']
# Uncomment to save interim result
#filename='data'
#dat = {'hyp' : hyp, 'evd' : evd, 'wMode' : wMode, 'W_std' : W_std,'weights' : weights, 'new_dat' : outData_1_freeform}
#np.savez_compressed(SPATH+filename, dat=dat)

#plotting the weights
img_filename='freeform_4_input_sigma_5'
sep_plot(wMode, W_std, weights,[0,800], [-30,30], SPATH+img_filename)

#perform_cross_validation(outData,700,hyper_guess,weights,optList,5,img_filename+'_trimmed_')

#plot with dprime, bcc, criterion
datapath = 'C:\\Users\\Cognition-Lab\\Documents\\Kruttika_files\\Data\\BhartiData\\Priya\\freeform\\'
feedback_data = scipy.io.loadmat(datapath + 'feedback_vals_window_50.mat')
feedback_vals = np.array(feedback_data['feedback_vals']);
left_dprime=feedback_vals[:,0];
left_c=feedback_vals[:,1];
left_bcc=feedback_vals[:,2];
right_dprime=feedback_vals[:,3];
right_c=feedback_vals[:,4];
right_bcc=feedback_vals[:,5];
trial_num=np.array(feedback_data['trial_num']);

params_1={
	'line_1_index':2,
	'line_2_index':3,
	'xlim': [0,800],
	'ylim': [-30,30]
}
params_2={
	'line_1':left_dprime,
	'line_2':right_dprime,
	'label_1':'dprime_left',
	'label_2':'dprime_right',
	'x-axis':trial_num,
	'ylim': [-1.5,4.5],
	'ylabel': 'd\''
}
plot_weights_behavior(wMode,weights,params_1,params_2,100,SPATH+img_filename+'_dprime_probed_ch.png')

params_1={
	'line_1_index':0,
	'line_2_index':1,
	'xlim': [0,800],
	'ylim': [-30,30]
}
params_2={
	'line_1':left_bcc,
	'line_2':right_bcc,
	'label_1':'bcc_left',
	'label_2':'bcc_right',
	'x-axis':trial_num,
	'ylim': [-2,2.5],
	'ylabel': 'bcc'
}
plot_weights_behavior(wMode,weights,params_1,params_2,100,SPATH+img_filename+'_bcc_probe.png')

params_2={
	'line_1':left_c,
	'line_2':right_c,
	'label_1':'criterion_left',
	'label_2':'criterion_right',
	'x-axis':trial_num,
	'ylim': [-2,2.5],
	'ylabel': 'criterion'
}
plot_weights_behavior(wMode,weights,params_1,params_2,100,SPATH+img_filename+'_criterion_probe.png')
