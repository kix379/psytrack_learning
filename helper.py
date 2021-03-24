import psytrack as psy
from plotting_functions import *


def perform_cross_validation(outData,length,hyper_guess,weights,optList,k,num_variables,img_filename,title,xlim_val,ylim_val):
	#trim the data if you're performing cross validation
	new_D = psy.trim(outData, END=length)
	hyp, evd, wMode, hess_info = psy.hyperOpt(new_D, hyper_guess, weights, optList)

	#cross validation 
	xval_logli, xval_pL = psy.crossValidate(new_D, hyper_guess, weights, optList, F=k, seed=41)

	W_std=hess_info['W_std']

	#plotting the weights
	if num_variables=='4' or num_variables=='6':
		sep_plot(wMode, W_std, weights,[0,xlim_val], [-ylim_val,ylim_val], img_filename+'.png',title)
	else:
		standard_plot(wMode, W_std, weights,[0,xlim_val], [-ylim_val,ylim_val], img_filename+'.png',title)
	fig_perf_xval = psy.plot_performance(new_D, xval_pL=xval_pL)
	fig_bias_xval = psy.plot_bias(new_D, xval_pL=xval_pL)

	fig_perf_xval.savefig(img_filename+'_'+str(k)+'fold_performance.png')
	fig_bias_xval.savefig(img_filename+'_'+str(k)+'fold_bias.png')
	return xval_logli

def compute_ic(evd,hyp,wMode,n):
	T=wMode[0].size     	#number of trials
	E=evd 					#log-evidence or log-likelihood
	K=len(hyp['sigma']) 	#number of hyperparameters
	print('T=%d, E=%f, K=%d' % (T,E,K))
	aic= (-2/T * E) + (2 * K/T)
	bic= (-2*E) + (np.log(T) * K)
	return round(aic,n), round(bic,n)
