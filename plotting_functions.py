from matplotlib import pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

def standard_plot(wMode, W_std, weights, xlim, ylim, filename):
	plt.figure()
	ax = plt.gca()
	for i in range(wMode.shape[0]):
		color = next(ax._get_lines.prop_cycler)['color']
		plt.plot(wMode[i], lw=1, linestyle='-', alpha=0.85, zorder=2*i+1, color=color)
		plt.fill_between(np.arange(len(wMode[i])), wMode[i] - 2 * W_std[i], wMode[i] + 2 * W_std[i], facecolor=color, alpha=0.2, zorder=2*i)
	plt.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)
	plt.xticks(50*np.arange(0,xlim[1]/50 + 1))
	plt.yticks(np.arange(-2,3,2))
	plt.xlim(xlim[0],xlim[1])
	plt.ylim(ylim[0],ylim[1])
	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['top'].set_visible(False)

	plt.xlabel("trials")
	plt.ylabel("weights")
	legendNames=  sorted(weights)
	plt.legend(legendNames, loc ="lower right") 
	plt.savefig(filename+'.png')
	plt.show()

def sep_plot(wMode, W_std, weights, xlim, ylim, filename):
	
	legendNames=  sorted(weights)
	plt.figure(figsize=(8,12))
	ax = plt.gca()
	for j in range(1,4):
		plt.subplot(3,1,j)
		first=2*(j-1)
		for i in [first,first+1]:
			color = next(ax._get_lines.prop_cycler)['color']
			plt.plot(wMode[i], c=color, lw=1, linestyle='-', alpha=0.85, zorder=2*i+1, label=legendNames[i])
			plt.fill_between(np.arange(len(wMode[i])), wMode[i] - 2 * W_std[i], wMode[i] + 2 * W_std[i],
                     facecolor=color, alpha=0.2, zorder=2*i)
			plt.axhline(0, color="black", linestyle="--", lw=0.5, alpha=0.5, zorder=0)
			plt.xticks(50*np.arange(0,xlim[1]/50 + 1))
			plt.yticks(np.arange(-2,3,2))
			plt.xlim(xlim[0],xlim[1])
			plt.ylim(ylim[0],ylim[1])
			plt.gca().spines['right'].set_visible(False)
			plt.gca().spines['top'].set_visible(False)
			plt.xlabel("trials")
			plt.ylabel("weights")
			plt.legend(loc ="lower right") 

	plt.savefig(filename+'.png')
	plt.show()
