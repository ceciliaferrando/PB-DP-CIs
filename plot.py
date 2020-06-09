import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib import transforms
import pandas as pd
import seaborn as sns
from scipy.stats import norm 

#####################


def plotresults(path, d, mode, epsilon, small=False):
    #plots grid of coverage results with varying N and fixed epsilon

    xlimit = 0.0

    i_dict = {0: "", 1: "_FISHER", 2: "_FISHERNP"}
    #j_dict = {0: "50", 1: "100", 2:"200"}
    #j_dict = {0: "10", 1: "50", 2: "100", 3:"500", 4:"1000", 5:"5000", 6:"10000"}
    j_dict = {0:"50", 1: "100", 2:"500", 3:"1000", 4:"5000", 5:"10000"}
    if small:
        j_dict = {0: "5", 1: "10", 2:"50", 3:"100"}
    colors = {"": 'blue', "_FISHER":'orange', "_FISHERNP":'red', "_BASIC": 'red'}
    titles = {"poisson": "Poisson", "gaussian": "Gaussian, known variance", "gaussian2": "Gaussian, unknown variance", "gamma": "Gamma"}
    
    list_ci_levels = [50, 60, 70, 80, 90, 95, 99]
    
    figuresize = (9.5, 6.5) if small==True else (13.5, 6.5)

    fig, axs = plt.subplots(len(i_dict), len(j_dict), figsize=figuresize, sharex=True, sharey=True)   
    
    for j in range(len(axs[0])):
        axs[0,j].set_title("n=" + j_dict[j])
    
    for i in range(len(i_dict)): 
        for j in range(len(j_dict)): 
        
            x = np.linspace(xlimit,1,100)
            y = x
            y2 = (1-x)/2
           
            axs[i,j].plot(x, y, 'grey', label='perfect coverage')
            
            results = np.load(path+ "/results_"+ d +"_N" + j_dict[j] +"_epsilon" + epsilon +"_" + mode + i_dict[i] +".npy", allow_pickle=True)
            results = results[:,0,]
            axs[i,j].scatter([ci/100 for ci in list_ci_levels], results, marker='o',s=35, color=colors[i_dict[i]], label='output coverage')

    
    fig.text(0.5, 0.04, 'Nominal coverage', size=20, ha='center', va='center')
    
    if small:
        fig.text(0.035, 0.5, 'Observed coverage', size=20, ha='center', va='center', rotation='vertical')
    else:
        fig.text(0.055, 0.5, 'Observed coverage', size=20, ha='center', va='center', rotation='vertical')
        
    plt.setp(axs[0, 0], ylabel='Private PB')
    plt.setp(axs[1, 0], ylabel='Private FI')
    #plt.setp(axs[2, 0], ylabel='public PB')
    plt.setp(axs[2, 0], ylabel='Public FI')
    plt.xlim((xlimit, 1.0))
    plt.ylim((xlimit, 1.0))
    plt.xticks([0, 0.5, 1], ["0", "0.5", "1"])
    plt.yticks([0, 0.5, 1], ["0", "0.5", "1"])
    
    #axs.flatten()[-4].legend(loc='upper center', bbox_to_anchor=(3.5, 3.9), ncol=3)
    #fig.legend(loc='upper center', fancybox=True, shadow=True)
    
    if small:
        plt.savefig(path +"/SMALLcoverage" + d + epsilon.replace(".", "") + mode + ".pdf")
    else:
        plt.savefig(path +"/coverage" + d + epsilon.replace(".", "") + mode + ".pdf")
    
    plt.close()
    
def plotresultsEps(path):
    #plots grid of coverage results with varying epsilon and fixed N

    xlimit = 0.0

    i_dict = {0: "", 1: "_FISHER", 2: "_FISHERNP"}
    #j_dict = {0: "10", 1: "50", 2: "100", 3:"500", 4:"1000"}
    #j_dict = {0: "10", 1: "50", 2: "100", 3:"500", 4:"1000", 5:"5000", 6:"10000"}
    j_dict = {0:"50", 1: "100", 2:"500", 3:"1000", 4:"5000", 5:"10000"}
    #j_dict = {0: "2", 1: "5", 2: "10", 3:"50"}
    colors = {"": 'blue', "_FISHER":'orange', "_FISHERNP":'red', "_BASIC": 'red'}
    titles = {"poisson": "Poisson", "gaussian": "Gaussian, known variance", "gaussian2": "Gaussian, unknown variance", "gamma": "Gamma"}
    
    list_ci_levels = [50, 60, 70, 80, 90, 95, 99]
    
    epsilons = ["0.1", "0.2", "0.3", "0.5", "0.7", "1.0"] 
    
    fig, axs = plt.subplots(3, len(epsilons), figsize=(13.5, 6.5), sharex=True, sharey=True)
    
    for j in range(len(axs[0])):
        axs[0,j].set_title(r'$\epsilon$=' + epsilons[j])

    fig.suptitle("n=100")
    
    for i in range(3): #our method, naive method
        for j,eps in enumerate(epsilons): 
    
            results = np.load(path+ "results_gamma" + "_N100" +"_epsilon" + eps + "_empirical" + i_dict[i] +".npy", allow_pickle=True)
            results = results[:,0]
        
            axs[i,j].scatter([ci/100 for ci in list_ci_levels], results, marker='o',s=35, color=colors[i_dict[i]], label='coverage')
            
            x = np.linspace(xlimit,1,100)
            y = x
            axs[i,j].plot(x, y, 'grey', label='y=x')
            
    fig.text(0.5, 0.04, 'Nominal coverage', size=20, ha='center', va='center')
    fig.text(0.055, 0.5, 'Observed coverage', size=20, ha='center', va='center', rotation='vertical')
        
    plt.setp(axs[0, 0], ylabel='Private PB')
    plt.setp(axs[1, 0], ylabel='Private FI')
    #plt.setp(axs[2, 0], ylabel='public PB')
    plt.setp(axs[2, 0], ylabel='Public FI')
    plt.xlim((xlimit, 1.0))
    plt.ylim((xlimit, 1.0))
    plt.xticks([0, 0.5, 1], ["0", "0.5", "1"])
    plt.yticks([0, 0.5, 1], ["0", "0.5", "1"])
    
    #axs.flatten()[-4].legend(loc='upper center', bbox_to_anchor=(3.5, 3.9), ncol=3)
    #fig.legend(loc='upper center', fancybox=True, shadow=True)
    
    plt.savefig(path +"/coverageeps"+ ".pdf")
    
    plt.close()
    
def plotresultsUpLow(path, d, mode, epsilon):
    #plots upper/lower tail failure rate with varying N and fixed epsilon

    xlimit = 0.0

    i_dict = {0: "", 1: "_FISHER", 2: "_FISHERNP"}
    #j_dict = {0: "10", 1: "50", 2: "100", 3:"500", 4:"1000"}
    #j_dict = {0: "10", 1: "50", 2: "100", 3:"500", 4:"1000", 5:"5000", 6:"10000"}
    #j_dict = {0:"50", 1: "100", 2:"500", 3:"1000", 4:"5000", 5:"10000"}
    j_dict = {0:"50", 1: "100", 2:"500"}
    #j_dict = {0: "2", 1: "5", 2: "10", 3:"50"}
    colorsUp = {"": 'blue', "_FISHER":'orange', "_FISHERNP":'red', "_BASIC": 'red'}
    colorsLow = {"": 'blue', "_FISHER":'orange', "_FISHERNP":'red', "_BASIC": 'red'}
    titles = {"poisson": "Poisson", "gaussian": "Gaussian, known variance", "gaussian2": "Gaussian, unknown variance", "gamma": "Gamma"}
    
    list_ci_levels = [50, 60, 70, 80, 90, 95, 99]

        
    fig, axs = plt.subplots(len(i_dict), len(j_dict), figsize=(13.5, 6.5), sharex=True, sharey=True)
    
    for j in range(len(axs[0])):
        axs[0,j].set_title("n=" + j_dict[j])

    
    for i in range(len(i_dict)): 
        for j in range(len(j_dict)): 
        
            x = np.linspace(xlimit,1,100)
            y2 = (1-x)/2

            axs[i,j].plot(x, y2, 'grey')

            uf = np.load(path+"/upperfailures_"+d+"_N" + j_dict[j] +"_epsilon" + epsilon +"_" + mode + i_dict[i] +".npy", allow_pickle=True)
            uf_rate = [int(u)/2000 for u in uf]
            lf= np.load(path+"/lowerfailures_"+d+"_N" + j_dict[j] +"_epsilon" + epsilon +"_" + mode + i_dict[i] +".npy", allow_pickle=True)
            lf_rate = [int(u)/2000 for u in lf]
            
            axs[i,j].scatter([ci/100 for ci in list_ci_levels], uf_rate, marker='x',s=40, color='limegreen', label='upper tail failure rate')
            axs[i,j].scatter([ci/100 for ci in list_ci_levels], lf_rate, marker='x',s=40, color='mediumvioletred', label='lower tail failure rate')

    
    fig.text(0.5, 0.04, 'Nominal coverage', size=20, ha='center', va='center')
    fig.text(0.055, 0.5, 'Rate of failure', size=20, ha='center', va='center', rotation='vertical')
        
    plt.setp(axs[0, 0], ylabel='Private PB')
    plt.setp(axs[1, 0], ylabel='Private FI')
    #plt.setp(axs[2, 0], ylabel='public PB')
    plt.setp(axs[2, 0], ylabel='Public FI')
    plt.xlim((xlimit, 1.0))
    plt.ylim((xlimit, 1.0))
    plt.xticks([0, 0.5, 1], ["0", "0.5", "1"])
    plt.yticks([0, 0.5, 1], ["0", "0.5", "1"])
    
    axs.flatten()[-4].legend(loc='upper center', bbox_to_anchor=(3.4, 3.96), ncol=3)
    #fig.legend(loc='upper center', fancybox=True, shadow=True)
    
    plt.savefig(path +"/failrate" + d + epsilon.replace(".", "") + mode + ".pdf")
    
    plt.close()
    

def draw_dodge(*args, **kwargs):
    
    # based on answers to https://stackoverflow.com/questions/50195997/how-to-add-axis-offset-in-matplotlib-plot
    func = args[0]
    dodge = kwargs.pop("dodge", 0)
    ax = kwargs.pop("ax", plt.gca())
    trans = ax.transData  + transforms.ScaledTranslation(dodge/72., 0, ax.figure.dpi_scale_trans)
    art = func(*args[1:], **kwargs)
    
    def iterate(art):
        if hasattr(art, '__iter__'):
            for obj in art:
                iterate(obj)
        else:
            art.set_transform(trans)
    iterate(art)
    
    return art
    
    
def plotIntervals(path, dist, mode, epsilon, small=False):
    #plots CIs with varying N and fixed epsilon
    
    #Ns = [50, 100, 200, 500, 1000, 5000, 10000]
    Ns = [50, 100, 500, 1000, 5000, 10000]
    #Ns = [2782]
    if small:
        Ns = [5, 10, 50, 100]
    #Ns = [50]
    true_theta = 4.1692107471768525
    #true_theta = 0.5
    list_ci_levels = [50, 60, 70, 80, 90, 95, 99]
    z_values = {50: 0.674, 60: 0.841, 70: 1.036, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
    
    methods = ["", "_FISHER", "_FISHERNP"]
    #methods = ["", "_BASIC", "_FISHER", "_FISHERNP"]
    colors = {"": 'blue', "_FISHER":'orange', "_FISHERNP":'red', "_BASIC": 'magenta'}
    offsets = {"": -0.80/4,  "_BASIC": -0.40/4, "_FISHER":+0.00/4, "_FISHERNP":+0.80/4}
    labels = {"": 'private PB', "_FISHER":'private FI', "_BASIC": 'public PB', "_FISHERNP":'public FI'}
    titles = {"poisson": "Poisson", "gaussian": "Gaussian, known variance", "gaussian2": "Gaussian, unknown variance", "gamma": "Gamma"}
    linestyles = {"": '-', "_FISHER":'-.', "_FISHERNP":'red', "_BASIC": 'magenta'}
    ms = {}
    sds = {}
    widths = {}
    intervals = []
    intervals_FI = []
    intervals_FI_np = []
    
    
    #extract means and stds to compute average intervals
    for jj, method in enumerate(methods):
        means, stdvs = [], []
        for N in Ns:
            results = np.load(path+ "/results_"+ dist +"_N" + str(N) +"_epsilon" + epsilon +"_" +mode + method +".npy", allow_pickle=True)
            m, std = results[-2,1,], results[-2,2,]   #95% CIs
            means.append(m)
            stdvs.append(std)
            #stdvs.append(std*np.sqrt(N))
        ms[method] = np.array(means)
        sds[method] = np.array(stdvs)
        
    for N in Ns:
        CI = np.load(path+ "/widths_"+ dist +"_N" + str(N) +"_epsilon" + epsilon +"_" +mode +".npy", allow_pickle=True)[-2]
        intervals.append(CI)
        
        CI_FI = np.load(path+ "/widths_"+ dist +"_N" + str(N) +"_epsilon" + epsilon +"_" +mode +"_FISHER.npy", allow_pickle=True)[-2]
        intervals_FI.append(CI_FI)
        
        CI_FI_np = np.load(path+ "/widths_"+ dist +"_N" + str(N) +"_epsilon" + epsilon +"_" +mode +"_FISHERNP.npy", allow_pickle=True)[-2]
        intervals_FI_np.append(CI_FI_np)
    
    
    #plot
    Dodge = np.arange(len(methods),dtype=float)*10
    Dodge -= Dodge.mean()   
    
    fig, ax = plt.subplots()
    plt.axhline(y=true_theta, color='black', linestyle=':', label='true parameter')
    for i,method in enumerate(methods):
        
        theta = ms[method]
        print(theta)
        print(intervals)
        
        if method == "":
            err_lo = [np.abs(theta[n] - intervals[n][0]) for n in range(len(theta))]
            err_hi = [np.abs(intervals[n][1] - theta[n]) for n in range(len(theta))]
            err = np.vstack((err_lo, err_hi))
            
        if method == "_FISHER":
            err_lo = [np.abs(theta[n] - intervals_FI[n][0]) for n in range(len(theta))]
            err_hi = [np.abs(intervals_FI[n][1] - theta[n]) for n in range(len(theta))]
            err = np.vstack((err_lo, err_hi))
            
        if method == "_FISHERNP":
            err_lo = [np.abs(theta[n] - intervals_FI_np[n][0]) for n in range(len(theta))]
            err_hi = [np.abs(intervals_FI_np[n][1] - theta[n]) for n in range(len(theta))]
            err = np.vstack((err_lo, err_hi))
            
            
        dg = draw_dodge(ax.errorbar, ["n=" + str(N) for N in Ns], ms[method], yerr = err, ax=ax, 
                            dodge=Dodge[i], marker="o", markersize=4, ls='none', capsize=2, elinewidth=2, markeredgewidth=1.5, color=colors[method], label=labels[method]) 
        #dg[-1][0].set_linestyle('--')
    

    plt.legend()
    plt.title("95% confidence intervals")
    plt.savefig(path+"/95CI" + dist + epsilon.replace(".","") + mode +".pdf") 


    
def plotIntervalsEps(path, dist, mode):
    #plots CIs with fixed N and varying epsilon
    
    #Ns = [50, 100, 200, 500, 1000, 5000, 10000]
    Ns = [50, 100, 500, 1000, 5000, 10000]
    #Ns = [2782]
    #Ns = [5, 10, 50, 100]
    #Ns = [50, 100, 500, 1000]
    true_theta = 4.1692107471768525
    #true_theta = 0.5
    list_ci_levels = [50, 60, 70, 80, 90, 95, 99]
    z_values = {50: 0.674, 60: 0.841, 70: 1.036, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
    
    epsilons = ["0.1", "0.2", "0.3", "0.5", "0.7", "1.0"] 
    
    methods = ["", "_FISHER", "_FISHERNP"]
    #methods = ["", "_BASIC", "_FISHER", "_FISHERNP"]
    colors = {"": 'blue', "_FISHER":'orange', "_FISHERNP":'red', "_BASIC": 'magenta'}
    offsets = {"": -0.80/4,  "_BASIC": -0.40/4, "_FISHER":+0.00/4, "_FISHERNP":+0.80/4}
    labels = {"": 'private PB', "_FISHER":'private FI', "_BASIC": 'public PB', "_FISHERNP":'public FI'}
    titles = {"poisson": "Poisson", "gaussian": "Gaussian, known variance", "gaussian2": "Gaussian, unknown variance", "gamma": "Gamma"}
    linestyles = {"": '-', "_FISHER":'-.', "_FISHERNP":'red', "_BASIC": 'magenta'}
    ms = {}
    sds = {}
    widths = {}
    intervals = []
    intervals_FI = []
    intervals_FI_np = []
    
    
    #extract means and stds to compute average intervals
    for jj, method in enumerate(methods):
        means, stdvs = [], []
        for eps in epsilons:
            results = np.load(path+ "/results_"+ dist +"_N100" +"_epsilon" + eps +"_" +mode + method +".npy", allow_pickle=True)
            m, std = results[-2,1,], results[-2,2,]   #95% CIs
            means.append(m)
            stdvs.append(std)
        ms[method] = np.array(means)
        sds[method] = np.array(stdvs)
        
    for eps in epsilons:
        CI = np.load(path+ "/widths_"+ dist +"_N100_epsilon" + eps +"_" +mode +".npy", allow_pickle=True)[-2]
        intervals.append(CI)
        
        CI_FI = np.load(path+ "/widths_"+ dist +"_N100_epsilon" + eps +"_" +mode +"_FISHER.npy", allow_pickle=True)[-2]
        intervals_FI.append(CI_FI)
        
        CI_FI_np = np.load(path+ "/widths_"+ dist +"_N100_epsilon" + eps +"_" +mode +"_FISHERNP.npy", allow_pickle=True)[-2]
        intervals_FI_np.append(CI_FI_np)
    
    
    #plot
    Dodge = np.arange(len(methods),dtype=float)*10
    Dodge -= Dodge.mean()   
    
    fig, ax = plt.subplots()
    plt.axhline(y=true_theta, color='black', linestyle=':', label='true parameter')
    for i,method in enumerate(methods):
        
        theta = ms[method]
        print(theta)
        print(intervals)
        
        if method == "":
            err_lo = [np.abs(theta[n] - intervals[n][0]) for n in range(len(theta))]
            err_hi = [np.abs(intervals[n][1] - theta[n]) for n in range(len(theta))]
            err = np.vstack((err_lo, err_hi))
            
        if method == "_FISHER":
            err_lo = [np.abs(theta[n] - intervals_FI[n][0]) for n in range(len(theta))]
            err_hi = [np.abs(intervals_FI[n][1] - theta[n]) for n in range(len(theta))]
            err = np.vstack((err_lo, err_hi))
            
        if method == "_FISHERNP":
            err_lo = [np.abs(theta[n] - intervals_FI_np[n][0]) for n in range(len(theta))]
            err_hi = [np.abs(intervals_FI_np[n][1] - theta[n]) for n in range(len(theta))]
            err = np.vstack((err_lo, err_hi))
            
            
        dg = draw_dodge(ax.errorbar, [r'$\epsilon$=' + eps for eps in epsilons], ms[method], yerr = err, ax=ax, 
                            dodge=Dodge[i], marker="o", markersize=4, ls='none', capsize=2, elinewidth=2, markeredgewidth=1.5, color=colors[method], label=labels[method]) 
        #dg[-1][0].set_linestyle('--')
    

    plt.legend()
    plt.title("95% confidence intervals")
    plt.savefig(path+"/95CIeps" + mode +".pdf") 
    
    

if __name__ == "__main__":
    
    SMALLER_SIZE = 12
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22
    
    plt.rc('font', size=MEDIUM_SIZE)          
    plt.rc('axes', titlesize=MEDIUM_SIZE)     
    plt.rc('axes', labelsize=MEDIUM_SIZE)    
    plt.rc('xtick', labelsize=SMALLER_SIZE)  
    plt.rc('ytick', labelsize=SMALLER_SIZE)    
    plt.rc('legend', fontsize=SMALL_SIZE)   
    plt.rc('figure', titlesize=BIGGER_SIZE)  

    path = "0609_gamma_trunc_2xbis"
    d = 'gamma'
    eps = '0.5'
    trueparam = 4.1692107471768525
    
    #plotresults(path, d, 'empirical', eps, small=False)   
    plotresultsUpLow(path, d, 'empirical', eps) 
    #plotIntervals(path, d, 'empirical', eps, small=False)
    #plotresultsEps(path)  
    #plotIntervalsEps(path, d, 'empirical')
    
            
    
    
    
    
    
    