import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib import transforms
import pandas as pd
import seaborn as sns
from scipy.stats import norm 

#####################


def plotresults(path, d, mode, epsilon):

    xlimit = 0.0

    i_dict = {0: "", 1: "_FISHER", 2:"_BASIC" , 3: "_FISHERNP"}
    #j_dict = {0: "50", 1: "100", 2: "200", 3: "300", 4:"400", 5:"500", 6:"750", 7:"1000", 8:"5000", 9:"10000"}
    j_dict = {0: "50", 1: "100", 2: "200", 3:"500", 4:"1000", 5:"5000", 6:"10000"}
    #j_dict = {0: "50", 1: "100", 2: "200", 3:"500", 4:"1000", 5:"5000"}
    #j_dict = {0: "50", 1: "100", 2: "500", 3:"1000", 4:"5000", 5:"10000"}
    #j_dict = {0: "50", 1: "100", 2: "500", 3:"1000"}
    #j_dict = {0: "5", 1: "10", 2: "50", 3:"100"}
    
    list_ci_levels = [50, 60, 70, 80, 90, 95, 99]

        
    fig, axs = plt.subplots(len(i_dict), len(j_dict), figsize=(17, 9), sharex=True, sharey=True)
    
    for j in range(len(axs[0])):
        axs[0,j].set_title("N = " + j_dict[j])

    fig.suptitle("Nominal vs observed interval coverage "+ "(" + str(d).title() +")")
    
    for i in range(len(i_dict)): #our method, naive method
        for j in range(len(j_dict)): # 50, 100, 500, 1000
        
            x = np.linspace(xlimit,1,100)
            y = x
            y2 = (1-x)/2
            #y3 = 0*x
            #axs[i,j].plot(x, y, 'grey', label='y=x')
            axs[i,j].plot(x, y, 'grey', label='perfect coverage')
            
            #axs = fig.add_subplot(i+1,j+1,1)
            results = np.load(path+ "/results_"+ d +"_N" + j_dict[j] +"_epsilon" + epsilon +"_" + mode + i_dict[i] +".npy", allow_pickle=True)
            print(results.shape)
            results = results[:,0,]
            #results_pois = np.load("results_"+ "poisson" +"_N" + j_dict[j] +"_epsilon0.5_analytic" + i_dict[i] +".npy", allow_pickle=True)
            #results_pois = np.abs((results_pois-np.array([el/100 for el in list_ci_levels])))
            
            # uf_gaus = np.load("upperfailures_"+d+"_N" + j_dict[j] +"_epsilon0.5_analytic" + i_dict[i] +".npy", allow_pickle=True)
            # uf_rate_gaus = [int(u)/2000 for u in uf_gaus]
            # lf_gaus = np.load("lowerfailures_"+d+"_N" + j_dict[j] +"_epsilon0.5_analytic" + i_dict[i] +".npy", allow_pickle=True)
            # lf_rate_gaus = [int(u)/2000 for u in lf_gaus]
            
            # uf = np.load("upperfailures_"+ d +"_N" + j_dict[j] +"_epsilon" + epsilon +"_" + mode + i_dict[i] +".npy", allow_pickle=True)
            # uf_rate = [int(u)/2000 for u in uf]
            # lf = np.load("lowerfailures_"+ d +"_N" + j_dict[j] +"_epsilon" + epsilon +"_" + mode + i_dict[i] +".npy", allow_pickle=True)
            # lf_rate = [int(u)/2000 for u in lf]

            
            #axs[i,j].scatter([ci/100 for ci in list_ci_levels], results, marker='x',s=50, color='r', label=d)
            axs[i,j].scatter([ci/100 for ci in list_ci_levels], results, marker='o',s=35, color='b', label='output coverage')
            
            #axs[i,j].scatter([ci/100 for ci in list_ci_levels], uf_rate, marker='x',s=40, color='g', label='upper tail error')
            #axs[i,j].scatter([ci/100 for ci in list_ci_levels], lf_rate, marker='x',s=40, color='r', label='lower tail error')
            
            
            #axs[i,j].axis(xmin=xlimit,xmax=1.0)
        

    #plt.setp(axs[-1, :], xlabel='confidence')
    plt.setp(axs[0, 0], ylabel='private PB')
    plt.setp(axs[1, 0], ylabel='private FI')
    plt.setp(axs[2, 0], ylabel='public PB')
    plt.setp(axs[3, 0], ylabel='public FI')
    plt.xlim((xlimit, 1.0))
    plt.ylim((xlimit, 1.0))
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    #axs.flatten()[-4].legend(loc='upper center', bbox_to_anchor=(3.5, 3.9), ncol=3)
    #fig.legend(loc='upper center', fancybox=True, shadow=True)
    
    plt.savefig(path +"/coverage_" + d + epsilon + mode + ".png")
    
    plt.close()
        
    
def draw_dodge(*args, **kwargs):
    func = args[0]
    dodge = kwargs.pop("dodge", 0)
    ax = kwargs.pop("ax", plt.gca())
    trans = ax.transData  + transforms.ScaledTranslation(dodge/72., 0,
                                   ax.figure.dpi_scale_trans)
    artist = func(*args[1:], **kwargs)
    def iterate(artist):
        if hasattr(artist, '__iter__'):
            for obj in artist:
                iterate(obj)
        else:
            artist.set_transform(trans)
    iterate(artist)
    return artist

        
def plotIntervals(path, dist, mode, epsilon):
    
    
    #j_dict = {0: "5", 1: "10", 2: "50", 3:"100", 4:"500", 5:"1000", 6:"5000"}
    Ns = [50, 100, 200, 500, 1000, 5000, 10000]
    #Ns = [5000, 10000]
    true_theta = 4.1692107471768525
    list_ci_levels = [50, 60, 70, 80, 90, 95, 99]
    z_values = {50: 0.674, 60: 0.841, 70: 1.036, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
    
    methods = ["", "_BASIC"]
    #methods = ["", "_BASIC", "_FISHER", "_FISHERNP"]
    colors = {"": 'teal', "_FISHER":'lightgreen', "_FISHERNP":'purple', "_BASIC": 'magenta'}
    offsets = {"": -0.80/4,  "_BASIC": -0.40/4, "_FISHER":+0.40/4, "_FISHERNP":+0.80/4}
    labels = {"": 'private PB', "_FISHER":'private Fisher', "_BASIC": 'public PB', "_FISHERNP":'public Fisher'}
    ms = {}
    sds = {}
    
    for jj, method in enumerate(methods):
        
        means, stdvs = [], []
        
        for N in Ns:
            results = np.load(path+ "/results_"+ dist +"_N" + str(N) +"_epsilon" + epsilon +"_" +mode + method +".npy", allow_pickle=True)
            m, std = results[2,1], results[2,2]   #70% ci
            means.append(m)
            stdvs.append(std)
            #stdvs.append(std*np.sqrt(N))
        ms[method] = np.array(means)
        sds[method] = np.array(stdvs)
        
    sds_private = sds[""]
    sds_public = sds["_BASIC"]
    z_values_list = list(z_values.values())
    width_private = np.array([2*z_values_list[i]*sds_private[i] for i,el in enumerate(z_values_list)])
    width_public = np.array([2*z_values_list[i]*sds_public[i] for i,el in enumerate(z_values_list)])
    #print(width_private)
    #print(width_public)
    ratio = np.divide(width_private,width_public)
    print(ratio)
    
        
    Dodge = np.arange(len(methods),dtype=float)*10
    Dodge -= Dodge.mean()   
    
    fig, ax = plt.subplots()
    plt.axhline(y=true_theta, color='black', linestyle='-', label='true parameter')
    for i,method in enumerate(methods):
        print(i, method)
        draw_dodge(ax.errorbar, ["N = " + str(N) for N in Ns], ms[method], yerr = z_values[70]*sds[method], ax=ax, 
                            dodge=Dodge[i], marker="o", markersize=5, ls='none', color=colors[method], label=labels[method]) 
        # plt.bar(np.array([5,10,50,100]) + offsets[method], means, 0.80, yerr=stdvs*z_values[70], color=colors[method])
        # 
        
        #plt.ylim(-2.5,12)

    
    # Show graphic
    plt.legend()
    plt.title("Average 70% confidence intervals, " + dist.title())
    plt.savefig(path+"/70CI" + dist + "_" + epsilon +"_" + mode +".png") 


        
    
    
if __name__ == "__main__":
    
    SMALLER_SIZE = 10
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    path = "0522"
    #plotresults(path, 'gaussian', 'empirical', '0.1')
    #plotresults(path, 'gaussian', 'analytic', '0.1')   
    plotIntervals(path, 'gaussian', 'analytic', '0.1')

            
    
    
    
    
    
    