import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib import transforms
import pandas as pd
from scipy.stats import norm 
import seaborn as sns



def MoE(path, dist, mode, epsilon, conf, r, df):
    
    #Ns = [50, 100, 200, 500, 1000, 5000, 10000]
    Ns = [10, 50, 100, 500, 1000, 5000, 10000]
    #Ns = [2782]
    #Ns = [5, 10, 50, 100]
    true_theta = 4.1692107471768525
    list_ci_levels = [50, 60, 70, 80, 90, 95, 99]
    z_values = {50: 0.674, 60: 0.841, 70: 1.036, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}
    conf_idx = {50: 0, 60: 1, 70: 2, 80: 3, 90: 4, 95: 5, 99: 6}
    moe_public = {}
    
    methods = [""]
    #methods = ["", "_BASIC", "_FISHER", "_FISHERNP"]
    colors = {"": 'blue', "_FISHER":'orange', "_FISHERNP":'red', "_BASIC": 'magenta'}
    offsets = {"": -0.80/4,  "_BASIC": -0.40/4, "_FISHER":+0.00/4, "_FISHERNP":+0.80/4}
    labels = {"": 'private PB', "_FISHER":'private FI', "_BASIC": 'public PB', "_FISHERNP":'public FI'}
    titles = {"poisson": "Poisson", "gaussian": "Gaussian, known variance", "gaussian2": "Gaussian, unknown variance", "gamma": "Gamma"}
    linestyles = {"": '-', "_FISHER":'-.', "_FISHERNP":'red', "_BASIC": 'magenta'}

    
    
    #extract means and stds to compute average intervals

    means, stdvs, pubws, ourws, ourcov, ratios = [], [], [], [], [], []
    for N in Ns:
        cov, m, std = np.load(path+ "/results_"+ dist +"_N" + str(N) +"_epsilon" + str(epsilon) +"_" +mode + "" +".npy", allow_pickle=True)[0]
        #m, std = results[conf_idx[95],0,], results[conf_idx[95],1,], results[conf_idx[95],2,]   #95% CIs
        ourw = np.load(path+ "/widths_"+ dist +"_N" + str(N) +"_epsilon" + str(epsilon) +"_" +mode + "" +".npy", allow_pickle=True)[0]
        #moe_public_N_conf = df['alg':['pub'], 'n': [N], 'range':[r], 'a':[(100-conf)/100], 'coverage':[]  ]
        info = df[(df['alg'] == 'pub') & (df['n'] == N) & (df['e'] == float(epsilon)) & (df['range'] == r) & (df['a'] == 0.05)]
        #moe_public_N_conf = df.at[:,'pub',N,epsilon,r,(100-conf)/100]
        pubw = info['width']
        means.append(m)
        stdvs.append(std)
        ourcov.append(cov)
        pubws.append(pubw)
        ourws.append(ourw)
    
        ratio = ourw/pubw
        ratios.append(ratio)
        
        
    print("OUR", ourws)
    print("OUR coverage", ourcov)

    #print("RATIOS", ratios)
    
    
def barplot(path, df, all = False):

    ax = sns.barplot(x="n", y="width", hue="algorithm:", data=df).set_yscale("log")
    plt.savefig("comp.pdf")
    
    

if __name__ == "__main__":
    
    
    df = pd.read_csv("comp_full_plot.csv") #path to results csv file with results from other methods

    path = "comp8"
    
    r = int(path[4:])
    eps = 0.1
    
    #MoE(path, "gaussian2", "empirical", eps, 95, r, df)
    
    df = df[(df['range']==r) & (df['e']==eps)]
    
    barplot(path, df, True)
    