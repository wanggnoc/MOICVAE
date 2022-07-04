import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from sklearn import metrics
from matplotlib import pyplot
from itertools import cycle
 
styles=['fivethirtyeight',
 'dark_background',
 'bmh',
 'classic',
 'seaborn-dark',
 'grayscale',
 'seaborn-deep',
 'seaborn-muted',
 'seaborn-colorblind',
 'seaborn-white',
 'seaborn-dark-palette',
 'ggplot',
 'tableau-colorblind10',
 '_classic_test',
 'seaborn-darkgrid',
 'seaborn-notebook',
 'Solarize_Light2',
 'seaborn-paper',
 'seaborn-whitegrid',
 'seaborn-pastel',
 'seaborn-talk',
 'seaborn-bright',
 'seaborn',
 'seaborn-ticks',
 'seaborn-poster',
 'fast']
 
def rocs_plot(P2, Y2,setname=['']):
 
    pyplot.figure(figsize=(5, 4), dpi=100)
    pyplot.style.use('seaborn-darkgrid')
    
    palette = pyplot.get_cmap('Set1')
    
    colors = cycle('rbgcmy')
    i=0
    for pp, yy,col in zip(Y2,P2,colors):
        fpr1, tpr1, thresholds1 = metrics.roc_curve(pp, yy, pos_label=1)
        roc_auc1 = metrics.auc(fpr1, tpr1)   
        pyplot.plot(fpr1, tpr1, lw=1, color=col, linewidth=1.5, alpha=0.9, label="%s:AUC=%0.3f" % (setname[i],roc_auc1))
        i += 1
    pyplot.xlim([0.00, 1.0])
    pyplot.ylim([0.00, 1.0])
    pyplot.xlabel("1-Specificity",fontsize=12)
    pyplot.ylabel("Sensitivity",fontsize=12)
    pyplot.title("ROC of GDSC ",fontsize=14,fontweight='bold')
    pyplot.legend(loc="lower right")
    pyplot.savefig('ROC.png',dpi=200)

y_ic50 = np.loadtxt('y_ic50_GDSC_2.txt')
y_true_ic50 = np.loadtxt('y_true_ic50_GDSC_2.txt')
#print(y_ic50)
#print(y_true_ic50)
y_true_ic50 = np.array([1 if a>=0.5 else 0 for a in y_true_ic50 ],dtype=int)


dsp_p = np.loadtxt('DSPLMF_y_ic50_GDSC.txt')
dsp_t = np.loadtxt('DSPLMF_y_true_ic50_GDSC.txt')
dsp_t = dsp_t.astype(int)
names=['MOICVAE','DSPLMF','AutoBorutaRF']
autob = np.loadtxt('AutoBorutaRF_GDSC_ROC.txt')
autob_t= 1-autob[:,0].astype(int)
autob_p = autob[:,1]
rocs_plot((y_ic50,dsp_p,autob_p), (y_true_ic50,dsp_t,autob_t),names)
