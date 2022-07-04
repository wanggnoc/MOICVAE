import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
pred = np.loadtxt('./y_ic50_GDSC.txt') #pred
label = np.loadtxt('./y_true_ic50_GDSC.txt') #true
from scipy.stats import pearsonr

index = np.loadtxt('index.txt')
index = index.astype('int')
#index_drug = index[:,index[:,0]==81]
name = '1'

def scatter_plot(pred=pred,label=label,name=name):
    rmse = np.sqrt(np.sum((pred-label)**2)/len(pred))
    pearson = pearsonr(pred, label)[0]
    
    y_ic50 = label
    yhat_ic50 = pred
    df = pd.DataFrame({
        'y_ic50':y_ic50,
        'yhat_ic50':yhat_ic50
    })
    sns.set(rc={'figure.figsize':(15,15)})
    with sns.axes_style('white'):
        fig = sns.jointplot(x='yhat_ic50',y='y_ic50',data=df,kind='hex',color='b')
        pearson = 'Pearson = {}'.format(round(pearson,4))
        r2 = 'R2 = {}'.format(round(pearsonr(y_ic50,yhat_ic50)[0]**2,4))
        rmse ='RMSE = {}'.format(round(rmse,4))
        n = 'n = {}'.format(len(y_ic50))
        fig.ax_joint.text(0.1,0.9,'drug:'+name,weight='bold')
        fig.ax_joint.text(0.1,0.8,rmse,weight='bold')
        fig.ax_joint.text(0.1,0.7,pearson,weight='bold')
        fig.ax_joint.text(0.1,0.6,r2,weight='bold')
        fig.ax_joint.text(0.1,0.5,n,weight='bold')
        fig.ax_joint.set_xticks(np.array([0.,0.25,0.5,0.75,1.]))
        fig.ax_joint.set_yticks(np.array([0.,0.25,0.5,0.75,1.]))
        fig.ax_joint.set_xlim([0.,1.])
        fig.ax_joint.set_ylim([0.,1.])
        fig.ax_joint.set_xlabel('Predicted IC50',weight='bold')
        fig.ax_joint.set_ylabel('Measured IC50',weight='bold')
        x0, x1 = fig.ax_joint.get_xlim()
        y0, y1 = fig.ax_joint.get_ylim()
        fig.ax_joint.plot([max(x0,y0),min(x1,y1)],[max(x0,y0),min(x1,y1)],':k',linewidth=1.75)
        plt.savefig('scatter_GDSC_'+name+'.png')
drugs = ['AICAR', 'AKT inhibitor VIII', 'AS601245', 'AZD6482', 'AZD7762',
       'AZD8055', 'BIBW2992', 'BMS-708163', 'Axitinib', 'BAY 61-3606',
       'BMS-754807', 'BX-795', 'Bexarotene', 'Bicalutamide', 'Bleomycin',
       'Bosutinib', 'Bryostatin 1', 'CCT018159', 'CCT007093', 'CHIR-99021',
       'CI-1040', 'Cisplatin', 'Cytarabine', 'DMOG', 'LAQ824', 'NVP-BEZ235',
       'Docetaxel', 'BIRB 0796', 'Doxorubicin', 'EHT 1864', 'Elesclomol',
       'embelin', 'Epothilone B', 'etoposide', 'FH535', 'FTI-277',
       'GSK-1904529A', 'GSK-650394', 'GW 441756', 'Gefitinib', 'Gemcitabine',
       'IPA-3', 'JNK Inhibitor VIII', 'JNK-9L', 'KU-55933', 'LFM-A13',
       'lenalidomide', 'CEP-701', 'OSI-906', 'MK-2206', 'Methotrexate',
       'Midostaurin', 'Mitomycin-C', 'AMG-706', 'NSC-87877', 'NU-7441',
       'ABT-263', 'Nilotinib', 'Nutlin-3', 'OSU-03012', 'Obatoclax Mesylate',
       'AZD-2281', 'pac-1', 'PD-0325901', 'PD173074', 'PF-4708671',
       'PF-562271', 'PLX4720', 'PD-0332991', 'Pazopanib', 'GDC-0449',
       'AP-24534', 'QS11', 'RO-3306', 'RDEA119', 'AG-014699', 'SB 216763',
       'SB590885', 'SL 0101-1', 'Camptothecin', 'GDC-0941', 'AZD6244',
       'JNJ-26854165', 'shikonin', 'TW 37', '17-AAG', 'temsirolimus',
       'thapsigargin', 'Tipifarnib', 'ATRA', 'VX-702', 'ABT-888',
       'Vinblastine', 'Vinorelbine', 'AUY922', 'Vorinostat', '681640',
       'ZM-447439']
for i in range(98):
    index_line = np.where(index[:,0] == i)[0]
    label2 = label[index_line]
    pred2 = pred[index_line]
    pearson = pearsonr(pred2, label2)[0]
    if pearson>0.68:
        scatter_plot(pred=pred2, label=label2,name=drugs[i])