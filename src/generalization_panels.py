"""Paneles de generalización con unidades y modo explícitos, sin usar test."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def diagnostics(folder,selection,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);h=pd.read_csv(folder/'history.csv')
    fig,axes=plt.subplots(2,3,figsize=(18,10),constrained_layout=True)
    epoch=selection['epoch'];blue='#3277be';orange='#ed7132'
    def lines(ax,fit,val,heading,ylabel):
        ax.plot(h.epoch,fit,color=blue,label='Entrenamiento')
        ax.plot(h.epoch,val,color=orange,label='Validación')
        ax.axvline(epoch,color='#555',ls='--',label='Época elegida por AUC val')
        ax.set(title=heading,xlabel='Época',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
    if {'train_patient_bce','validation_patient_bce'}.issubset(h):
        lines(axes[0,0],h.train_patient_bce,h.validation_patient_bce,
            'PRINCIPAL · BCE por paciente\nMismo checkpoint · eval · sin aumentos/dropout','BCE normal')
    else:
        axes[0,0].text(.5,.5,'Historial sin pérdidas por paciente\nNo se reconstruyen trayectorias inexistentes',ha='center',transform=axes[0,0].transAxes);axes[0,0].axis('off')
    lines(axes[0,1],h.train_patient_auc,h.patient_roc_auc,
        'PRINCIPAL · ROC-AUC por paciente\nEval · media de probabilidades de cortes','ROC-AUC')
    axes[0,1].set_ylim(0,1);axes[0,1].axhline(.5,color='gray',ls=':',lw=1)
    tr=selection['train_eval']['patient_metrics']['roc_auc'];va=selection['validation_eval']['patient_metrics']['roc_auc']
    bars=axes[0,2].bar(['Entrenamiento','Validación'],[tr,va],color=[blue,orange])
    axes[0,2].bar_label(bars,fmt='%.3f',padding=3)
    axes[0,2].set(title=f'AUC en la misma época {epoch}\nBrecha train − val = {tr-va:+.3f}',ylabel='ROC-AUC',ylim=(0,1.08))
    axes[0,2].axhline(.5,color='gray',ls=':',lw=1)
    axes[0,2].text(.5,.04,'TEST reservado: no evaluado',transform=axes[0,2].transAxes,ha='center',fontsize=9)
    s=json.loads((folder/'config_resolved.json').read_text());s=s.get('recipe',s)
    weighted=s.get('loss')=='ponderada' or folder.parent.name=='ponderada'
    if weighted and 'validation_objective_loss' in h:
        val=h.validation_objective_loss;heading='OPERATIVA · objetivo ponderado por corte'
    elif not weighted:
        val=h.validation_loss;heading='OPERATIVA · BCE normal por corte'
    else:
        val=None
    if val is not None:
        lines(axes[1,0],h.train_loss,val,heading+'\nTrain online (modo train); val en eval','Pérdida del objetivo')
    else:
        axes[1,0].text(.5,.07,'Val ponderada no guardada en este historial',ha='center',fontsize=8,transform=axes[1,0].transAxes,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        axes[1,0].plot(h.epoch,h.train_loss,color=blue,label='Train online ponderada');axes[1,0].legend(fontsize=8)
        axes[1,0].set(xlabel='Época',ylabel='BCE ponderada',title='OPERATIVA · datos históricos incompletos')
    lines(axes[1,1],h.train_eval_loss,h.validation_loss,
        'COMPARABLE · BCE normal por corte\nAmbas en eval · misma unidad','BCE normal')
    if 'training_cuts' in h:
        seen=h.training_cuts.cumsum().to_numpy()
    else:
        n=len(pd.read_csv(folder/'predictions_train_cuts_best.csv'))
        seen=h.epoch.to_numpy()*n
    axes[1,2].plot(seen,h.train_patient_auc,color=blue,label='Entrenamiento')
    axes[1,2].plot(seen,h.patient_roc_auc,color=orange,label='Validación')
    chosen=np.flatnonzero(h.epoch.to_numpy()==epoch)
    if len(chosen):axes[1,2].axvline(seen[chosen[0]],color='#555',ls='--')
    axes[1,2].set(title='AUC según exposición al entrenamiento\nPermite comparar épocas de distinto tamaño',xlabel='Cortes procesados acumulados en train',ylabel='ROC-AUC',ylim=(0,1))
    axes[1,2].legend(fontsize=8);axes[1,2].grid(alpha=.2)
    fig.suptitle(title+' · diagnóstico de sobreajuste',fontsize=16)
    path=folder/'diagnostico_sobreajuste.png';temp=folder/'diagnostico_sobreajuste.tmp.png'
    fig.savefig(temp,dpi=125);temp.replace(path);plt.close(fig)
    return path


def fold_overview(folders,destination):
    """Pérdidas/AUC por fold y brecha; no promedia curvas con épocas distintas."""
    if not folders:return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    fits=[];vals=[];labels=[]
    for folder in folders:
        s=json.loads((folder/'config_resolved.json').read_text());h=pd.read_csv(folder/'history.csv')
        selected=json.loads((folder/'selection.json').read_text());e=selected['epoch']
        suffix='' if (folder/'summary.json').exists() else ' parcial'
        label=f'Fold{s["fold"]}{suffix}';labels.append(label)
        if {'train_patient_bce','validation_patient_bce'}.issubset(h):
            axes[0].plot(h.epoch,h.train_patient_bce,color='#3277be',alpha=.5)
            axes[0].plot(h.epoch,h.validation_patient_bce,color='#ed7132',alpha=.5)
            chosen=h.loc[h.epoch==e].iloc[0];axes[0].scatter([e],[chosen.validation_patient_bce],s=25,color='#ed7132')
        axes[1].plot(h.epoch,h.train_patient_auc,color='#3277be',alpha=.5)
        axes[1].plot(h.epoch,h.patient_roc_auc,color='#ed7132',alpha=.5)
        fits.append(selected['train_eval']['patient_metrics']['roc_auc']);vals.append(selected['validation_eval']['patient_metrics']['roc_auc'])
        axes[1].scatter([e],[vals[-1]],s=25,color='#ed7132')
    for ax,title,ylabel in zip(axes[:2],['BCE por paciente · mismo modo eval','ROC-AUC por paciente · mismo modo eval'],['BCE normal','ROC-AUC']):
        ax.plot([],[],color='#3277be',label='Entrenamiento por fold');ax.plot([],[],color='#ed7132',label='Validación por fold')
        ax.scatter([],[],color='#ed7132',s=25,label='Época elegida');ax.set(title=title,xlabel='Época',ylabel=ylabel);ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[1].set_ylim(0,1)
    x=np.arange(len(labels));axes[2].bar(x-.18,fits,.36,color='#3277be',label='Entrenamiento');axes[2].bar(x+.18,vals,.36,color='#ed7132',label='Validación')
    axes[2].set(xticks=x,xticklabels=labels,ylim=(0,1.08),title='AUC en la época elegida de cada fold\nTEST reservado · sin barras inventadas',ylabel='ROC-AUC');axes[2].legend(fontsize=8)
    for i,(a,b) in enumerate(zip(fits,vals)):axes[2].text(i,max(a,b)+.03,f'Δ={a-b:+.2f}',ha='center',fontsize=8)
    fig.suptitle('C23 · curvas por fold · puntos elegidos con validación, no test independiente',fontsize=14)
    destination=Path(destination);temp=destination.with_name(destination.stem+'.tmp.png');fig.savefig(temp,dpi=125);temp.replace(destination);plt.close(fig)
    return destination
