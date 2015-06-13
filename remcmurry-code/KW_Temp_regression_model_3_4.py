# -*- coding: utf-8 -*-
"""
Created on Fri Feb 06 14:45:24 2015

@author: rmcmurry

updated April 1 2015 for python 3.4
"""
import numpy as np 
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot 
import matplotlib.pyplot as plt 
from scipy import stats
#import os
import pandas as pd
#import datetime as dt


#def m_xld_datetime(xldate, datemode):
#    # datemode: 0 for 1900-based, 1 for 1904-based
#    newdate=dt.datetime(1899, 12, 30) + dt.timedelta(days=xldate + 1462 * datemode)
#    return(newdate)


#%% PGE Data I/O setup and loading

st_YrTr=2012    # St-En year range for rgression training
en_YrTr=2013
st_YrEv=2014    # Evaluation year to build predictions from regression 
en_YrEv=st_YrEv #change to a year to evaluate predictions for multiple years
alFA=0.15        # transparency for graph distribution clouds
outputGraphs=True  # do you want plots saved?

#%% load data
shareDrivePGE='S:/Current Projects/Western Cooling Challenge/'+\
             'Field Installations/Munters EXP 5000 at Whole Foods San Ramon/'+\
             'data analysis/month data/supplemental/'

pgeFileName='_PGE_int.csv'        

pgeHeader=['Timestamp','pgOA_Temp','pgkW_Site','kVAR','KVA','pf']
i=0

#==============================================================================
# # load training data
#==============================================================================
for year in range(st_YrTr,en_YrTr+1,1):
    i=i+1
    if i==1:
        pgeinputPath=shareDrivePGE+'PGandE/'+str(year)+pgeFileName
        pgeDFtr=pd.read_csv(pgeinputPath,header=0,names=pgeHeader,index_col='Timestamp',
                      parse_dates=True,low_memory=False)
    else:
        pgeinputPath=shareDrivePGE+'PGandE/'+str(year)+pgeFileName
        pgedf=pd.read_csv(pgeinputPath,header=0,names=pgeHeader,index_col='Timestamp',
                          parse_dates=True,low_memory=False)
        pgeDFtr=pd.concat([pgeDFtr,pgedf])
        
pgeDFtr=pgeDFtr.dropna()
dfi=pd.DatetimeIndex(((pgeDFtr.index.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))
pgeDFtr.index=dfi


pgehourData={}  
pgehourData['kWmean']=[pgeDFtr[(pgeDFtr.index.hour==hr)].pgkW_Site.mean(skipna=True) for hr in range(24)]
#pgehourDF=pd.DataFrame(pgehourData)

tmHour=5
oHour=7
teHour=20
cHour=22

WFopen=np.array([pgeDFtr.pgOA_Temp[(pgeDFtr.index.hour>=oHour) & (pgeDFtr.index.hour<=teHour)],
                 pgeDFtr.pgkW_Site[(pgeDFtr.index.hour>=oHour) & (pgeDFtr.index.hour<=teHour)]])

WFtrans=np.array([pgeDFtr.pgOA_Temp[((pgeDFtr.index.hour>=tmHour) & (pgeDFtr.index.hour<=oHour)) | 
                  ((pgeDFtr.index.hour>=teHour) & (pgeDFtr.index.hour<=cHour))],
                  pgeDFtr.pgkW_Site[((pgeDFtr.index.hour>=tmHour) & (pgeDFtr.index.hour<=oHour)) |
                  ((pgeDFtr.index.hour>=teHour) & (pgeDFtr.index.hour<=cHour))]])

WFclose=np.array([pgeDFtr.pgOA_Temp[(pgeDFtr.index.hour<tmHour) | (pgeDFtr.index.hour>cHour)],
                  pgeDFtr.pgkW_Site[(pgeDFtr.index.hour<tmHour) | (pgeDFtr.index.hour>cHour)]])

WFopen=np.transpose(WFopen)
WFtrans=np.transpose(WFtrans)
WFclose=np.transpose(WFclose)

WFopenDF=pd.DataFrame(WFopen,columns=['temp','kW'])
WFtransDF=pd.DataFrame(WFtrans,columns=['temp','kW'])
WFcloseDF=pd.DataFrame(WFclose,columns=['temp','kW'])

#==============================================================================
# # load in evaluation / prediction data
#==============================================================================
i=0
for year in range(st_YrEv,en_YrEv+1,1):
    i=i+1
    if i==1:
        pgeinputPath=shareDrivePGE+'PGandE/'+str(year)+pgeFileName
        pgeDFev=pd.read_csv(pgeinputPath,header=0,names=pgeHeader,index_col='Timestamp',
                      parse_dates=True,low_memory=False)
    else:
        pgeinputPath=shareDrivePGE+'PGandE/'+str(year)+pgeFileName
        pgedf=pd.read_csv(pgeinputPath,header=0,names=pgeHeader,index_col='Timestamp',
                          parse_dates=True,low_memory=False)
        pgeDFev=pd.concat([pgeDFev,pgedf])
        
pgeDFev=pgeDFev.dropna()
dfi=pd.DatetimeIndex(((pgeDFev.index.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))
pgeDFev.index=dfi
 
pgehourData['kWmean_ev']=[pgeDFev[(pgeDFev.index.hour==hr)].pgkW_Site.mean(skipna=True) for hr in range(24)]
pgehourDF=pd.DataFrame(pgehourData)



evWFopen=np.array([pgeDFev.pgOA_Temp[(pgeDFev.index.hour>=oHour) & (pgeDFev.index.hour<=teHour)],
                   pgeDFev.pgkW_Site[(pgeDFev.index.hour>=oHour) & (pgeDFev.index.hour<=teHour)]])

evWFtrans=np.array([pgeDFev.pgOA_Temp[((pgeDFev.index.hour>=tmHour) & (pgeDFev.index.hour<=oHour)) | 
                  ((pgeDFev.index.hour>=teHour) & (pgeDFev.index.hour<=cHour))],
                  pgeDFev.pgkW_Site[((pgeDFev.index.hour>=tmHour) & (pgeDFev.index.hour<=oHour)) |
                  ((pgeDFev.index.hour>=teHour) & (pgeDFev.index.hour<=cHour))]])

evWFclose=np.array([pgeDFev.pgOA_Temp[(pgeDFev.index.hour<tmHour) | (pgeDFev.index.hour>cHour)],
                    pgeDFev.pgkW_Site[(pgeDFev.index.hour<tmHour) | (pgeDFev.index.hour>cHour)]])

evWFopen=np.transpose(evWFopen)
evWFtrans=np.transpose(evWFtrans)
evWFclose=np.transpose(evWFclose)

evWFopenDF=pd.DataFrame(evWFopen,columns=['temp','kW'])
evWFtransDF=pd.DataFrame(evWFtrans,columns=['temp','kW'])
evWFcloseDF=pd.DataFrame(evWFclose,columns=['temp','kW'])


#%% plot data

fig=plt.figure()
ax1=fig.add_subplot(311)
plt.plot(WFopen[:,0],WFopen[:,1],'+',c='#800080',label='Open Hours',
             alpha=alFA)
plt.plot(evWFopen[:,0],evWFopen[:,1],'x',c='#008000',label='ev Open Hours',
             alpha=alFA)
plt.ylabel('kW for WF Site')
plt.xlabel('Outdoor Air Temp ($^\circ$F)')
ax1.set_xlim(20,110)
plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
           borderaxespad=0,prop={'size':9},numpoints=1)
ax2=fig.add_subplot(312)
plt.plot(WFclose[:,0],WFclose[:,1],'+',c='#800080',label='Closed Hours',
             alpha=alFA)
plt.plot(evWFclose[:,0],evWFclose[:,1],'x',c='#008000',label='ev Closed Hours',
             alpha=alFA)
plt.ylabel('kW for WF Site')
plt.xlabel('Outdoor Air Temp ($^\circ$F)')
ax2.set_xlim(20,110)
plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
           borderaxespad=0,prop={'size':9},numpoints=1)
ax3=fig.add_subplot(313)
plt.plot(WFtrans[:,0],WFtrans[:,1],'+',c='#800080',label='Transition Hours',
             alpha=alFA)
plt.plot(evWFtrans[:,0],evWFtrans[:,1],'x',c='#008000',label='ev Transition Hours',
             alpha=alFA)
plt.ylabel('kW for WF Site')
plt.xlabel('Outdoor Air Temp ($^\circ$F)')
ax3.set_xlim(20,110)
plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
           borderaxespad=0,prop={'size':9},numpoints=1)

plotname='Training and Evaluation data clouds for closed transient and open hours'
outputPath=shareDrivePGE+'PGandE/output/'
if outputGraphs: plt.savefig(outputPath+str(st_YrTr)+'_to_'+str(en_YrEv)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

#%%  plot hour bin

plt.figure()

plt.plot(pgehourDF.index[0:6],pgehourDF.kWmean[0:6],'-',
         pgehourDF.index[22:],pgehourDF.kWmean[22:],c='#808000',label='kWmean close')

plt.plot(pgehourDF.index[5:8],pgehourDF.kWmean[5:8],'-',
         pgehourDF.index[20:23],pgehourDF.kWmean[20:23],c='#008080',label='kWmean trans')

plt.plot(pgehourDF.index[7:21],pgehourDF.kWmean[7:21],'-',c='#800080',label='kWmean open')

plt.plot(pgehourDF.index[0:6],pgehourDF.kWmean_ev[0:6],'-',
         pgehourDF.index[22:],pgehourDF.kWmean_ev[22:],c='#000080',label='ev kWmean close')

plt.plot(pgehourDF.index[5:8],pgehourDF.kWmean_ev[5:8],'-',
         pgehourDF.index[20:23],pgehourDF.kWmean_ev[20:23],c='#800000',label='ev kWmean trans')

plt.plot(pgehourDF.index[7:21],pgehourDF.kWmean_ev[7:21],'-',c='#008000',label='ev kWmean open')

plt.xlabel('Hour of Day')
plt.ylabel('kW')
plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
           borderaxespad=0,prop={'size':9},numpoints=1)

plotname='kW mean by hour'

plotname='Training and Evaluation hour bin data showing closed transient and open time'
outputPath=shareDrivePGE+'PGandE/output/'
if outputGraphs: plt.savefig(outputPath+str(st_YrTr)+'_to_'+str(en_YrEv)+'_'+plotname+'.png',
                             bbox_to_inches='tight')


#%% Break point itteration tool to find best regression break point to .1 deg F
# Open hours regression

BPmin=20
BPmax=90

BPbest=BPmin

#==============================================================================
# #start by gettting r squared of a linear regression 
#==============================================================================
oexog = WFopen[:,0] 
oexog = sm.add_constant(oexog)
oendog= WFopen[:,1]
res = sm.WLS(oendog, oexog).fit()
rsq=res.rsquared
rsq
#==============================================================================
# #ittereate by degree
#==============================================================================
for Bpoint in range (BPmin,BPmax,1):
    breaks = [0,Bpoint]  # 0 adds slope for entire array 
    oexog = np.column_stack([np.maximum(0, WFopen[:,0] - knot) for knot in breaks]) 
    oexog = sm.add_constant(oexog)
    oendog= WFopen[:,1]
    res = sm.WLS(oendog, oexog).fit()
    if res.rsquared > rsq:
        BPbest=Bpoint
        rsq=res.rsquared

#==============================================================================
# # refine to tenth of a degree
#==============================================================================
for Bpoint in np.arange(BPbest-1,BPbest+1,.1):
    breaks = [0,Bpoint]  # 0 adds slope for entire array 
    oexog = np.column_stack([np.maximum(0, WFopen[:,0] - knot) for knot in breaks]) 
    oexog = sm.add_constant(oexog)
    oendog= WFopen[:,1]
    res = sm.WLS(oendog, oexog).fit()
    if res.rsquared > rsq:
        BPbest=Bpoint
        rsq=res.rsquared

#==============================================================================
# # rerun best
#==============================================================================
breaks = [0,BPbest]  # 0 adds slope for entire array 
oexog = np.column_stack([np.maximum(0, WFopen[:,0] - knot) for knot in breaks]) 
oexog = sm.add_constant(oexog)
oendog= WFopen[:,1]
res = sm.WLS(oendog, oexog).fit()

#==============================================================================
# output regression results
#==============================================================================
print('params:')
print('WLS: ', res.params)
print('r^2: ', res.rsquared)
print('break point:', BPbest)
print('\nslopes:')
print('WLS: ', res.params[1:].cumsum())

print(res.summary())

# Closed hours regression


#==============================================================================
# #using a linear regression 
#==============================================================================
oexog = WFclose[:,0] 
oexog = sm.add_constant(oexog)
oendog= WFclose[:,1]
rescl = sm.WLS(oendog, oexog).fit()
rsqcl=rescl.rsquared
rsqcl

#==============================================================================
# output regression results
#==============================================================================
print('params:')
print('WLS: ', rescl.params)
print('r^2: ', rescl.rsquared)
print('\nslopes:')  
print('WLS: ', rescl.params[1:].cumsum())

print(rescl.summary())

# Transition hours regression


#==============================================================================
# #using a linear regression 
#==============================================================================
oexog = WFtrans[:,0] 
oexog = sm.add_constant(oexog)
oendog= WFtrans[:,1]
restr = sm.WLS(oendog, oexog).fit()
rsqtr=restr.rsquared
rsqtr

#==============================================================================
# output regression results
#==============================================================================
print('params:')
print('WLS: ', restr.params)
print('r^2: ', restr.rsquared)
print('\nslopes:')
print('WLS: ', restr.params[1:].cumsum())

print(restr.summary())



#%% temp binned mean values
tempWFopen={} 
trmin=20
trmax=110
trbin=1

tempWFopen['tempbin']=range(trmin,trmax,trbin)

tempWFopen['kW_mean']= [WFopenDF[((WFopenDF['temp']>tr) & (WFopenDF['temp']<
                 (tr+trbin)))].kW.mean() for tr in range(trmin,trmax,trbin)]
tempWFopen['kW_sDev']= [WFopenDF[((WFopenDF['temp']>tr) & (WFopenDF['temp']<
                 (tr+trbin)))].kW.std() for tr in range(trmin,trmax,trbin)]
tempstatsDF=pd.DataFrame(tempWFopen)
plt.figure()
plt.plot(tempstatsDF.tempbin,tempstatsDF.kW_mean, 'lime', label='kW mean') 
plt.plot(tempstatsDF.tempbin,tempstatsDF.kW_sDev, 'red', label='kW standard deviation')
plt.plot(tempstatsDF.tempbin,[tempstatsDF.kW_sDev.mean()]*len(tempstatsDF.tempbin),'b',label='avage St Dev over bins')

plotname='temperature trend of regession stats for open hours'
outputPath=shareDrivePGE+'PGandE/output/'
if outputGraphs: plt.savefig(outputPath+str(st_YrTr)+'_to_'+str(en_YrTr)+'_'+plotname+'.png',
                             bbox_to_inches='tight') 

# Closed Stats
tempWFclose={} 
#trmin=20
#trmax=110
#trbin=1

tempWFclose['tempbin']=range(trmin,trmax,trbin)

tempWFclose['kW_mean']= [WFcloseDF[((WFcloseDF['temp']>tr) & (WFcloseDF['temp']<
                 (tr+trbin)))].kW.mean() for tr in range(trmin,trmax,trbin)]
tempWFclose['kW_sDev']= [WFcloseDF[((WFcloseDF['temp']>tr) & (WFcloseDF['temp']<
                 (tr+trbin)))].kW.std() for tr in range(trmin,trmax,trbin)]
tempstatsDFcl=pd.DataFrame(tempWFclose)
plt.figure()
plt.plot(tempstatsDFcl.tempbin,tempstatsDFcl.kW_mean, 'lime', label='kW mean') 
plt.plot(tempstatsDFcl.tempbin,tempstatsDFcl.kW_sDev, 'red', label='kW standard deviation')
plt.plot(tempstatsDFcl.tempbin,[tempstatsDFcl.kW_sDev.mean()]*len(tempstatsDFcl.tempbin),'b',label='avage St Dev over bins')

plotname='temperature trend of regession stats for closed hours'
outputPath=shareDrivePGE+'PGandE/output/'
if outputGraphs: plt.savefig(outputPath+str(st_YrTr)+'_to_'+str(en_YrTr)+'_'+plotname+'.png',
                             bbox_to_inches='tight') 

# Transition stats
tempWFtrans={} 
#trmin=20
#trmax=110
#trbin=1

tempWFtrans['tempbin']=range(trmin,trmax,trbin)

tempWFtrans['kW_mean']= [WFtransDF[((WFtransDF['temp']>tr) & (WFtransDF['temp']<
                 (tr+trbin)))].kW.mean() for tr in range(trmin,trmax,trbin)]
tempWFtrans['kW_sDev']= [WFtransDF[((WFtransDF['temp']>tr) & (WFtransDF['temp']<
                 (tr+trbin)))].kW.std() for tr in range(trmin,trmax,trbin)]
tempstatsDFtr=pd.DataFrame(tempWFtrans)
plt.figure()
plt.plot(tempstatsDFtr.tempbin,tempstatsDFtr.kW_mean, 'lime', label='kW mean') 
plt.plot(tempstatsDFtr.tempbin,tempstatsDFtr.kW_sDev, 'red', label='kW standard deviation')
plt.plot(tempstatsDFtr.tempbin,[tempstatsDFtr.kW_sDev.mean()]*len(tempstatsDFtr.tempbin),'b',label='avage St Dev over bins')

plotname='temperature trend of regession stats for transient hours'
outputPath=shareDrivePGE+'PGandE/output/'
if outputGraphs: plt.savefig(outputPath+str(st_YrTr)+'_to_'+str(en_YrTr)+'_'+plotname+'.png',
                             bbox_to_inches='tight') 



#%%  plot of the regression training data and model with means and methods
plt.figure()
plt.plot(WFopen[:,0], WFopen[:,1], color='b', linestyle='None', marker='.', alpha=alFA) 
#plt.plot(x, y_true, label='true')
Tmin=WFopen[:,0].min()-5
Tmax=WFopen[:,0].max()+5


Tfit=range(np.int(Tmin),np.int(Tmax),1)
kWfit=res.params[0]+res.params[1]*(Tfit)+res.params[2]*(np.maximum(Tfit,BPbest)-BPbest)
Tvals=np.arange(Tmin,Tmax,.1)
breaks = [0,BPbest]
Treg = np.column_stack([np.maximum(0, Tvals - knot) for knot in breaks])
Treg = sm.add_constant(Treg)
kWreg=np.dot(Treg,res.params) 

plt.plot(Tfit, kWfit, 'orange', linewidth=2, label='WLS')
plt.plot(Tvals, kWreg, 'DeepPink', linewidth=2, label='WLS_dot')
plt.plot(range(trmin,trmax,trbin),tempWFopen['kW_mean'], '.', c='lime',label='kW_means') 
plt.legend() 
plt.title('kW vs OADB Segmented Regression with Statsmodels WLS(OLS)') 
plt.show()
plotname='regression model with linear selected fit data'
outputPath=shareDrivePGE+'PGandE/output/'
if outputGraphs: plt.savefig(outputPath+str(st_YrTr)+'_to_'+str(en_YrTr)+'_'+plotname+'.png',
                             bbox_to_inches='tight') 


#%% predict a kW 2014 based on 2012,2013 regression and plot all 3 datasets

evWFopenDF['kWpredDist']=evWFopenDF.temp*0
i=0
for evTemp in evWFopenDF.temp:
    if evTemp>BPbest:
        Tbp=evTemp-BPbest
    else:
        Tbp=0
    Ti=[1,evTemp,Tbp]
    kWav=np.dot(Ti,res.params)
    evWFopenDF.kWpredDist[i]=kWav+np.random.normal(size=1)*tempstatsDF.kW_sDev.mean()
    i+=1
plt.figure()
plt.plot(WFopenDF.temp, WFopenDF.kW, color='g', linestyle='None', marker='.', alpha=alFA)

plt.plot(evWFopenDF.temp, evWFopenDF.kW, color='b', linestyle='None', marker='.', alpha=alFA)

plt.plot(evWFopenDF.temp, evWFopenDF.kWpredDist, color='r', linestyle='None', marker='.', alpha=alFA)
plotname='training_predicted_actual data clouds'
outputPath=shareDrivePGE+'PGandE/output/'
if outputGraphs: plt.savefig(outputPath+str(st_YrEv)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

#%% time series plotting

#Build predicted time series
pgeDFev['pgkW_Site_pred']=[np.nan]*len(pgeDFev)
pgeDFev['pgkW_Site_pravg']=[np.nan]*len(pgeDFev)
#build open hour data
i=0
hopen=0
htrans=0
hclosed=0

for vals in pgeDFev.index:#[tstar:tend]:
    if (vals.hour>=oHour) & (vals.hour<teHour):
        if pgeDFev.pgOA_Temp[i]>BPbest:
            Tbp=pgeDFev.pgOA_Temp[i]-BPbest
        else:
            Tbp=0
        Ti=[1,pgeDFev.pgOA_Temp[i],Tbp]
        pgeDFev['pgkW_Site_pravg'][i]=np.dot(Ti,res.params)
        pgeDFev['pgkW_Site_pred'][i]=pgeDFev.pgkW_Site_pravg[i]+np.random.normal(size=1)*tempstatsDF.kW_sDev.mean()
        hopen=hopen+1
    elif ((vals.hour>=tmHour) & (vals.hour<oHour)) | ((vals.hour>=teHour) & (vals.hour<cHour)):
        Ti=[1,pgeDFev.pgOA_Temp[i]]
        pgeDFev['pgkW_Site_pravg'][i]=np.dot(Ti,restr.params)        
        pgeDFev['pgkW_Site_pred'][i]=pgeDFev.pgkW_Site_pravg[i]+np.random.normal(size=1)*tempstatsDFtr.kW_sDev.mean()
        htrans=htrans+1
    else:
        Ti=[1,pgeDFev.pgOA_Temp[i]]
        pgeDFev['pgkW_Site_pravg'][i]=np.dot(Ti,rescl.params)        
        pgeDFev['pgkW_Site_pred'][i]=pgeDFev.pgkW_Site_pravg[i]+np.random.normal(size=1)*tempstatsDFcl.kW_sDev.mean()
        hclosed=hclosed+1
    #print(Ti,pgeDFev.pgkW_Site_pravg[i],pgeDFev.pgkW_Site_pred[i])
    i=i+1
print(hopen,htrans,hclosed)

#%%

dstart =2
dnum = 10
tstar = (39*7*96) - 120 + (96 * dstart) #time that starts october plus day of oct to start at
tend = tstar + 96 * dnum
#i=tstar
RAoffset = 60
fig=plt.figure()
fig.set_size_inches(8,6)
ax=fig.add_subplot(111)
plt.plot(pgeDFev.index[tstar:tend],pgeDFev.pgkW_Site_pred[tstar:tend],'r-')
plt.plot(pgeDFev.index[tstar:tend],pgeDFev.pgkW_Site[tstar:tend],'g-')
#plt.plot(pardf.index[tstar:tend],pardf.kW_AC31[tstar:tend],'b-')
#plt.plot(df.index[tstar:tend],df.kW_System[tstar:tend],'m-')
#plt.plot(pardf.index[tstar:tend],(pardf.pRA_Temp[tstar:tend]-
#         ([RAoffset]*len(pardf.index[tstar:tend]))),'y-')
#plt.plot(df.index[tstar:tend],(df.RA_Temp[tstar:tend]-
#         ([RAoffset]*len(df.index[tstar:tend]))),'c-')
#plt.plot(df.index[tstar:tend],(df.PUMP_Status[tstar:tend]*5),'k-')
#plt.plot(df.index[tstar:tend],(df.CT_C1_Status[tstar:tend]*8),'b-')
#plt.plot(df.index[tstar:tend],(df.CT_C2_Status[tstar:tend]*12),'r-')
plt.xticks(rotation='vertical')
plt.show()
plotname='day time series 3prt regression method'
obsmonth=pgeDFev.index[tstar].month
if obsmonth>=10:
    placeHolder=''
else:
    placeHolder='0'
outputPath=shareDrivePGE+'PGandE/output/'
if outputGraphs: plt.savefig(outputPath+str(st_YrEv)+'_'+str(placeHolder)+\
                             str(obsmonth)+'_'+str(dnum)+plotname+'.png',
                             bbox_to_inches='tight')

#%% ARMA example code

#starttr=1
#endtr=5
#
#ARendog=pgeDFtr[['pgkW_Site','pgOA_Temp']]
#
#ARexog=np.array([pgeDFtr.pgOA_Temp[int(starttr*96):int(endtr*96)]])
#ARindex=np.array([pgeDFtr.index[int(starttr*96):int(endtr*96)]])
#
#model = sm.tsa.VAR(ARendog[int(starttr*96):int(endtr*96)])
#
##%%
#
#mdata = sm.datasets.macrodata.load_pandas().data
#
## prepare the dates index
#dates = mdata[['year', 'quarter']].astype(int).astype(str)
#quarterly = dates["year"] + "Q" + dates["quarter"]
#from statsmodels.tsa.base.datetools import dates_from_str
#quarterly = dates_from_str(quarterly)
#
#mdata = mdata[['realgdp','realcons','realinv']]
#mdata.index = pd.DatetimeIndex(quarterly)
#data = np.log(mdata).diff().dropna()
#
## make a VAR model
#model = sm.tsa.VAR(data)
##%% seg reg example code
#
#nobs =5000
#sig_e = 0.5 
#x = np.random.uniform(0, 10, nobs) 
#x.sort() 
#breaks = [0, 2, 5, 8]  # 0 adds slope for entire array 
#
#exog = np.column_stack([np.maximum(0, x - knot) for knot in breaks]) 
#
#exog = sm.add_constant(exog) 
#beta = np.array([1, 0.5, -0.8, 0.2, 1.]) 
#y_true = np.dot(exog, beta) 
#endog = y_true + sig_e * np.random.randn(nobs) 
#
#weights = np.ones(nobs) 
#weights[nobs//2:] *= 1.5**2 
#
#res = sm.OLS(endog, exog).fit()#, weights=weights).fit() 
#print('params:')
#print('DGP: ', beta)
#print('WLS: ', res.params)
#print('\nslopes:')
#print('DGP: ', beta[1:].cumsum())
#print('WLS: ', res.params[1:].cumsum())
#print(res.summary())
#plt.figure()
#plt.plot(x, endog, 'o', alpha=0.5) 
#plt.plot(x, y_true, label='true') 
#plt.plot(x, res.fittedvalues, label='WLS') 
#plt.legend() 
#plt.title('Segmented Regression with Statsmodels WLS') 
#plt.show() 