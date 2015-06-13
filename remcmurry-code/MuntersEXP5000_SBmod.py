# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\Users\rmcmurry\.spyder2\.temp.py
"""
'''
Created on Jul 21,2014

@author: rmcmurry
'''
''' Western Cooling Challenge Munters EXP 5000 DOAS Whole Foods San Ramon'''
'''currently woking on supplementary mode tagging sections'''


#%% Import Libraries

#import matplotlib
#matplotlib.use('Qt4Agg')
#matplotlib.rcParams['backend.qt4']='PySide'

import matplotlib.pyplot as plt
#import matplotlib.dates as dates
import matplotlib.ticker as mtick
#import pylab
import psychropy as ps
import datetime as dt
#from datetime import time
import numpy as np
import itertools
import time
import os
import pandas as pd
import statsmodels.api as sm
import calendar as cd
plt.close('all')


#%% Stacked bar class
'''
###############################################################################
#                                                                             #
#    stackedBarGraph.py - code for creating purdy stacked bar graphs          #
#                                                                             #
###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

__author__ = "Michael Imelfort"
__copyright__ = "Copyright 2014"
__credits__ = ["Michael Imelfort"]
__license__ = "GPL3"
__version__ = "0.0.1"
__maintainer__ = "Michael Imelfort"
__email__ = "mike@mikeimelfort.com"
__status__ = "Development"

###############################################################################
'''

class StackedBarGrapher:
    """Container class"""

    def __init__(self): pass

    def demo(self):
        d = np.array([[101.,0.,0.,0.,0.,0.,0.],
                      [92.,3.,0.,4.,5.,6.,0.],
                      [56.,7.,8.,9.,23.,4.,5.],
                      [81.,2.,4.,5.,32.,33.,4.],
                      [0.,45.,2.,3.,45.,67.,8.],
                      [99.,5.,0.,0.,0.,43.,56.]])

        d_heights = [1.,2.,3.,4.,5.,6.]
        d_widths = [.5,1.,3.,2.,1.,2.]
        d_labels = ["fred","julie","sam","peter","rob","baz"]
        d_colors = ['#2166ac', '#fee090', '#fdbb84', '#fc8d59',
                    '#e34a33', '#b30000', '#777777']
        gap = 0.05

        fig = plt.figure()
        ax1 = fig.add_subplot(321)
        self.stackedBarPlot(ax1,
                            d,
                            d_colors,
                            edgeCols=['#000000']*7,
                            xLabels=d_labels,
                            )
        plt.title("Straight up stacked bars")

        ax2 = fig.add_subplot(322)
        self.stackedBarPlot(ax2,
                            d,
                            d_colors,
                            edgeCols=['#000000']*7,
                            xLabels=d_labels,
                            scale=True
                            )
        plt.title("Scaled bars")

        ax3 = fig.add_subplot(323)
        self.stackedBarPlot(ax3,
                            d,
                            d_colors,
                            edgeCols=['#000000']*7,
                            xLabels=d_labels,
                            heights=d_heights,
                            yTicks=7,
                            )
        plt.title("Bars with set heights")

        ax4 = fig.add_subplot(324)
        self.stackedBarPlot(ax4,
                            d,
                            d_colors,
                            edgeCols=['#000000']*7,
                            xLabels=d_labels,
                            yTicks=7,
                            widths=d_widths,
                            scale=True
                            )
        plt.title("Scaled bars with set widths")

        ax5 = fig.add_subplot(325)
        self.stackedBarPlot(ax5,
                            d,
                            d_colors,
                            edgeCols=['#000000']*7,
                            xLabels=d_labels,
                            gap=gap
                            )
        plt.title("Straight up stacked bars + gaps")

        ax6 = fig.add_subplot(326)
        self.stackedBarPlot(ax6,
                            d,
                            d_colors,
                            edgeCols=['#000000']*7,
                            xLabels=d_labels,
                            scale=True,
                            gap=gap,
                            endGaps=True
                            )
        plt.title("Scaled bars + gaps + end gaps")

        # We change the fontsize of minor ticks label
        fig.subplots_adjust(bottom=0.4)

        plt.tight_layout()
        plt.show()
        plt.close(fig)
        del fig

    def stackedBarPlot(self,
                       ax,               # axes to plot onto
                       data,             # data to plot
                       cols,             # colors for each level
                       labels,           # stack item labels
                       negStack=False,   # set to True if plotting negative and positive values
                       xLabels = None,   # bar specific labels
                       yTicks = 6.,      # information used for making y ticks ["none", <int> or [[tick_pos1, tick_pos2, ... ],[tick_label_1, tick_label2, ...]]
                       edgeCols=None,    # colors for edges
                       showFirst=-1,     # only plot the first <showFirst> bars
                       scale=False,      # scale bars to same height
                       widths=None,      #set widths for each bar
                       heights=None,     # set heights for each bar
                       ylabel='',        # label for x axis
                       xlabel='',        # label for y axis
                       lblsize=8,        # set label font size
                       gap=0.,           # gap between bars
                       endGaps=False     # allow gaps at end of bar chart (only used if gaps != 0.)
                       ):

        # data fixeratering

        # make sure this makes sense
        if showFirst != -1:
            showFirst = np.min([showFirst, np.shape(data)[0]])
            data_copy = np.copy(data[:showFirst]).transpose().astype('float')
            data_shape = np.shape(data_copy)
            if heights is not None:
                heights = heights[:showFirst]
            if widths is not None:
                widths = widths[:showFirst]
            showFirst = -1
        else:
            data_copy = np.copy(data).transpose()
        data_shape = np.shape(data_copy)

        # determine the number of bars and corresponding levels from the shape of the data
        num_bars = data_shape[1]
        levels = data_shape[0]
        
        if widths is None:
            widths = np.array([1] * num_bars)
            x = np.arange(num_bars)
        else:
            x = [0]
            for i in range(1, len(widths)):
                x.append(x[i-1] + (widths[i-1] + widths[i])/2)
               
        # stack the data --
        # replace the value in each level by the cumulative sum of all preceding levels
        data_stack = np.reshape([float(i) for i in np.ravel(np.cumsum(data_copy, axis=0))], data_shape)
        # crate a plottign framework for negative stacked values
        print "i got here"
        if negStack:
            datum_bar = np.zeros(num_bars)
            dat_bot =   np.zeros((levels,num_bars))
            dat_top =   np.zeros((levels,num_bars))
            if scale:
                print "WARNING: setting scale does not work with negStack True."
                scale = False
            elif heights is not None:
                print "WARNING: setting heights does not work with negStack True."
                heights = None
            dat_bot[0]=[min(datum_bar[i],data_copy[0, i]) for i in range(num_bars)]
            stack_bot=dat_bot
            dat_top[0]=[max(datum_bar[i],data_copy[0, i]) for i in range(num_bars)]
            stack_top=dat_top
            
            # fill negStack dat and stack matracies for plotting dat_bot is most important
            for lev in np.arange(1,levels):
                for i in range(num_bars):
                    pos=data_copy[lev, i]
                    #print pos
                    if (pos >= 0):
                        dat_bot[lev, i]=min(stack_top[lev-1, i],dat_top[lev-1, i])
                        dat_top[lev, i]=stack_top[lev-1, i]+pos
                    else: 
                        dat_bot[lev, i]=stack_bot[lev-1, i]+pos
                        dat_top[lev, i]=max(stack_bot[lev-1, i],dat_bot[lev-1, i])
                    stack_bot[lev, i]=min(dat_bot[lev, i],stack_bot[lev-1, i])    
                    stack_top[lev, i]=max(dat_top[lev, i],stack_top[lev-1, i])
            # correction lines
            dat_bot[data_copy<0]=stack_bot[data_copy<0]
            dat_bot[data_copy>=0]=stack_top[data_copy>=0]-data_copy[data_copy>=0]
            dat_top[data_copy<0]=stack_bot[data_copy<0]-data_copy[data_copy<0]
            dat_top[data_copy>=0]=stack_top[data_copy>=0]
            #print dat_bot
            #print stack_bot
            #print dat_top
            #print stack_top
            #print data_copy
            # From what i can tell the dat_bot and dat_top matrixes are still
            # not filling correctly. Added last 4 lines to correct for them. 
            # Now the important one 'dat_bot' is working but all the others are 
            # the same. In the current itteration this plots the bars correctly
            #so leaving it alone for now.
              
        # scale the data is needed if you are negStacking this may not work well
        if scale:
            data_copy /= data_stack[levels-1]
            data_stack /= data_stack[levels-1]
            if heights is not None:
                print "WARNING: setting scale and heights does not make sense."
                heights = None
        elif heights is not None:  #if you are negStack otion this may be off
            data_copy /= data_stack[levels-1]
            data_stack /= data_stack[levels-1]
            for i in np.arange(num_bars):
                data_copy[:,i] *= heights[i]
                data_stack[:,i] *= heights[i]

        # ticks
        if negStack:
            if yTicks is not "none":
                # it is either a set of ticks or the number of auto ticks to make
                real_ticks = True
                try:
                    k = len(yTicks[1])
                except:
                    real_ticks = False

                if not real_ticks:
                    yTicks = float(yTicks)
                    # space the ticks along the y axis
                    y_tick_bot=np.min(dat_bot)*np.ones(yTicks)
                    y_ticks_at = y_tick_bot + np.arange(yTicks)/(yTicks-1)*(-np.min(dat_bot)+np.max(dat_bot+data_copy))
                    y_tick_labels = np.array([str(i) for i in y_ticks_at])
                    yTicks=(y_ticks_at, y_tick_labels)
        else:
            if yTicks is not "none":
                # it is either a set of ticks or the number of auto ticks to make
                real_ticks = True
                try:
                    k = len(yTicks[1])
                except:
                    real_ticks = False

                if not real_ticks:
                    yTicks = float(yTicks)
                    if scale:
                        # make the ticks line up to 100 %
                        y_ticks_at = np.arange(yTicks)/(yTicks-1)
                        y_tick_labels = np.array(["%0.2f"%(i * 100) for i in y_ticks_at])
                    else:
                        # space the ticks along the y axis
                        y_ticks_at = np.arange(yTicks)/(yTicks-1)*np.max(data_stack)
                        y_tick_labels = np.array([str(i) for i in y_ticks_at])
                        yTicks=(y_ticks_at, y_tick_labels)

        # plot

        if edgeCols is None:
            edgeCols = ["none"]*len(cols)

        # take cae of gaps
        gapd_widths = [i - gap for i in widths]

        # bars
        if negStack:
            dat_abs=data_copy
            dat_abs[dat_abs<0]=-dat_abs[dat_abs<0]
            for i in range(levels):                
                ax.bar(x,
                       dat_abs[i],
                       bottom=dat_bot[i],
                       color=cols[i],
                       edgecolor=edgeCols[i],
                       width=gapd_widths,
                       linewidth=0.5,
                       align='center',
                       label=labels[i]
                       )
        else:
            ax.bar(x,
                   data_stack[0],
                   color=cols[0],
                   edgecolor=edgeCols[0],
                   width=gapd_widths,
                   linewidth=0.5,
                   align='center',
                   label=labels[0]
                   )
    
            for i in np.arange(1,levels):
                ax.bar(x,
                       data_copy[i],
                       bottom=data_stack[i-1],
                       color=cols[i],
                       edgecolor=edgeCols[i],
                       width=gapd_widths,
                       linewidth=0.5,
                       align='center',
                       label=labels[i]
                       )

        # borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # make ticks if necessary
        if yTicks is not "none":
            ax.tick_params(axis='y', which='both', labelsize=lblsize, direction="out")
            ax.yaxis.tick_left()
            plt.yticks(yTicks[0], yTicks[1])
        else:
            plt.yticks([], [])

        if xLabels is not None:
            ax.tick_params(axis='x', which='both', labelsize=lblsize, direction="out")
            ax.xaxis.tick_bottom()
            plt.xticks(x, xLabels, rotation='vertical')
        else:
            plt.xticks([], [])

        # limits
        if endGaps:
            ax.set_xlim(-1.*widths[0]/2. - gap/2., np.sum(widths)-widths[0]/2. + gap/2.)
        else:
            ax.set_xlim(-1.*widths[0]/2. + gap/2., np.sum(widths)-widths[0]/2. - gap/2.)
        ax.set_ylim(0, yTicks[0][-1])#np.max(data_stack))

        # labels
        if xlabel != '':
            plt.xlabel(xlabel)
        if ylabel != '':
            plt.ylabel(ylabel)
        


#%% Input site
jobSite='Whole Foods San Ramon' #'PG&E project at San Ramon Whole Foods'
obsMonth=10
if obsMonth < 10: 
    placeHolder=0
else:
    placeHolder=''
obsYear=2014
endFileName=' Month Data Munters EXP 5000 at Whole Foods San Ramon.csv'

obsType='Month'
setThresholds=0

#==============================================================================
# Graphing and Output options
#==============================================================================

outputGraphs=1
smallBarColorMap=0
createCSV=1
alFA=0.35
wFig=8
hFig=6
tl_size=15


#%% Data input and output path

shareDrive='S:/Current Projects/Western Cooling Challenge/'+\
           'Field Installations/Munters EXP 5000 at Whole Foods San Ramon/'+\
           'data analysis/month data/'
inputPath=shareDrive+str(obsYear)+str(placeHolder)+str(obsMonth)+endFileName

outputPath=shareDrive+str(obsYear)+'_'+str(placeHolder)+str(obsMonth)+'/' 

if not os.path.isdir(outputPath): os.mkdir(outputPath)
    
#%% Ready in month file  

newHeader=['Timestamp','TZ','OA_Temp','OA_RH','RA_Temp','RA_RH','SA_Temp',
           'SA_RH','dPsa_Volts','dPsa_fan_inH2O','Vgas','Vwater_Gal',
           'AOrtu_fan1_Volts','CT_C1_A','CT_C2_A','CT_F1_A','CT_F2_A',
           'CT_PUMP_A','kW_System','kW_System_Avg','System_App_PF',
           'L1_Volts','L2_Volts','L3_Volts','L1_Amps','L2_Amps','L3_Amps',
           'Tsump_Temp','TsuctionC1_Temp','TsuctionC2_Temp',
           'TdischargeC1_Temp','TdischargeC2_Temp','TliquidC1_Temp',
           'TliquidC2_Temp','PsuctionC1_bar','PsuctionC2_bar',
           'PdischargeC1_bar','PdischargeC2_bar'
           ]

df=pd.read_csv(inputPath,header=0,names=newHeader,index_col='Timestamp',
               parse_dates=True,low_memory=False)
               
dfi=pd.DatetimeIndex(((df.index.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))

df.index=dfi
#the final two lines above round the time indexes to the nearest min

#%%read in water regression
#uses header fom input file
WRfname='water/'+str(201406)+' Water Data Munters EXP 5000 at Whole Foods San Ramon.csv'
WRpath=shareDrive+WRfname

WRDF=pd.read_csv(WRpath,header=0,names=newHeader,index_col='Timestamp',
               parse_dates=True,low_memory=False)
               
WRDFi=pd.DatetimeIndex(((WRDF.index.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))

WRDF.index=WRDFi

WRdata={}
WRdata['OA_Temp']=WRDF.OA_Temp
WRdata['Vwater_Gal']=WRDF.Vwater_Gal
WRdata['kW_System']=WRDF.kW_System
WRdf=pd.DataFrame(WRdata)
WRdf=WRdf.resample('h',how='mean')
WRdf=WRdf.dropna()

#%% Supplemental Data input and output path
shareDriveSup='S:/Current Projects/Western Cooling Challenge/'+\
             'Field Installations/Munters EXP 5000 at Whole Foods San Ramon/'+\
             'data analysis/month data/supplemental/'

parFileName='_RAM interval data with enviro.csv'

# alligne time series data and us CPC_dredger.py to make these files
cpcFileName='_CPC_month.csv'

pgeFileName='_PGE_int.csv'        

parinputPath=shareDriveSup+'parasense/'+str(obsYear)+str(placeHolder)+\
             str(obsMonth)+parFileName
             
cpcinputPath=shareDriveSup+'CPC/'+str(obsYear)+str(placeHolder)+\
             str(obsMonth)+cpcFileName
             
pgeinputPath=shareDriveSup+'PGandE/'+str(obsYear)+pgeFileName

parFile=True
cpcFile=True
pgeFile=True
filNAN=0
supsetThresh=1
dataFill='bfill'# options from {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}
                # hope to add if wanted 'mid'=centered fill,
                # 'int'=interpolated fill 

#%% supplementary data loading Parasense
if parFile:
    parHeader=['Timestamp','pRA_Temp','pOA_Temp','pRA_DP','pOA_DP','pRA_RH',
               'pOA_RH',
               'kwh_Main','kW_Main','kVA_Main','PF_Main',
               'kwh_RackA','kW_RackA','kVA_RackA','PF_RackA',
               'kwh_RackB','kW_RackB','kVA_RackB','PF_RackB',
               'kwh_Panel','kW_Panel','kVA_Panel','PF_Panel',
               'kwh_AC36','kW_AC36','kVA_AC36','PF_AC36',
               'kwh_AC30','kW_AC30','kVA_AC30','PF_AC30',
               'kwh_AC31','kW_AC31','kVA_AC31','PF_AC31'
               ]

    pardf=pd.read_csv(parinputPath,header=0,names=parHeader,index_col='Timestamp',
                      parse_dates=True,low_memory=False)

    p1mdf=pardf.asfreq('1Min', method=dataFill)

    pardf=p1mdf.reindex(index=df.index)

#%% supplementary data loading CPC
if cpcFile:    
    cpcHeader=['Timestamp','cOA_DB','cOA_RH','RA_RH_AH1',
               'SA_DB_AH1','IEC_stat','C1stat','C2stat','HEATstat',
               'SA_DB_AC1FS','C1stat_AC1FS','C2stat_AC1FS','HEATstatAC1FS',
               'SA_DB_AC2MFS','C1stat_AC2MFS','C2stat_AC2MFS','HEATstatAC2MFS',
               'SA_DB_AC3MRS','C1stat_AC3MRS','C2stat_AC3MRS','HEATstatAC3MRS',
               'SA_DB_AC4RS','C1stat_AC4RS','C2stat_AC4RS','HEATstatAC4RS',
               ]
    
    cpcdf=pd.read_csv(cpcinputPath,header=0,names=cpcHeader,index_col='Timestamp',
                   parse_dates=True,low_memory=False)
    

#%% supplementary data loading PGandE plus sam ramon weater from Rob Davis lab

if pgeFile:
    pgeHeader=['Timestamp','pgOA_Temp','pgkW_Site','kVAR','KVA','pf']
    
    pgeDF=pd.read_csv(pgeinputPath,header=0,names=pgeHeader,index_col='Timestamp',
                   parse_dates=True,low_memory=False)
                   
    pgedf=pgeDF[df.index[0]:(df.index[-1]+dt.timedelta(0,60))]
                   
    pg1mdf=pgedf.asfreq('1Min', method=dataFill)
    
    pg1mdf=pg1mdf[0:-1]

    pgedf=pg1mdf.reindex(index=df.index)
    
#%% Let the games begin
elevation=475 #ft on roof of SR WFM 455+20
mDotWater=5987.6 #lb/hr need to get this or use a tabulated value

feet2meter=0.3048
min2hour=60
Pa_psi=1.45*10**-4 ## psi/Pa
atmPressure=101325*(1-(2.25577*10**-5)*(elevation*feet2meter))**5.25588*Pa_psi


#%% Parasense data Thresholds
#if supsetThresh:
#    for supcomponent in ['kW_AC36','kW_AC30','kW_AC31','kW_Main','kW_RackA',
#                         'kW_RackB','kW_Panel']:
#        plt.figure()    
#        pardf[supcomponent].hist(bins=100)
#        plt.title(supcomponent)
#        plt.show()
#%% Parasense data Mode tagging
               
pardf['opMode36']=['Transient']*len(df)
pardf['opModeStatus36']=['Transient']*len(df)
pardf['opMode30']=['Transient']*len(df)
pardf['opModeStatus30']=['Transient']*len(df)
pardf['opMode31']=['Transient']*len(df)
pardf['opModeStatus31']=['Transient']*len(df)
      
#Parasense thresholds
AC36Th_Vent=1
AC36Th_IEC=4
AC36Th_DX1=7
AC36Th_DX2=9.5
AC30Th_Vent=0.6 
AC31Th_Vent=0.6
AC30Th_DX1=2.5 
AC31Th_DX1=2.5
AC30Th_DX2=6.5 
AC31Th_DX2=6.5

## Correcting data based off of inspection
AC36St_Vent=[0]*len(df)
AC36St_IEC=[0]*len(df)
AC36St_DX1=[0]*len(df)
AC36St_DX2=[0]*len(df)
AC36_Status=[0]*len(df)

AC30St_Vent=[0]*len(df)
AC30St_DX1=[0]*len(df)
AC30St_DX2=[0]*len(df)
AC30_Status=[0]*len(df)

AC31St_Vent=[0]*len(df)
AC31St_DX1=[0]*len(df)
AC31St_DX2=[0]*len(df)
AC31_Status=[0]*len(df)

#==============================================================================
# Use binary counters to determine mode states should only be able to have 
# (2^n)-1 as sum of counters if working right
#==============================================================================

for x,iAC36,iAC30,iAC31 in itertools.izip(
        range(len(pardf)),pardf.kW_AC36.values,pardf.kW_AC30.values,
        pardf.kW_AC31.values):
    
    if iAC36 > AC36Th_Vent:
        AC36St_Vent[x]=1
    if iAC36 > AC36Th_IEC:
        AC36St_IEC[x]=2
    if iAC36 > AC36Th_DX1:
        AC36St_DX1[x]=4
    if iAC36 > AC36Th_DX2:
        AC36St_DX2[x]=8
    AC36_Status[x]=AC36St_Vent[x]+AC36St_IEC[x]+AC36St_DX1[x]+AC36St_DX2[x]
    
    if iAC30 > AC30Th_Vent:
        AC30St_Vent[x]=1
    if iAC30 > AC30Th_DX1:
        AC30St_DX1[x]=2
    if iAC30 > AC30Th_DX2:
        AC30St_DX2[x]=4
    AC30_Status[x]=AC30St_Vent[x]+AC30St_DX1[x]+AC30St_DX2[x]
    
    if iAC31 > AC31Th_Vent:
        AC31St_Vent[x]=1
    if iAC31 > AC31Th_DX1:
        AC31St_DX1[x]=2
    if iAC31 > AC31Th_DX2:
        AC31St_DX2[x]=4
    AC31_Status[x]=AC31St_Vent[x]+AC31St_DX1[x]+AC31St_DX2[x]

# summing counters    
pardf['AC36_Status']=AC36_Status
pardf['AC30_Status']=AC30_Status
pardf['AC31_Status']=AC31_Status

#defining component boolean states
AC36OFF=(pardf.AC36_Status==0) & (pardf.kW_AC36<AC36Th_Vent)
AC36VENT=(pardf.AC36_Status==1) & (pardf.kW_AC36>=AC36Th_Vent)
AC36IEC=(pardf.AC36_Status==3) & (pardf.kW_AC36>=AC36Th_IEC)
AC36DX1=(pardf.AC36_Status==7) & (pardf.kW_AC36>=AC36Th_DX1)
AC36DX2=(pardf.AC36_Status==15) & (pardf.kW_AC36>=AC36Th_DX2)

AC30OFF=(pardf.AC30_Status==0) & (pardf.kW_AC30<AC30Th_Vent)
AC30VENT=(pardf.AC30_Status==1) & (pardf.kW_AC30>=AC30Th_Vent)
AC30DX1=(pardf.AC30_Status==3) & (pardf.kW_AC30>=AC30Th_DX1)
AC30DX2=(pardf.AC30_Status==7) & (pardf.kW_AC30>=AC30Th_DX2)

AC31OFF=(pardf.AC31_Status==0) & (pardf.kW_AC31<AC31Th_Vent)
AC31VENT=(pardf.AC31_Status==1) & (pardf.kW_AC31>=AC31Th_Vent)
AC31DX1=(pardf.AC31_Status==3) & (pardf.kW_AC31>=AC31Th_DX1)
AC31DX2=(pardf.AC31_Status==7) & (pardf.kW_AC31>=AC31Th_DX2)

#AC36ON=~ AC36OFF
#AC30ON=~ AC30OFF
#AC31ON=~ AC31OFF

#defining modes by component boolean states
#General/Cooling modes

#OFF=            f1OFF
#VENT=           c1OFF & c2OFF & f1ON  & iecOFF & h1OFF #& h2OFF
#IEC__ONLY=      c1OFF & c2OFF & f1ON  & iecON  & h1OFF #& h2OFF
#IEC__DX1=       c1ON  & c2OFF & f1ON  & iecON  & h1OFF #& h2OFF
#IEC__DX2=       c1ON  & c2ON  & f1ON  & iecON  & h1OFF #& h2OFF
#
##Heating Modes
#HEAT1=          c1OFF & c2OFF & f1ON  & iecOFF & h1ON  #& h2OFF
##HEAT2=          c1OFF & c2OFF & f1ON  & iecOFF & h1ON  & h2ON
#
##Cooling plus Heat1
#IEC__HEAT1=     c1OFF & c2OFF & f1ON  & iecON  & h1ON  #& h2OFF
#IEC__DX1__HEAT1=c1ON  & c2OFF & f1ON  & iecON  & h1ON  #& h2OFF
#IEC__DX2__HEAT1=c1ON  & c2ON  & f1ON  & iecON  & h1ON  #& h2OFF

#Cooling plus Heat2
#IEC__HEAT2=     c1OFF & c2OFF & f1ON  & iecON  & h1ON  & h2ON
#IEC__DX1__HEAT2=c1ON  & c2OFF & f1ON  & iecON  & h1ON  & h2ON
#IEC__DX2__HEAT2=c1ON  & c2ON  & f1ON  & iecON  & h1ON  & h2ON

    

# AC36 mode labeling and SS determination 
allModes36=[AC36OFF,AC36VENT,AC36IEC,AC36DX1,AC36DX2]

allModeNames36=['OFF','VENT','IEC','IEC and DX1','IEC and DX2']



transTime36=14
for mode,modeName in itertools.izip(allModes36,allModeNames36):
    pardf.loc[(mode.index[mode.values==True]),('opMode36')]=modeName

for i in xrange(len(pardf)):
    if i>=transTime36 and i<=len(pardf):
        currentMode=pardf.opMode36[i]               
        steadyState=[currentMode]*transTime36==[x for x 
                                              in pardf.opMode36[range(i-transTime36,i,1)]]
        
        if steadyState:
            pardf.loc[pardf.index[i],('opModeStatus36')]='SS'
        
        
# AC30 mode labeling and SS determination 
allModes30=[AC30OFF,AC30VENT,AC30DX1,AC30DX2]

allModeNames30=['OFF','VENT','DX1','DX2']
        
transTime30=10
for mode,modeName in itertools.izip(allModes30,allModeNames30):
    pardf.loc[(mode.index[mode.values==True]),('opMode30')]=modeName

for i in xrange(len(pardf)):
    if i>=transTime30 and i<=len(pardf):
        currentMode=pardf.opMode30[i]               
        steadyState=[currentMode]*transTime30==[x for x 
                                              in pardf.opMode30[range(i-transTime30,i,1)]]
        
        if steadyState:
            pardf.loc[pardf.index[i],('opModeStatus30')]='SS'

# AC31 mode labeling and SS determination        
allModes31=[AC31OFF,AC31VENT,AC31DX1,AC31DX2]

allModeNames31=['OFF','VENT','DX1','DX2']

transTime31=10
for mode,modeName in itertools.izip(allModes31,allModeNames31):
    pardf.loc[(mode.index[mode.values==True]),('opMode31')]=modeName

for i in xrange(len(pardf)):
    if i>=transTime31 and i<=len(pardf):
        currentMode=pardf.opMode31[i]               
        steadyState=[currentMode]*transTime31==[x for x 
                                              in pardf.opMode31[range(i-transTime31,i,1)]]
        
        if steadyState:
            pardf.loc[pardf.index[i],('opModeStatus31')]='SS'


#%% CPC mode tagging
'''\
               'Timestamp','cOA_DB','cOA_RH','RA_RH_AH1',
               'SA_DB_AH1','IEC_stat','C1stat','C2stat','HEATstat',
               'SA_DB_AC1FS','C1stat_AC1FS','C2stat_AC1FS','HEATstatAC1FS',
               'SA_DB_AC2MFS','C1stat_AC2MFS','C2stat_AC2MFS','HEATstatAC2MFS',
               'SA_DB_AC3MRS','C1stat_AC3MRS','C2stat_AC3MRS','HEATstatAC3MRS',
               'SA_DB_AC4RS','C1stat_AC4RS','C2stat_AC4RS','HEATstatAC4RS',\
'''

if setThresholds:
    for component in ['AH1_IEC', 'AH1_C1', 'AH1_C2', 'AH1_Heat',
                      'AC1_C1', 'AC1_C2', 'AC1_Heat',
                      'AC2_C1', 'AC2_C2', 'AC2_Heat',
                      'AC3_C1', 'AC3_C2', 'AC3_Heat',
                      'AC4_C1', 'AC4_C2', 'AC4_Heat']:
        plt.figure()    
        df[component].hist(bins=100)
        plt.title(component)
  
cpcdf['AH1opMode']=['Transient']*len(df)
cpcdf['AH1opModeStatus']=['Transient']*len(df)
cpcdf['AC1opMode']=['Transient']*len(df)
cpcdf['AC1opModeStatus']=['Transient']*len(df)
cpcdf['AC2opMode']=['Transient']*len(df)
cpcdf['AC2opModeStatus']=['Transient']*len(df)
cpcdf['AC3opMode']=['Transient']*len(df)
cpcdf['AC3opModeStatus']=['Transient']*len(df)
cpcdf['AC4opMode']=['Transient']*len(df)
cpcdf['AC4opModeStatus']=['Transient']*len(df)      
#munters thresholds
f1Thresh=3.5
f2Thresh=1 
c1Thresh=4.5
c2Thresh=4.5 
pumpThresh=2 
h1Thresh=35   #signifies tagging if within heat band of a SA 35+ deg above OA
#h2Thresh=45   #signifies tagging if within heat band of a SA 45+ deg above OA, 
#decided to combine the two heating due to large amount of trasiency
heat_band=5#the minute width of the heat band(checks values +- from current pt)

## Correcting data based off of inspection

CT_C1_Status=[0]*len(df)
CT_C2_Status=[0]*len(df)
CT_F1_Status=[0]*len(df)
CT_F2_Status=[0]*len(df)
PUMP_Status=[0]*len(df)
Heat1_Status=[0]*len(df)
#Heat2_Status=[0]*len(df)

for x,iC1,iC2,iF1,iF2,iPump,iOA,iSA,iPow in itertools.izip(
                range(len(df)),df.CT_C1_A.values,df.CT_C2_A.values,
                df.CT_F1_A.values,df.CT_F2_A.values,df.CT_PUMP_A.values,
                df.OA_Temp.values,df.SA_Temp.values,df.kW_System.values):
    
    if iC1 > c1Thresh:
        CT_C1_Status[x]=1
    
    if iC2 > c2Thresh:
        CT_C2_Status[x]=1
        
    if iF1 > f1Thresh:
        CT_F1_Status[x]=1
        
    if iF2 > f2Thresh:
        CT_F2_Status[x]=1
        
    if iPump > pumpThresh:
        PUMP_Status[x]=1 
    
    if x <= heat_band:
        xmin=0
        xmax=x+heat_band
    elif len(df)-x <= heat_band:
        xmin=x-heat_band
        xmax=len(df)       
    else:
        xmin=x-heat_band
        xmax=x+heat_band
    
    h_peak=np.max(df.SA_Temp[xmin:xmax]-df.OA_Temp[xmin:xmax])
    
    #dOASA=df.SA_Temp[x]-df.OA_Temp[x]
    
    if h_peak > h1Thresh:
        Heat1_Status[x]=1 
        
    #if h_peak > h2Thresh and dOASA > 40:
    #    Heat2_Status[x]=1 
    
    
df['CT_C1_Status']=CT_C1_Status
df['CT_C2_Status']=CT_C2_Status
df['CT_F1_Status']=CT_F1_Status
df['CT_F2_Status']=CT_F2_Status
df['PUMP_Status']=PUMP_Status
df['Heat1_Status']=Heat1_Status
#df['Heat2_Status']=Heat2_Status
    


#==============================================================================
# kW_system=df.kW_System
# currentStage=[0]*len(kW_system)
# for x,i in enumerate(kW_system.values):
#     if x > 1:
#         
#         if i>=stg1Thresh[0] and i<=stg1Thresh[1]:
#             #print x,i
#             if kW_system[x]-kW_system[x-1] > 0.7 :
# 
#                 currentStage[x]=2
#             else:
#                 currentStage[x]=1
#         if i > stg2Thresh:
#             currentStage[x]=2
#     
# df['currentStage']=currentStage
#==============================================================================

#defining component boolean states
c1ON=(df.CT_C1_Status==1) | (df.CT_C1_A>=c1Thresh)
c2ON=(df.CT_C2_Status==1) | (df.CT_C2_A>=c2Thresh)
f1ON=(df.CT_F1_Status==1) | (df.CT_F1_A>=f1Thresh)
f2ON=(df.CT_F2_Status==1) | (df.CT_F2_A>=f2Thresh)
pumpON=(df.PUMP_Status==1) | (df.CT_PUMP_A>=pumpThresh)
h1ON=(df.Heat1_Status==1) | (df.SA_Temp>=h1Thresh+70)
#h2ON=(df.Heat2_Status==1) | (df.SA_Temp>=h2Thresh+60)

c1OFF=~ c1ON
c2OFF=~ c2ON
f1OFF=~ f1ON
f2OFF=~ f2ON
pumpOFF=~ pumpON
h1OFF=~ h1ON
        
#%% Mode tagging
if setThresholds:
    for component in ['CT_C1_A','CT_C2_A','CT_F1_A','CT_F2_A','CT_PUMP_A',
                      'AOrtu_fan1_Volts']:
        plt.figure()    
        df[component].hist(bins=100)
        plt.title(component)
  
df['opMode']=['Transient']*len(df)
df['opModeStatus']=['Transient']*len(df)
      
#munters thresholds
f1Thresh=3.5
f2Thresh=1 
c1Thresh=4.5
c2Thresh=4.5 
pumpThresh=2 
h1Thresh=35   #signifies tagging if within heat band of a SA 35+ deg above OA
#h2Thresh=45   #signifies tagging if within heat band of a SA 45+ deg above OA,
# decided to combine the two heating due to large amount of trasiency
heat_band=5#the minute width of the heat band(checks values +- from current pt)

## Correcting data based off of inspection

CT_C1_Status=[0]*len(df)
CT_C2_Status=[0]*len(df)
CT_F1_Status=[0]*len(df)
CT_F2_Status=[0]*len(df)
PUMP_Status=[0]*len(df)
Heat1_Status=[0]*len(df)
#Heat2_Status=[0]*len(df)

for x,iC1,iC2,iF1,iF2,iPump,iOA,iSA,iPow in itertools.izip(
                range(len(df)),df.CT_C1_A.values,df.CT_C2_A.values,
                df.CT_F1_A.values,df.CT_F2_A.values,df.CT_PUMP_A.values,
                df.OA_Temp.values,df.SA_Temp.values,df.kW_System.values):
    
    if iC1 > c1Thresh:
        CT_C1_Status[x]=1
    
    if iC2 > c2Thresh:
        CT_C2_Status[x]=1
        
    if iF1 > f1Thresh:
        CT_F1_Status[x]=1
        
    if iF2 > f2Thresh:
        CT_F2_Status[x]=1
        
    if iPump > pumpThresh:
        PUMP_Status[x]=1 
    
    if x <= heat_band:
        xmin=0
        xmax=x+heat_band
    elif len(df)-x <= heat_band:
        xmin=x-heat_band
        xmax=len(df)       
    else:
        xmin=x-heat_band
        xmax=x+heat_band
    
    h_peak=np.max(df.SA_Temp[xmin:xmax]-df.OA_Temp[xmin:xmax])
    
    #dOASA=df.SA_Temp[x]-df.OA_Temp[x]
    
    if h_peak > h1Thresh:
        Heat1_Status[x]=1 
        
    #if h_peak > h2Thresh and dOASA > 40:
    #    Heat2_Status[x]=1 
    
    
df['CT_C1_Status']=CT_C1_Status
df['CT_C2_Status']=CT_C2_Status
df['CT_F1_Status']=CT_F1_Status
df['CT_F2_Status']=CT_F2_Status
df['PUMP_Status']=PUMP_Status
df['Heat1_Status']=Heat1_Status
#df['Heat2_Status']=Heat2_Status
    


#==============================================================================
# kW_system=df.kW_System
# currentStage=[0]*len(kW_system)
# for x,i in enumerate(kW_system.values):
#     if x > 1:
#         
#         if i>=stg1Thresh[0] and i<=stg1Thresh[1]:
#             #print x,i
#             if kW_system[x]-kW_system[x-1] > 0.7 :
# 
#                 currentStage[x]=2
#             else:
#                 currentStage[x]=1
#         if i > stg2Thresh:
#             currentStage[x]=2
#     
# df['currentStage']=currentStage
#==============================================================================

#defining component boolean states
c1ON=(df.CT_C1_Status==1) | (df.CT_C1_A>=c1Thresh)
c2ON=(df.CT_C2_Status==1) | (df.CT_C2_A>=c2Thresh)
f1ON=(df.CT_F1_Status==1) | (df.CT_F1_A>=f1Thresh)
f2ON=(df.CT_F2_Status==1) | (df.CT_F2_A>=f2Thresh)
pumpON=(df.PUMP_Status==1) | (df.CT_PUMP_A>=pumpThresh)
h1ON=(df.Heat1_Status==1) | (df.SA_Temp>=h1Thresh+70)
#h2ON=(df.Heat2_Status==1) | (df.SA_Temp>=h2Thresh+60)

c1OFF=~ c1ON
c2OFF=~ c2ON
f1OFF=~ f1ON
f2OFF=~ f2ON
pumpOFF=~ pumpON
h1OFF=~ h1ON
#h2OFF=~ h2ON

#combining f2 and pump tags to an iec tag
iecON= f2ON | pumpON
iecOFF=~ iecON

#defining modes by component boolean states
#General/Cooling modes

OFF=            f1OFF
VENT=           c1OFF & c2OFF & f1ON  & iecOFF & h1OFF #& h2OFF
IEC__ONLY=      c1OFF & c2OFF & f1ON  & iecON  & h1OFF #& h2OFF
IEC__DX1=       c1ON  & c2OFF & f1ON  & iecON  & h1OFF #& h2OFF
IEC__DX2=       c1ON  & c2ON  & f1ON  & iecON  & h1OFF #& h2OFF

#Heating Modes
HEAT1=          c1OFF & c2OFF & f1ON  & iecOFF & h1ON  #& h2OFF
#HEAT2=          c1OFF & c2OFF & f1ON  & iecOFF & h1ON  & h2ON

#Cooling plus Heat1
IEC__HEAT1=     c1OFF & c2OFF & f1ON  & iecON  & h1ON  #& h2OFF
IEC__DX1__HEAT1=c1ON  & c2OFF & f1ON  & iecON  & h1ON  #& h2OFF
IEC__DX2__HEAT1=c1ON  & c2ON  & f1ON  & iecON  & h1ON  #& h2OFF

#Cooling plus Heat2
#IEC__HEAT2=     c1OFF & c2OFF & f1ON  & iecON  & h1ON  & h2ON
#IEC__DX1__HEAT2=c1ON  & c2OFF & f1ON  & iecON  & h1ON  & h2ON
#IEC__DX2__HEAT2=c1ON  & c2ON  & f1ON  & iecON  & h1ON  & h2ON

    
#Labling modes in dataframe

allModes=[OFF,VENT,IEC__ONLY,IEC__DX1,IEC__DX2,HEAT1,IEC__HEAT1,
          IEC__DX1__HEAT1,IEC__DX2__HEAT1
          ]

allModeNames=['OFF','VENT','IEC ONLY','IEC and DX1','IEC and DX2','HEAT1',
              'IEC and HEAT1','IEC and DX1 and HEAT1','IEC and DX2 and HEAT1'
              ]
transTime=14
for mode,modeName in itertools.izip(allModes,allModeNames):
    df.loc[(mode.index[mode.values==True]),('opMode')]=modeName

for i in xrange(len(df)):
    if i>=transTime and i<=len(df):
        currentMode=df.opMode[i]               
        steadyState=[currentMode]*transTime==[x for x 
                                              in df.opMode[range(i-transTime,i,1)]]
        
        if steadyState:
            df.loc[df.index[i],('opModeStatus')]='SS'
#transTime=14
#cycleWindow=2*transTime
#for mode,modeName in itertools.izip(allModes,allModeNames):
#    for i in df[mode].index:
#        df['opmode',i]=modeName#df.opMode[i]=modeName


# tagging opperating states based on modes

#for i in xrange(len(df)):
#    if i>=transTime and i<=len(df)-cycleWindow:
#        previousMode=df.opMode[i-1]
#        currentMode=df.opMode[i]
#       
#        #if (previousMode.__contains__('DC') and 
#        #    previousMode.__contains__(currentMode) and not 
#        #    currentMode.__contains__('DC')):
#        #    if max(df.CT_PUMP_A[i : i+cycleWindow :])>=pumpThresh:
#        #        df.opMode[i]=previousMode
#        #        currentMode=df.opMode[i]
#                       
#        steadyState=[currentMode]*transTime==[x for x 
#                                              in df.opMode[i-transTime:i:]]
#        
#        if steadyState:
#            df.opModeStatus[i]='SS'
#        else:
#            continue

#%% priority raw data graphing
dstart =4
dnum = 3
tstar = 1440 * dstart
tend = tstar + 1440 * dnum
RAoffset = 60
fig=plt.figure()
fig.set_size_inches(wFig,hFig)
ax=fig.add_subplot(111)
plt.plot(pardf.index[tstar:tend],pardf.kW_AC30[tstar:tend],'r-')
plt.plot(pardf.index[tstar:tend],pardf.kW_AC36[tstar:tend],'g-')
plt.plot(pardf.index[tstar:tend],pardf.kW_AC31[tstar:tend],'b-')
plt.plot(df.index[tstar:tend],df.kW_System[tstar:tend],'m-')
plt.plot(pardf.index[tstar:tend],(pardf.pRA_Temp[tstar:tend]-
         ([RAoffset]*len(pardf.index[tstar:tend]))),'y-')
plt.plot(df.index[tstar:tend],(df.RA_Temp[tstar:tend]-
         ([RAoffset]*len(df.index[tstar:tend]))),'c-')
plt.plot(df.index[tstar:tend],(df.PUMP_Status[tstar:tend]*5),'k-')
plt.plot(df.index[tstar:tend],(df.CT_C1_Status[tstar:tend]*8),'b-')
plt.plot(df.index[tstar:tend],(df.CT_C2_Status[tstar:tend]*12),'r-')
plt.show()
#plt.ylabel('kW or RA-60 deg F')
#plt.xlabel('date time from '+str(obsMonth)+' '+str(np.floor(tstar/1440))+
#           ' thru '+str(np.floor(tend/1440)-1)+' '+ str(obsYear))

#%% Tracer gas mapping of airflow
rho_SCF=ps.psych(14.696,'Tdb',70,'RH',0,'MAD') 

df['mDotOA']=[(-42.024*x**2+1158.3*x-1012.8)*min2hour*rho_SCF 
              for x in df.dPsa_fan_inH2O]

df['mDotSA']=[x for x in df.mDotOA]

psCalc={}
for data in ['OA_Temp','OA_RH','SA_Temp','SA_RH','RA_Temp','RA_RH',
             'mDotOA','mDotSA']:
    psCalc[data]=df[data].values

for calc in ['vDotSA','vDotOA','wOA','wSA','wRA','OAwb','OAwbd','RAdp','OAdp','mCPmin',
             'OSAF','senSystemCap','latSystemCap','totSystemCap',
             'senSystemCOP','latSystemCOP','totSystemCOP','senRoomCap',
             'latRoomCap','totRoomCap','senRoomCOP','latRoomCOP',
             'totRoomCOP']:
    psCalc[calc]=[np.nan]*len(df)
t=time.time()    
#%% Air property calculations
for i in xrange(len(df)):
    # Absolute humidity ratio
    psCalc['wOA'][i]=ps.psych(atmPressure,'Tdb',psCalc['OA_Temp'][i],'RH',
                              psCalc['OA_RH'][i]/100,'W')
    psCalc['wSA'][i]=ps.psych(atmPressure,'Tdb',psCalc['SA_Temp'][i],'RH',
                              psCalc['SA_RH'][i]/100,'W') 
    psCalc['wRA'][i]=ps.psych(atmPressure,'Tdb',psCalc['RA_Temp'][i],'RH',
                              psCalc['RA_RH'][i]/100,'W')
    psCalc['OAwb'][i]=ps.psych(atmPressure,'Tdb',psCalc['OA_Temp'][i],'RH',
                               psCalc['OA_RH'][i]/100,'Twb')
    psCalc['OAwbd'][i]=psCalc['OA_Temp'][i]-psCalc['OAwb'][i]
    
    psCalc['RAdp'][i]=ps.psych(atmPressure,'Tdb',psCalc['RA_Temp'][i],'RH',
                              psCalc['RA_RH'][i]/100,'DP')
    psCalc['OAdp'][i]=ps.psych(atmPressure,'Tdb',psCalc['OA_Temp'][i],'RH',
                              psCalc['OA_RH'][i]/100,'DP')

    # mass flow rate of air
    rho_air=ps.psych(atmPressure,'Tdb',df.OA_Temp[i],'RH',df.OA_RH[i]/100,'MAD')
    psCalc['vDotSA'][i]=psCalc['mDotSA'][i]/rho_air/min2hour #CFM
    psCalc['vDotOA'][i]=psCalc['mDotOA'][i]/rho_air/min2hour # CFM

    # This is now correct,the effectiveness of the heat exchanger is a function 
    # of CPmin- whichever is smaller between mDotWater*CPwater and mDotAir*CPair 
    psCalc['mCPmin'][i]=psCalc['mDotOA'][i]*0.24 if psCalc['mDotOA'][i]*0.24 <\
                                                   mDotWater*1 else mDotWater*1


for calc in ['vDotSA','vDotOA','wOA','wSA','wRA','OAwb','OAwbd','RAdp','OAdp',
             'senSystemCap','latSystemCap','totSystemCap',
             'senSystemCOP','latSystemCOP','totSystemCOP','senRoomCap',
             'latRoomCap','totRoomCap','senRoomCOP','latRoomCOP','totRoomCOP',
             'mCPmin']:
    df[calc]=psCalc[calc]
    
#%% Calculate real time Capacity and COP values
print time.time()-t  

  
# System cooling metrics
# capacity (kBTU/hr)
df['senSystemCap']=df.mDotSA*(0.24/1000)*(df.OA_Temp-df.SA_Temp)
df['latSystemCap']=df.mDotSA*(971.4/1000)*(df.wOA-df.wSA)
df['totSystemCap']=df.senSystemCap+df.latSystemCap
# COP
df['senSystemCOP']=df.senSystemCap/3.412/df.kW_System
df['latSystemCOP']=df.latSystemCap/3.412/df.kW_System
df['totSystemCOP']=df.totSystemCap/3.412/df.kW_System

# Room cooling metrics
# capacity (kBTU/hr)
df['senRoomCap']=df.mDotSA*(0.24/1000)*(df.RA_Temp-df.SA_Temp)
df['latRoomCap']=df.mDotSA*(971.4/1000)*(df.wRA-df.wSA)
df['totRoomCap']=df.senRoomCap+df.latRoomCap
# COP
df['senRoomCOP']=df.senRoomCap/3.412/df.kW_System
df['latRoomCOP']=df.latRoomCap/3.412/df.kW_System
df['totRoomCOP']=df.totRoomCap/3.412/df.kW_System

if max(df.mDotSA)*0.24 < mDotWater:
    mCPmin=df.mDotSA*0.24
else:
    mCPmin=mDotWater*1
    
df=df[(df.SA_Temp > 0)]

#%% build water data from water regression file

# water use in gal /min vs OA temp regression


#==============================================================================
# build the linear regression 
#==============================================================================
exog = WRdf[(WRdf.kW_System>4) & (WRdf.Vwater_Gal>=0.001)].OA_Temp 
exog = sm.add_constant(exog)
endog= WRdf[(WRdf.kW_System>4) & (WRdf.Vwater_Gal>=0.001)].Vwater_Gal
WRres = sm.WLS(endog, exog).fit()
WRrsq=WRres.rsquared
#WRrsq
#(WRdf.Vwater_Gal<1) & (WRdf.Vwater_Gal<1) & 
#==============================================================================
# output regression results
#==============================================================================
#print 'params:' 
#print 'WLS: ', WRres.params
#print 'r^2: ', WRres.rsquared
#print '\nslopes:'  
#print 'WLS: ', WRres.params[1:].cumsum()
#
print WRres.summary()

#==============================================================================
# Recreate water data for given month
#==============================================================================
pexog=(df.OA_Temp*df.CT_F2_Status)
pexog = sm.add_constant(pexog)
df['pVwater_GPM']=df.CT_F2_Status*(np.dot(pexog,WRres.params))



#%% daily data trends
dayData={}
monthstats=cd.monthrange(obsYear,obsMonth)
lastday=monthstats[1]
monDays=range(1,lastday+1,1)

dayData['OAtempMin']=[df[(df.index.day==dy)].OA_Temp.min() for dy 
                                                              in monDays]
dayData['OAtempMean']=[df[(df.index.day==dy)].OA_Temp.mean() for dy 
                                                              in monDays]
dayData['OAtempMax']=[df[(df.index.day==dy)].OA_Temp.max() for dy 
                                                              in monDays]
dayData['wGal']=[df[(df.index.day==dy)].pVwater_GPM.sum() for dy 
                                                              in monDays]
dayDF=pd.DataFrame(dayData)                                                              

#%% hour data trends
hourData={}   
hourData['OAtempMin']=[df[(df.index.hour==hr)].OA_Temp.min() for hr 
                                                              in range(24)]
hourData['OAtempMean']=[df[(df.index.hour==hr)].OA_Temp.mean() for hr 
                                                              in range(24)]
hourData['OAtempMax']= [df[(df.index.hour==hr)].OA_Temp.max() for hr 
                                                              in range(24)]
hourData['OAwbdMean']= [df[(df.index.hour==hr)].OAwbd.mean() for hr 
                                                              in range(24)]
hourData['wGal']=[df[(df.index.hour==hr)].Vwater_Gal.sum() for hr 
                                                              in range(24)]                                                              
hourData['RAtempMin']=[df[(df.index.hour==hr)].RA_Temp.min() for hr 
                                                              in range(24)]
hourData['RAtempMean']=[df[(df.index.hour==hr)].RA_Temp.mean() for hr 
                                                              in range(24)]
hourData['RAtempMax']= [df[(df.index.hour==hr)].RA_Temp.max() for hr 
                                                              in range(24)]
hourData['mDotSA']=[df[(df.index.hour==hr)].mDotSA.mean()/min2hour/rho_SCF for 
                                                           hr in range(24)]
hourData['mDotOA']=[df[(df.index.hour==hr)].mDotOA.mean()/
                df[(df.index.hour==hr)].mDotSA.mean()  for hr in range(24)]

for key,data in df.groupby('opMode'):
    key[0:]
    if (key[0:]=='VENT'):
        hourData['senSystem_'+key+'_cool']=[data[(data.index.hour==hr) & 
                         (data['senSystemCap']>=0)].senSystemCap.sum()*(1./60) 
                         for hr in range(24)]                             
        hourData['latSystem_'+key+'_cool']=[data[(data.index.hour==hr) & 
                         (data['latSystemCap']>=0)].latSystemCap.sum()*(1./60) 
                         for hr in range(24)]                             
        hourData['totSystem_'+key+'_cool']=[data[(data.index.hour==hr) & 
                         (data['totSystemCap']>=0)].totSystemCap.sum()*(1./60) 
                         for hr in range(24)]                             
        hourData['senRoom_'+key+'_cool']  =[data[(data.index.hour==hr) & 
                             (data['senRoomCap']>=0)].senRoomCap.sum()*(1./60) 
                             for hr in range(24)]
        hourData['latRoom_'+key+'_cool']  =[data[(data.index.hour==hr) & 
                             (data['latRoomCap']>=0)].latRoomCap.sum()*(1./60) 
                             for hr in range(24)]
        hourData['totRoom_'+key+'_cool']  =[data[(data.index.hour==hr) & 
                             (data['totRoomCap']>=0)].totRoomCap.sum()*(1./60) 
                             for hr in range(24)]
        hourData['senSystem_'+key+'_heat']=[data[(data.index.hour==hr) & 
                          (data['senSystemCap']<0)].senSystemCap.sum()*(1./60) 
                          for hr in range(24)]
        hourData['latSystem_'+key+'_heat']=[data[(data.index.hour==hr) & 
                          (data['latSystemCap']<0)].latSystemCap.sum()*(1./60) 
                          for hr in range(24)]
        hourData['totSystem_'+key+'_heat']=[data[(data.index.hour==hr) & 
                          (data['totSystemCap']<0)].totSystemCap.sum()*(1./60) 
                          for hr in range(24)]
        hourData['senRoom_'+key+'_heat']  =[data[(data.index.hour==hr) & 
                              (data['senRoomCap']<0)].senRoomCap.sum()*(1./60) 
                              for hr in range(24)]
        hourData['latRoom_'+key+'_heat']  =[data[(data.index.hour==hr) & 
                              (data['latRoomCap']<0)].latRoomCap.sum()*(1./60) 
                              for hr in range(24)]
        hourData['totRoom_'+key+'_heat']  =[data[(data.index.hour==hr) & 
                              (data['totRoomCap']<0)].totRoomCap.sum()*(1./60) 
                              for hr in range(24)]
    else:
        hourData['senSystem_'+key] = [data[(data.index.hour==hr)].senSystemCap.\
                                      sum()*(1./60) for hr in range(24)]
        hourData['latSystem_'+key] = [data[(data.index.hour==hr)].latSystemCap.\
                                      sum()*(1./60) for hr in range(24)]
        hourData['totSystem_'+key] = [data[(data.index.hour==hr)].totSystemCap.\
                                      sum()*(1./60) for hr in range(24)]
        hourData['senRoom_'+key] = [data[(data.index.hour==hr)].senRoomCap.\
                                      sum()*(1./60) for hr in range(24)]
        hourData['latRoom_'+key] = [data[(data.index.hour==hr)].latRoomCap.\
                                      sum()*(1./60) for hr in range(24)]
        hourData['totRoom_'+key] = [data[(data.index.hour==hr)].totRoomCap.\
                                      sum()*(1./60) for hr in range(24)]


hourData['Transient']=    [len(df[(df.index.hour==hr) & 
                     (df['opMode']=='Transient')])*(1./60) for hr in range(24)]
hourData['OFF']=          [len(df[(df.index.hour==hr) & 
                           (df['opMode']=='OFF')])*(1./60) for hr in range(24)]
hourData['VENT']=         [len(df[(df.index.hour==hr) & 
                          (df['opMode']=='VENT')])*(1./60) for hr in range(24)]
hourData['IEC ONLY']=     [len(df[(df.index.hour==hr) & 
                      (df['opMode']=='IEC ONLY')])*(1./60) for hr in range(24)]
hourData['IEC and DX1']=  [len(df[(df.index.hour==hr) & 
                   (df['opMode']=='IEC and DX1')])*(1./60) for hr in range(24)]
hourData['IEC and DX2']=  [len(df[(df.index.hour==hr) & 
                   (df['opMode']=='IEC and DX2')])*(1./60) for hr in range(24)]
hourData['HEAT1']=        [len(df[(df.index.hour==hr) & 
                         (df['opMode']=='HEAT1')])*(1./60) for hr in range(24)]
hourData['IEC and HEAT1']=[len(df[(df.index.hour==hr) & 
                 (df['opMode']=='IEC and HEAT1')])*(1./60) for hr in range(24)]
hourData['IEC and DX1 and HEAT1']=[len(df[(df.index.hour==hr) & 
         (df['opMode']=='IEC and DX1 and HEAT1')])*(1./60) for hr in range(24)]
hourData['IEC and DX2 and HEAT1']=[len(df[(df.index.hour==hr) & 
         (df['opMode']=='IEC and DX2 and HEAT1')])*(1./60) for hr in range(24)]

hourDF=pd.DataFrame(hourData)




#%% OA temp data trends for plotting data by temp bins

tempData={} 
trmin=50
trmax=100
trbin=5

    
tempData['OFF']=        [len(df[((df['OA_Temp']>tr) & (df['OA_Temp']<
                         (tr+trbin))) & (df['opMode']=='OFF')])*(1./60) 
                         for tr in range(trmin,trmax,trbin)]
tempData['VENT']=       [len(df[((df['OA_Temp']>tr) & (df['OA_Temp']<
                         (tr+trbin))) & (df['opMode']=='VENT')])*(1./60) 
                         for tr in range(trmin,trmax,trbin)]
tempData['IEC ONLY']=   [len(df[((df['OA_Temp']>tr) & (df['OA_Temp']<
                         (tr+trbin))) & (df['opMode']=='IEC ONLY')])*(1./60) 
                         for tr in range(trmin,trmax,trbin)]
tempData['IEC and DX1']=[len(df[((df['OA_Temp']>tr) & (df['OA_Temp']<
                         (tr+trbin))) & (df['opMode']=='IEC and DX1')])*(1./60) 
                         for tr in range(trmin,trmax,trbin)]
tempData['IEC and DX2']=[len(df[((df['OA_Temp']>tr) & (df['OA_Temp']<
                         (tr+trbin))) & (df['opMode']=='IEC and DX2')])*(1./60) 
                         for tr in range(trmin,trmax,trbin)]
tempData['HEAT1']=      [len(df[((df['OA_Temp']>tr) & (df['OA_Temp']<
                         (tr+trbin))) & (df['opMode']=='HEAT1')])*(1./60) 
                         for tr in range(trmin,trmax,trbin)]
tempData['IEC and HEAT1']=[len(df[((df['OA_Temp']>tr) & (df['OA_Temp']<
             (tr+trbin))) & (df['opMode']=='IEC and HEAT1')])*(1./60) 
             for tr in range(trmin,trmax,trbin)]
tempData['Transient']=  [len(df[((df['OA_Temp']>tr) & (df['OA_Temp']<
             (tr+trbin))) & (df['opMode']=='Transient')])*(1./60) 
             for tr in range(trmin,trmax,trbin)]
tempData['IEC and DX1 and HEAT1']=[len(df[((df['OA_Temp']>tr) & (df['OA_Temp']<
             (tr+trbin))) & (df['opMode']=='IEC and DX1 and HEAT1')])*(1./60) 
             for tr in range(trmin,trmax,trbin)]
tempData['IEC and DX2 and HEAT1']=[len(df[((df['OA_Temp']>tr) & (df['OA_Temp']<
             (tr+trbin))) & (df['opMode']=='IEC and DX2 and HEAT1')])*(1./60) 
             for tr in range(trmin,trmax,trbin)]

tempDF=pd.DataFrame(tempData)
tempDF.index=range(trmin,trmax,trbin)

#%% RA temp data trends for plotting data by temp bins

RAtempData={} 
RAmin=65
RAmax=78
RAbin=1


RAtempData['OFF']=     [len(df[((df['RA_Temp']>tr) & (df['RA_Temp']<
                        (tr+RAbin))) & (df['opMode']=='OFF')])*(1./60) 
                        for tr in range(RAmin,RAmax,RAbin)]
RAtempData['VENT']=    [len(df[((df['RA_Temp']>tr) & (df['RA_Temp']<
                        (tr+RAbin))) & (df['opMode']=='VENT')])*(1./60) 
                        for tr in range(RAmin,RAmax,RAbin)]
RAtempData['IEC ONLY']=[len(df[((df['RA_Temp']>tr) & (df['RA_Temp']<
                        (tr+RAbin))) & (df['opMode']=='IEC ONLY')])*(1./60) 
                        for tr in range(RAmin,RAmax,RAbin)]
RAtempData['IEC and DX1']=[len(df[((df['RA_Temp']>tr) & (df['RA_Temp']<
                        (tr+RAbin))) & (df['opMode']=='IEC and DX1')])*(1./60) 
                        for tr in range(RAmin,RAmax,RAbin)]
RAtempData['IEC and DX2']=[len(df[((df['RA_Temp']>tr) & (df['RA_Temp']<
                        (tr+RAbin))) & (df['opMode']=='IEC and DX2')])*(1./60) 
                        for tr in range(RAmin,RAmax,RAbin)]
RAtempData['HEAT1']=   [len(df[((df['RA_Temp']>tr) & (df['RA_Temp']<
                        (tr+RAbin))) & (df['opMode']=='HEAT1')])*(1./60) 
                        for tr in range(RAmin,RAmax,RAbin)]
RAtempData['IEC and HEAT1']=[len(df[((df['RA_Temp']>tr) & (df['RA_Temp']<
                        (tr+RAbin))) & (df['opMode']=='IEC and HEAT1')])*(1./60) 
                        for tr in range(RAmin,RAmax,RAbin)]
RAtempData['Transient']=[len(df[((df['RA_Temp']>tr) & (df['RA_Temp']<
                        (tr+RAbin))) & (df['opMode']=='Transient')])*(1./60) 
                        for tr in range(RAmin,RAmax,RAbin)]
RAtempData['IEC and DX1 and HEAT1']=[len(df[((df['RA_Temp']>tr) & (df['RA_Temp']<
                 (tr+RAbin))) & (df['opMode']=='IEC and DX1 and HEAT1')])*(1./60) 
                 for tr in range(RAmin,RAmax,RAbin)]
RAtempData['IEC and DX2 and HEAT1']=[len(df[((df['RA_Temp']>tr) & (df['RA_Temp']<
                 (tr+RAbin))) & (df['opMode']=='IEC and DX2 and HEAT1')])*(1./60) 
                 for tr in range(RAmin,RAmax,RAbin)]

RAtempDF=pd.DataFrame(RAtempData)
RAtempDF.index=range(RAmin,RAmax,RAbin)

#%% set up color matrix for mode based plotting and text/ figure size
wFig=8
hFig=6
tl_size=15
#colors
mode_colors=['#9df5a8', '#F5F49D', '#07913C', '#00b5fc', '#0061fc',
             '#F59DE8', '#F59DA9', '#8C1525',
             '#FF861C', '#555555', '#BBBBBB']
#possible modes for plotting           
mode_type=['VENT', 'VENT_heat', 'IEC ONLY', 'IEC and DX1', 'IEC and DX2',
           'IEC and HEAT1', 'IEC and DX1 and HEAT1','IEC and DX2 and HEAT1',
           'HEAT1',  'OFF', 'Transient']     
#find all opmodes           
im=0
modesUS=[None for pm in range(len(df.groupby(['opMode'])))] 

for key in df.groupby(['opMode']):
    modesUS[im]=key[0]
    im=im+1
modesUS
#sort opmodes into plotting order

if 'VENT' in modesUS:
    modes = [None for pm in range(len(modesUS)+1)]
    modesC= [None for pm in range(len(modesUS))]
else:
    modes=[None for pm in range(len(modesUS))]
    modesC= [None for pm in range(len(modesUS))]
im=0
ic=0
for pmt in mode_type:
    for i in range(len(modesUS)):
        if modesUS[i]==pmt:
            if pmt=='VENT':
                modes[im]=modesUS[i]
                im=im+1
                modes[im]=(modesUS[i]+'_heat')
                im=im+1
                modesC[ic]=modesUS[i]
                ic=ic+1
            else:
                modes[im]=modesUS[i]
                im=im+1
                modesC[ic]=modesUS[i]
                ic=ic+1
modes
modesC

#%% exploratory plots for power
colorMap=0
fig=plt.figure()
ax=fig.add_subplot(111)
plt.plot(pg1mdf.pgOA_Temp,pg1mdf.pgkW_Site,'.',c='#800080',
             alpha=alFA)
plt.plot(p1mdf.pOA_Temp,p1mdf.kW_Main,'.',c='#008080',
             alpha=alFA)
#plt.ylabel('kW for WF Site')
#plt.xlabel('Outdoor Air Temp ($^\circ$F)')
#ax.set_ylim(50,120)
#ax.set_xlim(60,90)
#plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
#           borderaxespad=0,prop={'size':9},numpoints=1)


#if outputGraphs: plt.savefig(outputPath+'OA Temp vs SA Temp.png',
#                             bbox_to_inches='tight')

#plotname='RA Temp vs SA Temp inc. off and trasient'
#
#if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
#                             str(obsMonth)+'_'+plotname+'.png',
#                             bbox_to_inches='tight')

#%% Month OA temperature profile
fig=plt.figure()
fig.set_size_inches(wFig,hFig)
ax=fig.add_subplot(111)
plt.plot((hourDF.index+1),hourDF.OAtempMean,'r-',label='Mean OA')
plt.plot((hourDF.index+1),hourDF.OAtempMin,'r--',
         (hourDF.index+1),hourDF.OAtempMax,'r--',label='Min/Max OA')
#plt.xlabel('Hour of Day')
#plt.ylabel('Temperature ($^\circ$F)')
#plt.legend()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)

# this ouput file can be used for all hourDF based graphs
plotname='OA month profile'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')
                             
if createCSV: hourDF.to_csv(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                        str(obsMonth)+'_hourDF.csv',na_rep='=NA()')
                        
# use hourDF output file to create XL plots                        
                        
#%% Month RA temperature profile
fig=plt.figure()
fig.set_size_inches(wFig,hFig)
ax=fig.add_subplot(111)
plt.plot((hourDF.index+1),hourDF.RAtempMin,'r--',
         (hourDF.index+1),hourDF.RAtempMax,'r--',label='Min/Max RA')
plt.plot((hourDF.index+1),hourDF.RAtempMean,'r-',label='Mean RA')
#plt.xlabel('Hour of Day')
#plt.ylabel('Temperature ($^\circ$F)')
#plt.legend()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)

plotname='RA month profile'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use hourDF output file to create XL plots
                             
#%% Mode runtime by OA temp bins

SBG = StackedBarGrapher()

md_colors=['#000000' for mc in range(len(modesC))]           
plotmodes=[None for pm in range(len(modesC))]
           
for mode in modesC:
    plotmodes[modesC.index(mode)]=mode
    md_colors[modesC.index(mode)]=mode_colors[mode_type.index(mode)]
 
tmpArray=[tempData[tmpmode] for tmpmode in plotmodes]
tmpArrayT = np.copy(tmpArray).transpose()           
md_labels=plotmodes
#tl_size=15
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax = fig.add_subplot(111)
SBG.stackedBarPlot(ax,
                   tmpArrayT,
                   md_colors,
                   md_labels,
                   xLabels=range(trmin,trmax,trbin),
                   yTicks=9,
                   gap=.2,
                   lblsize=tl_size,
                   scale=False,
                   endGaps=True
                   )

#                   xlabel='Outside Air Temps',
#                   ylabel='Total Runtime Hours',                   
plt.yticks(range(0,201,20))
ax.set_ylim(0,200)                   
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
#plt.title('Runtime Hours by Mode')
#handle,label=ax1.get_legend_handles_labels()
#plt.legend(handle,label,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
#           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1,fancybox=True)

# this ouput file can be used for all tempDF based graphs
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)

#%%
plotname='Mode Runtime Hours temp bins'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')
                             
if createCSV: tempDF.to_csv(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                        str(obsMonth)+'_OAtempDF.csv',na_rep='=NA()')
                        
if createCSV: RAtempDF.to_csv(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                        str(obsMonth)+'_RAtempDF.csv',na_rep='=NA()')

# use OAtempDF output file to create XL plots

#%% Mode runtime by RA temp bins

SBG = StackedBarGrapher()

md_colors=['#000000' for mc in range(len(modesC))]           
plotmodes=[None for pm in range(len(modesC))]           
for mode in modesC:
    plotmodes[modesC.index(mode)]=mode
    md_colors[modesC.index(mode)]=mode_colors[mode_type.index(mode)]
 
tmpArray=[RAtempData[tmpmode] for tmpmode in plotmodes]
tmpArrayT = np.copy(tmpArray).transpose()           
md_labels=plotmodes
#tl_size=15
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax = fig.add_subplot(111)
SBG.stackedBarPlot(ax,
                   tmpArrayT,
                   md_colors,
                   md_labels,
                   xLabels=range(RAmin,RAmax,RAbin),
                   yTicks=9,
                   gap=.2,
                   lblsize=tl_size,
                   scale=False,
                   endGaps=True
                   )

#                   xlabel='Room Air Temps',
#                   ylabel='Total Runtime Hours',                   
plt.yticks(range(0,201,20))
ax.set_ylim(0,200)                    
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
#plt.title('Runtime Hours by Mode')
#handle,label=ax1.get_legend_handles_labels()
#plt.legend(handle,label,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
#           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1,fancybox=True)

# this ouput file can be used for all tempDF based graphs
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)

plotname='Mode Runtime Hours RA temp bins'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')
                             
#if createCSV: tempDF.to_csv(outputPath+str(obsYear)+'_'+str(placeHolder)+\
#                        str(obsMonth)+'_OAtempDF.csv',na_rep='=NA()')
#                        
#if createCSV: RAtempDF.to_csv(outputPath+str(obsYear)+'_'+str(placeHolder)+\
#                        str(obsMonth)+'_RAtempDF.csv',na_rep='=NA()')

# use RAtempDF output file to create XL plots

#%% new mode runtime by hourbins vs OA temp stats

SBG = StackedBarGrapher()

md_colors=['#000000' for mc in range(len(modesC))]           
plotmodes=[None for pm in range(len(modesC))]           
for mode in modesC:
    plotmodes[modesC.index(mode)]=mode
    md_colors[modesC.index(mode)]=mode_colors[mode_type.index(mode)]

hrArray=[hourData[hrmode] for hrmode in plotmodes]
hrArrayT = np.copy(hrArray).transpose()           
md_labels=plotmodes
#tl_size=15
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax1 = fig.add_subplot(111)
SBG.stackedBarPlot(ax1,
                   hrArrayT,
                   md_colors,
                   md_labels,
                   xLabels=range(1,25,1),
                   yTicks=7,
                   gap=.2,
                   lblsize=tl_size,
                   scale=False,
                   endGaps=True
                   )
                   
#                   xlabel='Hour of the Day',
#                   ylabel='Total Runtime Hours',

ax1.set_ylim(0,80)
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
ax1.tick_params(axis='y', pad=0)
#july looked better with 1000 roo scaleing
#ax1.axhline(0,color='k',linestyle='-')
for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(tl_size)
ax2=ax1.twinx()
#ax2.set_ylabel('OA Temp ($^\circ$F)')
ax2.set_ylim(0,110)

plt.plot(hourDF.index,hourDF.OAtempMean,'r-',label='Mean OA')
plt.plot(hourDF.index,hourDF.OAtempMin,'r--',
         hourDF.index,hourDF.OAtempMax,'r--',label='Min/Max OA')
ax2.tick_params(axis='y', which='both', labelsize=tl_size, direction="out")
plt.yticks([50, 60, 70, 80, 90, 100])
#handle,label=ax1.get_legend_handles_labels()
#handle2,label2=ax2.get_legend_handles_labels()
#all_handles=handle+handle2
#all_labels=label+label2

#plt.legend(all_handles,all_labels,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
#           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1,fancybox=True)


plt.show()

plotname='Mode Runtime Hours with OA stats'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')
                             
# use hourDF output file to create XL plots

#%% Daily water use by day of month with OA stats

SBG = StackedBarGrapher()

w_color=['#00B7FF']           
plotval=['wGal']           
                    
hrArray=[dayData[Dval] for Dval in plotval]
hrArrayT = np.copy(hrArray).transpose()
coolarr=hrArrayT
heatarr=hrArrayT
           
w_labels=plotval
tl_size=15
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax1 = fig.add_subplot(111)
SBG.stackedBarPlot(ax1,
                   hrArrayT,
                   w_color,
                   w_labels,
                   negStack=False,
                   xLabels=monDays,
                   yTicks=7,
                   gap=.2,
                   lblsize=tl_size,
                   scale=False,
                   endGaps=True
                   )
                   
#                   xlabel='Hour of the Day',
#                   ylabel='Sensible System Cooling Capacity (total kBTU)',

hrArraySumT=[np.sum(hrArrayT[i]) for i in range(len(hrArrayT))]
ylmax=np.max(hrArraySumT)
rylmax=ylmax + 100*(ylmax%100>0) - ylmax%100
ylmin=np.min(hrArraySumT)
ax1.set_ylim(0,rylmax+400)
plt.yticks(range(0,int(rylmax+rylmax/5),int(rylmax/5)))
#july looked better with 1000 roo scaleing
#ax1.axhline(0,color='k',linestyle='-')
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
ax1.tick_params(axis='y', pad=0)
for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(tl_size)
ax2=ax1.twinx()
#ax2.set_ylabel('OA Temp ($^\circ$F)')
ax2.set_ylim(0,110)
plt.plot(dayDF.index,dayDF.OAtempMean,'r-',label='Mean OA')
plt.plot(dayDF.index,dayDF.OAtempMin,'r--',
         dayDF.index,dayDF.OAtempMax,'r--',label='Min/Max OA')
ax2.tick_params(axis='y', which='both', labelsize=tl_size, direction="out", pad=0)
plt.yticks([40, 50, 60, 70, 80, 90, 100])
#handle,label=ax1.get_legend_handles_labels()
#handle2,label2=ax2.get_legend_handles_labels()
#all_handles=handle+handle2
#all_labels=label+label2

#plt.legend(all_handles,all_labels,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
#           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1)

plt.show()

plotname='Daily Water use with OA stats'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use dayDF output file to create XL plots

#%% 

#==============================================================================
# Mode Stacked Cumulative Sensible System Cooling (kBTU) by hour of day with OA stats
#==============================================================================


SBG = StackedBarGrapher()


md_colors=['#000000' for mc in range(len(modes))]           
plotmodes=[None for pm in range(len(modes))]           
for mode in modes:
    if mode=='VENT':
        plotmodes[modes.index(mode)]='senSystem_'+mode+'_cool'
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
    else:
        plotmodes[modes.index(mode)]='senSystem_'+mode
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
                   
hrArray=[hourData[hrmode] for hrmode in plotmodes]
hrArrayT = np.copy(hrArray).transpose()
coolarr=hrArrayT
heatarr=hrArrayT
           
md_labels=plotmodes
tl_size=15
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax1 = fig.add_subplot(111)
SBG.stackedBarPlot(ax1,
                   hrArrayT,
                   md_colors,
                   md_labels,
                   negStack=True,
                   xLabels=range(1,25,1),
                   yTicks=7,
                   gap=.2,
                   lblsize=tl_size,
                   scale=False,
                   endGaps=True
                   )
                   
#                   xlabel='Hour of the Day',
#                   ylabel='Sensible System Cooling Capacity (total kBTU)',

hrArraySumT=[np.sum(hrArrayT[i]) for i in range(len(hrArrayT))]
barSk=2000
ylmax=np.max(hrArraySumT)
rylmax=ylmax + barSk*(ylmax%barSk>0) - ylmax%barSk
ylmin=np.min(hrArraySumT)
fylmin=-((abs(ylmin) + barSk*(abs(ylmin)%barSk>0) - abs(ylmin)%barSk))/4
ax1.set_ylim(fylmin,((fylmin)+(rylmax-(fylmin)+barSk/2)*1.3))
wTick=min(abs(rylmax),abs(fylmin))/4
plt.yticks(range(int(fylmin),int(rylmax+wTick),int(wTick)))
#july looked better with 1000 roo scaleing
ax1.axhline(0,color='k',linestyle='-')
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
ax1.tick_params(axis='y', pad=0)
for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(tl_size)
ax2=ax1.twinx()
#ax2.set_ylabel('OA Temp ($^\circ$F)')
ax2.set_ylim(-100,110)
plt.plot(hourDF.index,hourDF.OAtempMean,'r-',label='Mean OA')
plt.plot(hourDF.index,hourDF.OAtempMin,'r--',
         hourDF.index,hourDF.OAtempMax,'r--',label='Min/Max OA')
ax2.tick_params(axis='y', which='both', labelsize=tl_size, direction="out")
plt.yticks([40, 50, 60, 70, 80, 90, 100])
#handle,label=ax1.get_legend_handles_labels()
#handle2,label2=ax2.get_legend_handles_labels()
#all_handles=handle+handle2
#all_labels=label+label2

#plt.legend(all_handles,all_labels,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
#           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1)

plt.show()

plotname='SenSysCap Mode Runtime Hours with OA stats'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use hourDF output file to create XL plots
                             
#%% Mode Stacked Cumulative Latent System Cooling (kBTU) by hour of day with OA stats


SBG = StackedBarGrapher()


md_colors=['#000000' for mc in range(len(modes))]           
plotmodes=[None for pm in range(len(modes))]           
for mode in modes:
    if mode=='VENT':
        plotmodes[modes.index(mode)]='latSystem_'+mode+'_cool'
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
    else:
        plotmodes[modes.index(mode)]='latSystem_'+mode
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
           
hrArray=[hourData[hrmode] for hrmode in plotmodes]
hrArrayT = np.copy(hrArray).transpose()
coolarr=hrArrayT
heatarr=hrArrayT           
md_labels=plotmodes
#tl_size=15
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax1 = fig.add_subplot(111)
SBG.stackedBarPlot(ax1,
                   hrArrayT,
                   md_colors,
                   md_labels,
                   negStack=True,
                   xLabels=range(1,25,1),
                   yTicks=7,
                   gap=.2,
                   lblsize=tl_size,
                   scale=False,
                   endGaps=True
                   )
#                   xlabel='Hour of the Day',
#                   ylabel='Latent System Cooling Capacity (total kBTU)',

#hrArraySumT=[np.sum(hrArrayT[i]) for i in range(len(hrArrayT))]
#ylmax=np.max(hrArraySumT)
#ylmin=np.min(hrArraySumT)
#ax1.set_ylim(ylmin-50,(ylmin-50+(ylmax-ylmin+100)*2))                   

hrArraySumT=[np.sum(hrArrayT[i]) for i in range(len(hrArrayT))]
barSk=100
ylmax=np.max(hrArraySumT)
rylmax=ylmax + barSk*(ylmax%barSk>0) - ylmax%barSk
ylmin=np.min(hrArraySumT)
fylmin=-(abs(ylmin) + barSk*(abs(ylmin)%barSk>0) - abs(ylmin)%barSk)
ax1.set_ylim(fylmin,(fylmin+(rylmax-fylmin+barSk/2)*2))
wTick=min(abs(rylmax),abs(fylmin))/2
plt.yticks(range(int(fylmin),int(rylmax+wTick),int(wTick)))
#july looked better with 1000 roo scaleing
ax1.axhline(0,color='k',linestyle='-')
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
ax1.tick_params(axis='y', pad=0)
for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(tl_size)

ax2=ax1.twinx()
#ax2.set_ylabel('OA Temp ($^\circ$F)')
ax2.set_ylim(-30,110)
plt.plot(hourDF.index,hourDF.OAtempMean,'r-',label='Mean OA')
plt.plot(hourDF.index,hourDF.OAtempMin,'r--',
         hourDF.index,hourDF.OAtempMax,'r--',label='Min/Max OA')
ax2.tick_params(axis='y', which='both', labelsize=tl_size, direction="out")
plt.yticks([40, 50, 60, 70, 80, 90, 100])
#handle,label=ax1.get_legend_handles_labels()
#handle2,label2=ax2.get_legend_handles_labels()
#all_handles=handle+handle2
#all_labels=label+label2
#
#plt.legend(all_handles,all_labels,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
#           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1)

plt.show()
                            
plotname='LatSysCap Mode Runtime Hours with OA stats'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use hourDF output file to create XL plots

#%% 

#==============================================================================
# Mode Stacked Cumulative Sensible Room Cooling (kBTU) by hour of day with OA stats
#==============================================================================

SBG = StackedBarGrapher()


md_colors=['#000000' for mc in range(len(modes))]           
plotmodes=[None for pm in range(len(modes))]           
for mode in modes:
    if mode=='VENT':
        plotmodes[modes.index(mode)]='senRoom_'+mode+'_cool'
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
    else:
        plotmodes[modes.index(mode)]='senRoom_'+mode
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
           
hrArray=[hourData[hrmode] for hrmode in plotmodes]
hrArrayT = np.copy(hrArray).transpose()
coolarr=hrArrayT
heatarr=hrArrayT
           
md_labels=plotmodes
#tl_size=15
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax1 = fig.add_subplot(111)
SBG.stackedBarPlot(ax1,
                   hrArrayT,
                   md_colors,
                   md_labels,
                   negStack=True,
                   xLabels=range(1,25,1),
                   yTicks=7,
                   gap=.2,
                   lblsize=tl_size,
                   scale=False,
                   endGaps=True
                   )
#                   xlabel='Hour of the Day',
#                   ylabel='Sensible Room Cooling Capacity (total kBTU)',

hrArraySumT=[np.sum(hrArrayT[i]) for i in range(len(hrArrayT))]

barSk=1000
ylmax=np.max(hrArraySumT)
rylmax=ylmax + barSk*(ylmax%barSk>0) - ylmax%barSk
ylmin=np.min(hrArraySumT)
fylmin=-((abs(ylmin) + barSk*(abs(ylmin)%barSk>0) - abs(ylmin)%barSk))/4
ax1.set_ylim(fylmin,((fylmin)+(rylmax-(fylmin)+barSk/2)*1.3))
wTick=min(abs(rylmax),abs(fylmin))/4
plt.yticks(range(int(fylmin),int(rylmax+wTick),int(wTick)))
#july looked better with 1000 roo scaleing
ax1.axhline(0,color='k',linestyle='-')
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
ax1.tick_params(axis='y', pad=0)
for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(tl_size)

ax2=ax1.twinx()
#ax2.set_ylabel('OA Temp ($^\circ$F)')
ax2.set_ylim(-100,110)
plt.plot(hourDF.index,hourDF.OAtempMean,'r-',label='Mean OA')
plt.plot(hourDF.index,hourDF.OAtempMin,'r--',
         hourDF.index,hourDF.OAtempMax,'r--',label='Min/Max OA')
ax2.tick_params(axis='y', which='both', labelsize=tl_size, direction="out")
plt.yticks([40, 50, 60, 70, 80, 90, 100])
#handle,label=ax1.get_legend_handles_labels()
#handle2,label2=ax2.get_legend_handles_labels()
#all_handles=handle+handle2
#all_labels=label+label2
#
#plt.legend(all_handles,all_labels,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
#           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1)


plt.show()

plotname='SenRmCap Mode Runtime Hours with OA stats'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use hourDF output file to create XL plots

#%% Mode Stacked Cumulative Latent Room Cooling (kBTU) by hour of day with OA stats
SBG = StackedBarGrapher()


md_colors=['#000000' for mc in range(len(modes))]           
plotmodes=[None for pm in range(len(modes))]           
for mode in modes:
    if mode=='VENT':
        plotmodes[modes.index(mode)]='latRoom_'+mode+'_cool'
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
    else:
        plotmodes[modes.index(mode)]='latRoom_'+mode
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
           
hrArray=[hourData[hrmode] for hrmode in plotmodes]
hrArrayT = np.copy(hrArray).transpose()
coolarr=hrArrayT
heatarr=hrArrayT
           
md_labels=plotmodes
tl_size=15
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax1 = fig.add_subplot(111)
SBG.stackedBarPlot(ax1,
                   hrArrayT,
                   md_colors,
                   md_labels,
                   negStack=True,
                   xLabels=range(1,25,1),
                   yTicks=7,
                   gap=.2,
                   xlabel='Hour of the Day',
                   ylabel='Latent Room Cooling Capacity (total kBTU)',
                   lblsize=tl_size,
                   scale=False,
                   endGaps=True
                   )

hrArraySumT=[np.sum(hrArrayT[i]) for i in range(len(hrArrayT))]
ylmax=np.max(hrArraySumT)
ylmin=np.min(hrArraySumT)
ax1.set_ylim(ylmin-50,(ylmin-50+(ylmax-ylmin+100)*2))
ax1.axhline(0,color='k',linestyle='-')
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
ax2=ax1.twinx()
ax2.set_ylabel('OA Temp ($^\circ$F)')
ax2.set_ylim(-30,110)
plt.plot(hourDF.index,hourDF.OAtempMean,'r-',label='Mean OA')
plt.plot(hourDF.index,hourDF.OAtempMin,'r--',
         hourDF.index,hourDF.OAtempMax,'r--',label='Min/Max OA')
ax2.tick_params(axis='y', which='both', labelsize=tl_size, direction="out")
plt.yticks([40, 50, 60, 70, 80, 90, 100])
handle,label=ax1.get_legend_handles_labels()
handle2,label2=ax2.get_legend_handles_labels()
all_handles=handle+handle2
all_labels=label+label2

#plt.legend(all_handles,all_labels,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
#           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1)


plt.show()

plotname='LatRmCap Mode Runtime Hours with OA stats'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use hourDF output file to create XL plots
                             
#%% Mode Stacked Cumulative Total Room Cooling (kBTU) by hour of day with OA stats

SBG = StackedBarGrapher()


md_colors=['#000000' for mc in range(len(modes))]           
plotmodes=[None for pm in range(len(modes))]           
for mode in modes:
    if mode=='VENT':
        plotmodes[modes.index(mode)]='totRoom_'+mode+'_cool'
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
    else:
        plotmodes[modes.index(mode)]='totRoom_'+mode
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
           
hrArray=[hourData[hrmode] for hrmode in plotmodes]
hrArrayT = np.copy(hrArray).transpose()
coolarr=hrArrayT
heatarr=hrArrayT
           
md_labels=plotmodes
tl_size=15
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax1 = fig.add_subplot(111)
SBG.stackedBarPlot(ax1,
                   hrArrayT,
                   md_colors,
                   md_labels,
                   negStack=True,
                   xLabels=range(1,25,1),
                   yTicks=7,
                   gap=.2,
                   xlabel='Hour of the Day',
                   ylabel='Total Room Cooling Capacity (total kBTU)',
                   lblsize=tl_size,
                   scale=False,
                   endGaps=True
                   )

hrArraySumT=[np.sum(hrArrayT[i]) for i in range(len(hrArrayT))]
ylmax=np.max(hrArraySumT)
ylmin=np.min(hrArraySumT)
ax1.set_ylim(ylmin-200,(ylmin-200+(ylmax-ylmin+400)*2))
ax1.axhline(0,color='k',linestyle='-')
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
ax2=ax1.twinx()
ax2.set_ylabel('OA Temp ($^\circ$F)')
ax2.set_ylim(-30,110)
plt.plot(hourDF.index,hourDF.OAtempMean,'r-',label='Mean OA')
plt.plot(hourDF.index,hourDF.OAtempMin,'r--',
         hourDF.index,hourDF.OAtempMax,'r--',label='Min/Max OA')
ax2.tick_params(axis='y', which='both', labelsize=tl_size, direction="out")
plt.yticks([40, 50, 60, 70, 80, 90, 100])
handle,label=ax1.get_legend_handles_labels()
handle2,label2=ax2.get_legend_handles_labels()
all_handles=handle+handle2
all_labels=label+label2

#plt.legend(all_handles,all_labels,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
#           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1)


plt.show()

plotname='TotRmCap Mode Runtime Hours with OA stats'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use hourDF output file to create XL plots

#%% Mode Stacked Cumulative Total System Cooling (kBTU) by hour of day with OA stats
SBG = StackedBarGrapher()


md_colors=['#000000' for mc in range(len(modes))]           
plotmodes=[None for pm in range(len(modes))]           
for mode in modes:
    if mode=='VENT':
        plotmodes[modes.index(mode)]='totSystem_'+mode+'_cool'
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
    else:
        plotmodes[modes.index(mode)]='totSystem_'+mode
        md_colors[modes.index(mode)]=mode_colors[mode_type.index(mode)]
           
hrArray=[hourData[hrmode] for hrmode in plotmodes]
hrArrayT = np.copy(hrArray).transpose()
coolarr=hrArrayT
heatarr=hrArrayT
           
md_labels=plotmodes
tl_size=15
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax1 = fig.add_subplot(111)
SBG.stackedBarPlot(ax1,
                   hrArrayT,
                   md_colors,
                   md_labels,
                   negStack=True,
                   xLabels=range(1,25,1),
                   yTicks=7,
                   gap=.2,
                   xlabel='Hour of the Day',
                   ylabel='Total System Cooling Capacity (total kBTU)',
                   lblsize=tl_size,
                   scale=False,
                   endGaps=True
                   )

hrArraySumT=[np.sum(hrArrayT[i]) for i in range(len(hrArrayT))]
ylmax=np.max(hrArraySumT)
ylmin=np.min(hrArraySumT)
ax1.set_ylim(ylmin-200,(ylmin-200+(ylmax-ylmin+400)*2))
ax1.axhline(0,color='k',linestyle='-')
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
ax2=ax1.twinx()
ax2.set_ylabel('OA Temp ($^\circ$F)')
ax2.set_ylim(-30,110)
#linprop=range(55,105,5)
#for oatemp in linprop:
#    ax2.axhline(oatemp,color='k',linestyle='-.')
plt.plot(hourDF.index,hourDF.OAtempMean,'r-',label='Mean OA')
plt.plot(hourDF.index,hourDF.OAtempMin,'r--',
         hourDF.index,hourDF.OAtempMax,'r--',label='Min/Max OA')
#plt.yticks(range(55,105,5),str(lab) for lab in range(55,105,5))
ax2.tick_params(axis='y', which='both', labelsize=tl_size, direction="out")
plt.yticks([40, 50, 60, 70, 80, 90, 100])
handle,label=ax1.get_legend_handles_labels()
handle2,label2=ax2.get_legend_handles_labels()
all_handles=handle+handle2
all_labels=label+label2

plt.legend(all_handles,all_labels,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1)


plt.show()

plotname='TotSysCap Mode Runtime Hours with OA stats'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use hourDF output file to create XL plots
                             
#%%stack plot play  a general for for making a negStack barplot only colors for 5 bars max
                             
#SBG = StackedBarGrapher()
#
#md_colors=['#9df5a8', '#07913C', '#00b5fc',
#           '#0061fc', '#F59DE8']
#           
#           
#ArrPlay=np.random.randn(2,5)
#md_labels=plotmodes
#tl_size=9
#fig = plt.figure()
#
#ax1 = fig.add_subplot(111)
#SBG.stackedBarPlot(ax1,
#                   ArrPlay,
#                   md_colors,
#                   md_labels,
#                   negStack=True,
#                   xLabels=range(1,25,1),
#                   yTicks=10,
#                   gap=.2,
#                   xlabel='Hour of the Day',
#                   ylabel='Total Runtime Hours',
#                   lblsize=tl_size,
#                   scale=False,
#                   endGaps=True
#                   )
#                   
#ax1.set_ylim(-6,6)
#
#handle,label=ax1.get_legend_handles_labels()
#all_handles=handle#+handle2
#all_labels=label#+label2
#
#plt.legend(all_handles,all_labels,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
#           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1)
#
#
#plt.show()
#                           

#%% Sensible System Cooling Capacity vs OA

fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax = fig.add_subplot(111)
for key,data in df.groupby(['opMode','opModeStatus']):
    if 'Transient' not in key and 'OFF' not in key:
        mColor=mode_colors[mode_type.index(key[0])]
        if key[0]=='HEAT1':
            plt.plot(data.OA_Temp,abs(data.senSystemCap),'.',c=mColor,label=key[0],
                     alpha=alFA)
        else:
            plt.plot(data.OA_Temp,data.senSystemCap,'.',c=mColor,label=key[0],
                     alpha=alFA)
ax.set_ylim(-50,350)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)
#plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
#           borderaxespad=0,prop={'size':9},numpoints=1)
#plt.ylabel('Sensible System Capcity (kBTU/hr)')
#plt.xlabel('Outside Air Temperature ($^\circ$F)')

# this ouput file can be used for all raw DF based graphs
plotname='Sensible System Capacity vs OA'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')
#%%                             
if createCSV: df.to_csv(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                        str(obsMonth)+'_rawDF.csv',na_rep='=NA()')

# use rawDF output file to create XL plots

#%% Sensible System COP vs OA
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax = fig.add_subplot(111)
for key,data in df.groupby(['opMode','opModeStatus']):
    if 'Transient' not in key and 'OFF' not in key and 'HEAT1' not in key and 'VENT' not in key:
        mColor=mode_colors[mode_type.index(key[0])]
        plt.plot(data.OA_Temp,data.senSystemCOP,'.',c=mColor,label=key[0],alpha=alFA)
ax.set_ylim(-2,8)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)
#plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
#           borderaxespad=0,prop={'size':9},numpoints=1)
#plt.ylabel('Sensible System COP')
#plt.xlabel('Outside Air Temperature ($^\circ$F)')

plotname='Sensible System COP vs OA'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use rawDF output file to create XL plots

#%% Sensible ROOM Cooling Capcity vs OA
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax = fig.add_subplot(111)
for key,data in df.groupby(['opMode','opModeStatus']):
    if 'Transient' not in key and 'OFF' not in key:
        mColor=mode_colors[mode_type.index(key[0])]
        if key[0]=='HEAT1':
            plt.plot(data.OA_Temp,abs(data.senRoomCap),'.',c=mColor,label=key[0],
                     alpha=alFA)
        else:
            plt.plot(data.OA_Temp,data.senRoomCap,'.',c=mColor,label=key[0],
                     alpha=alFA)
ax.set_ylim(-50,350)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)
#plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
#           borderaxespad=0,prop={'size':9},numpoints=1)
#plt.ylabel('Sensible Room Capacity (kBTU/hr)')
#plt.xlabel('Outside Air Temperature ($^\circ$F)')

plotname='Sensible Room Capacity vs OA'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use rawDF output file to create XL plots

#%% Sensible ROOM COP vs OA
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax = fig.add_subplot(111)
for key,data in df.groupby(['opMode','opModeStatus']):
    if 'Transient' not in key and 'OFF' not in key and 'HEAT1' not in key and 'VENT' not in key: 
        mColor=mode_colors[mode_type.index(key[0])]
        plt.plot(data.OA_Temp,data.senRoomCOP,'.',c=mColor,label=key[0],
                 alpha=alFA)
ax.set_ylim(-2,8)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)
#plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
#           borderaxespad=0,prop={'size':9},numpoints=1)
#plt.ylabel('Sensible Room COP')
#plt.xlabel('Outside Air Temperature ($^\circ$F)')

plotname='Sensible Room COP vs OA'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use rawDF output file to create XL plots

#%% kW System vs OA
fig=plt.figure()
fig.set_size_inches(wFig,hFig)
ax=fig.add_subplot(111)
for key,data in df.groupby(['opMode','opModeStatus']):
    if 'Transient' not in key and 'OFF' not in key:
        mColor=mode_colors[mode_type.index(key[0])]
        plt.plot(data.OA_Temp,data.kW_System,'.',c=mColor,label=key[0],
                 alpha=alFA)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)
#plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
#           borderaxespad=0,prop={'size':9},numpoints=1)
#plt.ylabel('System Instantaneous Power (kW)')
#plt.xlabel('Outside Air Temperature ($^\circ$F)')

plt.yticks(range(0,15,2))

plotname='kW System vs OA'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use rawDF output file to create XL plots

#%% DualCool water coil effectiveness vs OAwbd
#colorMap=0
#plt.figure()
#plt.ylim(0,1)
#for key,data in df.groupby(['opMode','opModeStatus','pumpStatus']):
#    if 'DC' in key[0] and 1 in key and 'Transient' not in key:
#        plt.plot(data.OAwbd,data.dcEffectiveness,'.',c=getColor(1),
#                 label=key[0],alpha=alFA)
#plt.ylabel('DC Water Coil Effectiveness')
#plt.xlabel('Outside Air Wet Bulb Depression ($^\circ$F)')
#plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
#           borderaxespad=0,prop={'size':9},numpoints=1)
#if outputGraphs: plt.savefig(outputPath+'Water coil effectiveness vs '+\
#                             'OAwbd.png',bbox_to_inches='tight')
#
#%% DualCool water temperature
#colorMap=0
#plt.figure()
#Tcolors=brewer2mpl.get_map('Accent','Qualitative',3).hex_colors
#for key,data in df.groupby(['opMode','opModeStatus']):
#    if 'DC' in key[0] and 'Transient' not in key:
#        plt.plot(data.OAwb,data.Twcin_F,'.',c=Tcolors[0],alpha=alFA)
#        plt.plot(data.OAwb,data.Twcout_F,'.',c=Tcolors[1],alpha=alFA)
#        plt.plot(data.OAwb,data.Tsump_F,'.',c=Tcolors[2],alpha=alFA)
#        plt.plot(data.OAwb,data.OAwb,c='k')
#plt.xlabel('Outside Air Wet Bulb Temperature ($^\circ$F)')
#plt.ylabel('Temperature ($^\circ$F)')
#plt.legend(('Twcin','Twcout','Tsump'),bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,
#           ncol=5,mode='expand',borderaxespad=0,prop={'size':9},numpoints=1)
#if outputGraphs: plt.savefig(outputPath+'dualcool water temperature.png',
#                             bbox_to_inches='tight')
#
#%% Ventilation cooling
#colorMap=0
#fig,ax1=plt.subplots()
#ax2=plt.twinx()
#for key,data in df.groupby(['opMode','opModeStatus','pumpStatus']):
#   if 'DC' in key[0] and 1 in key and 'Transient' not in key:
#        ax1.plot(data.OAwbd,(data.OA_Temp-data.PA_Temp),'.',
#                 c=getColor(1),label=key[0]+'_Air',alpha=alFA)
#        ax2.plot(data.OAwbd,(data.Twcout_F-data.Twcin_F),'.',
#                 c=getColor(1),label=key[0]+'__Water Coil',alpha=alFA)
#handles1,labels1=ax1.get_legend_handles_labels()
#handles2,labels2=ax2.get_legend_handles_labels()
#handles=handles1+handles2
#labels=labels1+labels2
#plt.legend(handles,labels,bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,
#           mode='expand',borderaxespad=0,prop={'size':9},numpoints=1)
#ax1.set_ylabel('Air $\Delta$T across DualCool Coil ($^\circ$F)')
#ax2.set_ylabel('Water $\Delta$T across DualCool Coil ($^\circ$F)')
#plt.xlabel('Outside Air Wetbulb Depression ($^\circ$F)')
#if outputGraphs: plt.savefig(outputPath+'Ventilation air deltaT'+\
#                             ' across DC coil.png',bbox_to_inches='tight')
#
#%% DC coil wet bulb effectiveness
#colorMap=0
#plt.figure()
#for key,data in df.groupby(['opMode','opModeStatus','pumpStatus']):
#    if 'DC' in key[0] and 1 in key and 'Transient' not in key:
#        plt.plot(data.OAwbd,data.OAwbe,'.',c=getColor(1),label=key[0],
#                 alpha=alFA)
#plt.ylabel('Wet Bulb Effectiveness (%)')
#plt.xlabel('Outside Air Wet Bulb Depression ($^\circ$F)')
#plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
#           borderaxespad=0,prop={'size':9},numpoints=1)
#if outputGraphs: plt.savefig(outputPath+'DC Coil WBE.png',
#                             bbox_to_inches='tight')
#    
#%% OA Temp vs SA Temp no off and transients
fig = plt.figure()
fig.set_size_inches(wFig,hFig)
ax = fig.add_subplot(111)
for key,data in df.groupby(['opMode','opModeStatus']):
    if 'Transient' not in key and 'OFF' not in key:
        mColor=mode_colors[mode_type.index(key[0])]
        plt.plot(data.OA_Temp,data.SA_Temp,'.',c=mColor,label=key[0],
                 alpha=alFA)
plt.ylabel('Supply Air Temp')
plt.xlabel('Outside Air Temp ($^\circ$F)')
plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
           borderaxespad=0,prop={'size':9},numpoints=1)

plotname='OA Temp vs SA Temp'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use rawDF output file to create XL plots

#%% OA Temp vs SA Temp with transients but no off
fig=plt.figure()
fig.set_size_inches(wFig,hFig)
ax=fig.add_subplot(111)
for key,data in df.groupby(['opMode','opModeStatus']):
    if 'OFF' not in key:
        mColor=mode_colors[mode_type.index(key[0])]
        plt.plot(data.OA_Temp,data.SA_Temp,'.',c=mColor,label=key[0],
                 alpha=alFA)
#plt.ylabel('Supply Air Temp')
#plt.xlabel('Outside Air Temp ($^\circ$F)')
ax.set_ylim(40,120)
ax.set_xlim(40,120)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)
#plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
#           borderaxespad=0,prop={'size':9},numpoints=1)
#if outputGraphs: plt.savefig(outputPath+'OA Temp vs SA Temp.png',
#                             bbox_to_inches='tight')

plotname='OA Temp vs SA Temp inc trasient'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use rawDF output file to create XL plots

#%% RA Temp vs SA Temp with transients but no off

fig=plt.figure()
fig.set_size_inches(wFig,hFig)
ax=fig.add_subplot(111)
for key,data in df.groupby(['opMode','opModeStatus']):
    if 'OFF' not in key:
        mColor=mode_colors[mode_type.index(key[0])]
        plt.plot(data.RA_Temp,data.SA_Temp,'.',c=mColor,label=key[0],
                 alpha=alFA)
#plt.ylabel('Supply Air Temp')
#plt.xlabel('Room Air Temp ($^\circ$F)')
ax.set_ylim(40,120)
ax.set_xlim(65,80)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)
#plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
#           borderaxespad=0,prop={'size':9},numpoints=1)
#if outputGraphs: plt.savefig(outputPath+'OA Temp vs SA Temp.png',
#                             bbox_to_inches='tight')

plotname='RA Temp vs SA Temp inc trasient'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use rawDF output file to create XL plots

#%% hourly Water use by aveage OA temp and OA wbd
fig=plt.figure()
fig.set_size_inches(wFig,hFig)
ax=fig.add_subplot(111)
#plt.plot(WRdf[(WRdf.kW_System>4) & (WRdf.Vwater_Gal<1) & (WRdf.Vwater_Gal>=0.001)].resample('h',how='mean').OAwbd,
#         WRdf[(WRdf.kW_System>4) & (WRdf.Vwater_Gal<1) & (WRdf.Vwater_Gal>=0.001)].resample('h',how='mean').Vwater_Gal,
#         '.', c='b', label='hourly Gal vs OA wbd', alpha=alFA)             
plt.plot(df.resample('h',how='mean').OA_Temp,
         (df.resample('h',how='mean').pVwater_GPM * 60),
         '.', c='b', label='hourly Gal vs OA temp', alpha=alFA)
plt.plot(WRdf.resample('h',how='mean').OA_Temp,
         (WRdf.resample('h',how='mean').Vwater_Gal * 60),
         '.', c='r', label='hourly Gal vs OA temp', alpha=alFA)
#         np.dot(exog, WRres.params),
#         '.', c='b', label='hourly Gal vs OA temp', alpha=alFA)

#plt.ylabel('Gal')
#plt.xlabel('Air Temp ($^\circ$F)')
ax.set_ylim(-5,35)
ax.set_xlim(40,110)
#plt.legend(bbox_to_anchor=(0. ,1.02 ,1.,0.3),loc=8,ncol=5,mode='expand',
#           borderaxespad=0,prop={'size':9},numpoints=1)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(tl_size)

plotname='hourly Water use by aveage OA temp model data and predicted'

if outputGraphs: plt.savefig(outputPath+str(obsYear)+'_'+str(placeHolder)+\
                             str(obsMonth)+'_'+plotname+'.png',
                             bbox_to_inches='tight')

# use rawDF output file to create XL plots

#%% Airflow by hour
#colorMap=0
#plt.figure()
#plt.bar(hourDF.index,hourDF.mDotSA,color=getColor(1),align='center')
#plt.ylim(0,9000)
#plt.xlim(-1,24)
#if outputGraphs: plt.savefig(outputPath+'Supply Airflow by Hour.png',
#                             bbox_to_inches='tight',pad_inches=0)
#    

#%%  OA fraction by hour
#colorMap=0
#plt.figure()
#plt.bar(hourDF.index,hourDF.mDotOA,color= getColor(1),align='center')
#plt.xlim(-1,24)
#plt.ylim(0,1)
#if outputGraphs: plt.savefig(outputPath+'OA Fraction by Hour.png',
#                             bbox_to_inches='tight',pad_inches=0)
#%% Output file

#if createCSV: df.to_csv(outputPath+'python outputs.csv'):

#==============================================================================
# fileforJay=df[df.OA_Temp>=85]
# colsforJay=['opMode','vDotSA','vDotOA','OA_Temp','OA_RH','RA_Temp','RA_RH',
#             'SA_Temp','SA_RH','kW_System']
#==============================================================================
#==============================================================================
# headerforJay=['Current Operating Mode','Supply Airflow (CFM)',
#               'Outside Airflow (CFM)','OA_Temp (F)','OA_RH(%)','RA_Temp (F)',
#               'RA_RH (%)','SA_Temp (F)','SA_RH (%)','System Power (kW)']
#==============================================================================

if createCSV: df.to_csv(outputPath+str(obsYear)+'_'+str(placeHolder)+\
           str(obsMonth)+'_Munters_EXP_5000_at_SRWF_python.csv',na_rep='=NA()')
