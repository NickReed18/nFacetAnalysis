import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from scipy.stats import mode

class CouplingAnalysis():
    """
    Wrapper class for gamma calibration analysis. Takes root files from Co60 data runs and 
    performs calibration analysis to align channels across each FPGA. After running the 
    correction factor calculation, saves the correction factors to a .txt file. Can also be used
    for calculating the detection efficiency 

    ATTRIBUTES
    ------------------------
    data : dict
        Dictionary for storing processed data from ROOT files in, in preparation for further analysis.
    
    set : dict
        Dictionary for storing run settings from ROOT files.        

    cwind : int
        Timing window in samples for calculating coincidence between two events. If the time 
        separation in samples between two events is less than this value, they are considered 
        to be coincident.

    fits : dict
        Dictionary to store bin values predicted by the half-Gaussian fit.

    dfs : dict
        Dictionary to store the DataFrames of runs that have been fitted. 

    factors : dict
        Dictionary to store the calculated correction factors per run.

    reconstructed : dict
        Dictionary to store a list of reconstructed cubes after cube reconstruction. 

    cubedicts : dict
        Dictionary to store calculated efficiency for detecting Co60 gamma rays per channel.

    channel_convs : dict
        Dictionary to store factors to convert from ADC to energy per channel.


    METHODS
    ------------------------
    add_data(self, dfile)
        Loads a ROOT file and performs pre-processing in order to prepare the data for 
        gamma calibration analysis.

    gen_df(self, dfile, binsize, maxval, coincidence_cut = 2, bkg = None, ToTcut = 20, corrected = False)
        Performs the first fit of a half-Gaussian to the data, for each channel. First a histogram 
        of the peak value over pedestal values is calculated, and then the fit is done to the histogram bins. 
        If a background data file is provided, also performs background subtraction before doing calculating 
        this fit. The histogram, fit parameters, chi-squared and fractional error on fit are all recorded in the
        DataFrame, which is saved to self.dfs.

    refit_mean(self, dfile, chsqthresh = 2)
        Re-calculates the fit for channels with a chi-squared greater than a provided threshold, using the mean
        of the other fits as an initial guess for the optimisation of the fit. Saves the updated fit back to the
        same DataFrame in self.dfs.
    """
    def __init__(self, coincidence_window = 100):
        """
        Initializes dictionaries for storing data and DataFrames.

        PARAMETERS
        ------------------------
        coincidence_window : int
            The timing window for determining coincident events. Default is 100 samples.
        """
        self.data={}
        self.set={}
        self.cwind=coincidence_window
        self.fits={}
        self.dfs={}
        self.factors={}
        self.reconstructed={}
        self.cubedicts={}
        self.channel_convs={}
        
    def add_data(self, dfile):
        """
        Takes in a ROOT file and reconstructs the x, y and z coordinates of each channel, 
        and calculates coincidences. The reconstructed data is then saved to the self.data dictionary.

        PARAMETERS
        ------------------------
        dfile : str
            Path to the ROOT file to be saved. Its label in the dictionary is the same as this string.
        """
        file=uproot.open(dfile)
        ends=[]
        for key in file.keys():
            if key[0:1]==b'e':
                ends.append(int(key[7:]))
        maxkey=max(ends)
        data=file['events;{}'.format(maxkey)].pandas.df(flatten=False)
        settings=file['settings'].pandas.df(flatten=False)
        data['x']=-2
        data['y']=-2
        data['z']=-2
        data['subtracted']=0
        for chann in range(len(settings)):
            data.loc[(data[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'x']=settings.loc[chann,'x']  
            data.loc[(data[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'y']=settings.loc[chann,'y']  
            data.loc[(data[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'z']=settings.loc[chann,'z']  
            data.loc[(data[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'subtracted']=data.loc[(data[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'peakValue']-settings.loc[chann,'pedestal']
        data.sort_values(['time'],inplace=True)
        data['deltaT']=data['time'].diff()
        data['inWindow']=data['deltaT']<self.cwind
        data['EventNum']=(~data['inWindow']).cumsum()
        data.set_index('EventNum',inplace=True)
        data['coincidence']=data.groupby(data.index)['totValue'].count()
        self.data[dfile]=data
        self.set[dfile]=settings
    
    def gen_df(self, dfile, binsize, maxval, coincidence_cut = 2, bkg = None, ToTcut = 20, corrected = False):
        """
        Generates a histogram of the peak over pedestal values, then performs a half-Gaussian fit to the histogram values, 
        for each channel. Appends a DataFrame containing the histogram, fit parameters, fit fractional error and chi-squared
        to the self.dfs dictionary. Can optionally also perform background subtraction.

        PARAMETERS
        ------------------------
        dfile : str
            Name of the desired data set to process, as stored in self.data i.e. the path used when loading the data set.

        binsize : int
            Size of each histogram bin in ADC. 

        maxval : int
            Upper range for bin calculation.

        coincidence_cut : int
            The coincidence value to cut on to determine valid gamma events. Any events with a coincidence less than this value
            are considered valid for the calibration analysis. Default value is 2, i.e. events with coincidence < 2 are used 
            for the analysis.

        bkg : str
            Name of the background data set for background subtraction, as stored in self.data. Default value is None.

        ToTcut : int
            Value of time over threshold to identify a gamma event. Events are identified as gamma events if they have a 
            ToT value less than or equal to this threshold. Default value is 20. 

        corrected : bool
            Boolean flag to identify if this is a first attempt at fitting, or a refit of a scaled channel. Default value 
            is False.
        """
        nbins=int(maxval/binsize)
        space=maxval/nbins
        bins=np.linspace(0,maxval,nbins)
        centres=(bins + space/2)[:-1]
        data=self.data[dfile]
        bkg=self.data[bkg]
        
        def half_gauss(x,mu,sigma,amp):
            return np.heaviside(x-mu,1)*amp*np.exp(-1*(x-mu)**2/(2*sigma**2)) + np.heaviside(mu-x,0)
        
        
        for i in range(2):
            for j in range(16):
                d=data.query('fpga=={}&channel=={}&coincidence=={}&totValue<{}'.format(i,j,coincidence_cut,ToTcut))
                if corrected:
                    dhist,dbins=np.histogram(d['corrected'],bins=bins)
                else:
                    dhist,dbins=np.histogram(d['subtracted'],bins=bins)
                b=bkg.query('fpga=={}&channel=={}&coincidence=={}&totValue<{}'.format(i,j,coincidence_cut,ToTcut))
                if corrected:
                    bhist,bbins=np.histogram(b['corrected'],bins=bins)
                else:
                    bhist,bbins=np.histogram(b['subtracted'],bins=bins)
                dt=d['time'].values[-1]
                bt=b['time'].values[-1]

                min_t = min(dt,bt)
                subhist = (dhist/dt - bhist/bt)*min_t
                subhist2 = np.copy(subhist)
                subhist2[:np.argmax(subhist)]=1
                suberror = min_t*np.sqrt(dhist/dt**2 + bhist/bt**2)
                suberror2 = np.copy(suberror)
                suberror2[:np.argmax(subhist)]=1
                      
                #med=np.median(np.repeat(centres[:np.argmax(subhist<0)],subhist[:np.argmax(subhist<0)].astype(int)))
                
                p0=[centres[np.argmax(subhist)],np.sqrt(d['subtracted'].var()),np.max(subhist2)]
                popt,pcov=curve_fit(half_gauss,centres,subhist2,p0,sigma=suberror2)
                fit=half_gauss(centres,popt[0],popt[1],popt[2])
                resid=(subhist2-fit)/suberror2
                def chsq(obv,exp,error):
                    return np.sum((obv-exp)**2/(error)**2)/(len(centres)-3)
                    
                chs=chsq(subhist2,fit,suberror2)
                t={'fpga':i,'channel':j,'centres':centres,'data':subhist,'error':suberror,'space':space,'p0':p0,'popt':popt,'pcov':pcov,'fit':fit,'residuals':resid,'chsq':chs}
                entry=pd.DataFrame.from_dict(t,orient='index').transpose()    
            
                if (i==0)&(j==0):
                    df=entry
                else:
                    df=pd.concat((df,entry),axis=0)
        if corrected:
            self.dfs[dfile+"_corrected"]=df
        else:
            self.dfs[dfile]=df
        

    def refit_mean(self,dfile,chsqthresh=2):
        """
        Recalculates the half-Gaussian fit for channels with a chi-squared value 
        """
        df=self.dfs[dfile]
        
        mu0,sig0=df.query("(fpga==0)&chsq<{}".format(chsqthresh))['popt'].explode().values.reshape(-1,3)[:,:2].mean(axis=0)
        mu1,sig1=df.query("(fpga==1)&chsq<{}".format(chsqthresh))['popt'].explode().values.reshape(-1,3)[:,:2].mean(axis=0)
        refs=df.query("chsq>{}".format(chsqthresh))
        refs.reset_index(drop=True,inplace=True)
        fits=df['fit'].values
        opts=df['popt'].values
        covs=df['pcov'].values
        residuals=df['residuals'].values
        chisq=df['chsq'].values
        
        for ind in refs.index:
            subhist=refs.loc[ind,'data']
            suberror=refs.loc[ind,'error']
            cent=refs.loc[ind,'centres']
            hist2=np.copy(subhist)
            err2=np.copy(suberror)
            hist2[:np.argmax(subhist)]=1
            err2[:np.argmax(subhist)]=1
            
            def half_gauss(x,mu,sigma,amp):
                return np.heaviside(x-mu,1)*amp*np.exp(-1*(x-mu)**2/(2*sigma**2)) + np.heaviside(mu-x,0)        
            def chsq(obv,exp,error):
                return np.sum((obv-exp)**2/(error)**2)/(len(cent)-3)
            
            if refs.iloc[ind]['fpga']==0:
                mu=mu0
                sig=sig0
            if refs.iloc[ind]['fpga']==1:
                mu=mu1
                sig=sig1
            p0=[mu,sig,np.max(subhist)]
            popt,pcov=curve_fit(half_gauss,cent,hist2,p0,sigma=err2)
            fit=half_gauss(cent,popt[0],popt[1],popt[2])
            resid=(hist2-fit)/err2
            i=np.where((df['fpga'].values==refs.loc[ind,'fpga'])&(df['channel'].values==refs.loc[ind,'channel']))[0][0]
            fits[i]=fit
            covs[i]=pcov
            opts[i]=popt
            residuals[i]=resid
            chisq[i]=chsq(hist2,fit,err2)
            
        df['fit']=fits
        df['popt']=opts
        df['pcov']=covs
        df['residuals']=residuals
        df['chsq']=chisq
            
    
    def individual_gauss_fit(self,dfile):

        df=self.dfs[dfile]
        
        fig,(ax1,ax2)=plt.subplots(2,1,gridspec_kw={'height_ratios':[3,1]})
        self.curr_pos=0
        f=df.iloc[self.curr_pos]
        
        def key_event(e):
            if e.key=="right":
                self.curr_pos+=1
                self.curr_pos%=len(df)
                f=df.iloc[self.curr_pos]
                
                ax1.cla()
                ax2.cla()
                
                fig.suptitle("FPGA {}, channel {}".format(f['fpga'],f['channel']),fontsize=24)
                ax1.bar(f['centres'],f['data'],width=f['space'],alpha=0.3,label='Data')
                ax1.plot(f['centres'],f['fit'],c='r',label='Fit')
                ax1.set_xlabel('Peak value over pedestal',fontsize=20)
                ax1.set_ylabel('Frequency',fontsize=20)
                ax1.set_xlim(0,6000)
                ax1.plot(np.repeat(int(f['popt'][:2].sum()),100),np.linspace(0,np.nanmax(f['data']),100),label=r'Fit $\mu$ + $\sigma$')
                ax1.legend(loc='upper right')
                ax1.text(int(f['popt'][:2].sum())+200,int(np.nanmax(f['fit'])),r"$\mu$ + $\sigma%$ = "+str(int(f['popt'][:2].sum())))
                
                ax2.scatter(f['centres'],f['residuals'],marker='+')
                ax2.set_ylabel(r'Residuals \ $\sigma$',fontsize=20)
                ax2.set_xlim(0,6000)
                ma=np.nanmax(np.abs(f['residuals']))+1
                ax2.set_ylim(-1*ma,ma)
                ax2.fill_between((0,6000),(-1,-1),(1,1),alpha=0.3,label=r'1 $\sigma$')
                ax2.legend(loc='upper right')
                
                fig.canvas.draw()
                
            elif e.key=="left":
                self.curr_pos-=1
                self.curr_pos%=len(df)
                f=df.iloc[self.curr_pos]
                
                ax1.cla()
                ax2.cla()
                
                fig.suptitle("FPGA {}, channel {}".format(f['fpga'],f['channel']),fontsize=24)
                ax1.bar(f['centres'],f['data'],width=f['space'],alpha=0.3,label='Data')
                ax1.plot(f['centres'],f['fit'],c='r',label='Fit')
                ax1.set_xlabel('Peak value over pedestal',fontsize=20)
                ax1.set_ylabel('Frequency',fontsize=20)
                ax1.set_xlim(0,6000)
                ax1.plot(np.repeat(int(f['popt'][:2].sum()),100),np.linspace(0,max(f['data']),100),label=r'Fit $\mu$ + $\sigma$')
                ax1.legend(loc='upper right')
                ax1.text(int(f['popt'][:2].sum())+200,int(np.nanmax(f['fit'])),r"$\mu$ + $\sigma%$ = "+str(int(f['popt'][:2].sum())))
                
                ax2.scatter(f['centres'],f['residuals'],marker='+')
                ax2.set_ylabel(r'Residuals \ $\sigma$',fontsize=20)
                ax2.set_xlim(0,6000)
                ma=np.nanmax(np.abs(f['residuals']))+1
                ax2.set_ylim(-1*ma,ma)
                ax2.fill_between((0,6000),(-1,-1),(1,1),alpha=0.3,label=r'1 $\sigma$')
                ax2.legend(loc='upper right')
                
                fig.canvas.draw()
                
        
        fig.canvas.mpl_connect('key_press_event',key_event)
                
                
        fig.suptitle("FPGA {}, channel {}".format(f['fpga'],f['channel']),fontsize=24)
        ax1.bar(f['centres'],f['data'],width=f['space'],alpha=0.3,label='Data')
        ax1.plot(f['centres'],f['fit'],c='r',label='Fit')
        ax1.set_xlabel('Peak value over pedestal',fontsize=20)
        ax1.set_ylabel('Frequency',fontsize=20)
        ax1.set_xlim(0,6000)
        ax1.plot(np.repeat(int(f['popt'][:2].sum()),100),np.linspace(0,max(f['data']),100),label=r'Fit $\mu$ + $\sigma$')
        ax1.legend(loc='upper right')
        ax1.text(int(f['popt'][:2].sum())+200,int(np.nanmax(f['fit'])),r"$\mu$ + $\sigma%$ = "+str(int(f['popt'][:2].sum())))
                
        ax2.scatter(f['centres'],f['residuals'],marker='+')
        ax2.set_ylabel(r'Residuals \ $\sigma$',fontsize=20)
        ax2.set_xlim(0,6000)
        ma=np.nanmax(np.abs(f['residuals']))+1
        ax2.set_ylim(-1*ma,ma)
        ax2.fill_between((0,6000),(-1,-1),(1,1),alpha=0.3,label=r'1 $\sigma$')
        ax2.legend(loc='upper right')
                
        plt.show()
        
        
    def plot_hists(self,dfile):
        data=self.dfs[dfile]
        fig,ax=plt.subplots(2,4,figsize=(40,20))
        for i in range(2):
            for j in range(4):
                d=data.iloc[16*i+4*j:16*i+4*j+4]
                ax[i,j].bar(d.query('fpga=={}&channel=={}'.format(str(i),str(4*j)))['centres'].values[0],
                             d.query('fpga=={}&channel=={}'.format(str(i),str(4*j)))['data'].values[0],
                             width=d.query('fpga=={}&channel=={}'.format(str(i),str(4*j)))['space'].values[0],
                             label="fpga {}, channel {}".format(str(i),str(4*j)),alpha=0.3,linewidth=0)
                ax[i,j].bar(d.query('fpga=={}&channel=={}'.format(str(i),str(4*j+1)))['centres'].values[0],
                             d.query('fpga=={}&channel=={}'.format(str(i),str(4*j+1)))['data'].values[0],
                             width=d.query('fpga=={}&channel=={}'.format(str(i),str(4*j+1)))['space'].values[0],
                             label='fpga {}, channel {}'.format(str(i),str(4*j+1)),alpha=0.3,linewidth=0)
                ax[i,j].bar(d.query('fpga=={}&channel=={}'.format(str(i),str(4*j+2)))['centres'].values[0],
                             d.query('fpga=={}&channel=={}'.format(str(i),str(4*j+2)))['data'].values[0],
                             width=d.query('fpga=={}&channel=={}'.format(str(i),str(4*j+2)))['space'].values[0],
                             label='fpga {}, channel {}'.format(str(i),str(4*j+2)),alpha=0.3,linewidth=0)
                ax[i,j].bar(d.query('fpga=={}&channel=={}'.format(str(i),str(4*j+3)))['centres'].values[0],
                             d.query('fpga=={}&channel=={}'.format(str(i),str(4*j+3)))['data'].values[0],
                             width=d.query('fpga=={}&channel=={}'.format(str(i),str(4*j+3)))['space'].values[0],
                             label='fpga {}, channel {}'.format(str(i),str(4*j+3)),alpha=0.3,linewidth=0)
                ax[i,j].legend(loc='upper right')
                ax[i,j].set_xlabel('Peak value over pedestal',fontsize=20)
                ax[i,j].set_ylabel('Frequency',fontsize=20)
                            
                            
        fig.suptitle(dfile,fontsize=24)
        plt.show()
        
        
    def fit_param_dist(self,dfile,sep=False):
        df=self.dfs[dfile]
        musig=pd.DataFrame(df['popt'].explode().values.reshape(-1,3),columns=['mu','sigma','amp'])
        musig.set_index(df.index,inplace=True)
        plotdf=pd.concat([df.loc[:,['fpga','channel','chsq']],musig],axis=1)
        if sep:
            plot0=plotdf.query('fpga==0')
            plot1=plotdf.query('fpga==1')
            fig,ax=plt.subplots(2,2,figsize=(10,20))
            fig.set_tight_layout(True)
            ax[0,0].bar((plot0['fpga']*16+plot0['channel']),plot0['mu']+plot0['sigma'],width=0.8)
            ax[0,0].plot((plot0['fpga']*16+plot0['channel']),np.repeat((plot0['mu']+plot0['sigma']).mean(),16),c='r',label=r'$\langle E_i \rangle$ = {}'.format((plot0['mu']+plot0['sigma']).mean()))
            ax[0,0].legend(loc='upper right')
            ax[0,0].set_ylabel(r"$E_i$",fontsize=24)
            ax[0,0].set_xlabel("Channel",fontsize=20)
            ax[0,0].set_title("FPGA 0: Y channels",fontsize=20)
            
            ax02=ax[0,0].twinx()
            ax02.scatter(plot0['fpga']*16+plot0['channel'],plot0['chsq'],c='orange')
            ax02.set_ylabel('Reduced chi-squared',fontsize=20,c='orange')
            
            
            ax[1,0].bar((plot1['fpga']*16+plot1['channel']),plot1['mu']+plot1['sigma'],width=0.8)
            ax[1,0].plot((plot1['fpga']*16+plot1['channel']),np.repeat((plot1['mu']+plot1['sigma']).mean(),16),c='r',label=r'$\langle E_i \rangle$ = {}'.format((plot1['mu']+plot1['sigma']).mean()))
            ax[1,0].legend(loc='upper right')
            ax[1,0].set_ylabel(r"$E_i$",fontsize=24)
            ax[1,0].set_xlabel("Channel",fontsize=20)
            ax[1,0].set_title("FPGA 1: X channels",fontsize=20)

            ax12=ax[1,0].twinx()
            ax12.scatter(plot1['fpga']*16+plot1['channel'],plot1['chsq'],c='orange')
            ax12.set_ylabel('Reduced chi-squared',fontsize=20,c='orange')
            
            musig0=plot0['mu']+plot0['sigma']
            msigplot0=(musig0-musig0.mean())/musig0.mean()
            musig1=plot1['mu']+plot1['sigma']
            msigplot1=(musig1-musig1.mean())/musig1.mean()
            bins=np.linspace(-0.5,0.5,20)
            
            ax[0,1].hist(msigplot0,bins=bins,ec='b')
            ax[0,1].set_ylabel("Frequency",fontsize=20)
            ax[0,1].set_xlabel(r"$(E_i - \langle E_i \rangle)/\langle E_i \rangle$",fontsize=20)
            
            ax[1,1].hist(msigplot1,bins=bins,ec='b')
            ax[1,1].set_ylabel("Frequency",fontsize=20)
            ax[1,1].set_xlabel(r"$(E_i - \langle E_i \rangle)/\langle E_i \rangle$",fontsize=20)
        
        
        else:    
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.bar((plotdf['fpga']*16+plotdf['channel']),plotdf['mu']+plotdf['sigma'],width=0.8)
            ax.plot((plotdf['fpga']*16+plotdf['channel']),np.repeat((plotdf['mu']+plotdf['sigma']).mean(),32),c='r',label=r'Mean $\mu$ + $\sigma$')
            ax.legend(loc='upper right')
            ax.set_ylabel(r"Peak value over pedestal $\mu$ + $\sigma$",fontsize=24)
            ax.set_xlabel("Channel",fontsize=24)

            ax2=ax.twinx()
            ax2.scatter(plotdf['fpga']*16+plotdf['channel'],plotdf['chsq'],c='orange')
            ax2.set_ylabel('Reduced chi-squared',fontsize=24,c='orange')

        plt.show()
        
        
    def chann_correction_factors(self,dfile,coincidence_cut=2,binsize=128,maxval=4096*1.5,bkg=None,ToTcut=20):
        self.gen_df(dfile,binsize,maxval,coincidence_cut,bkg)
        self.refit_mean(dfile)
        df=self.dfs[dfile]
        mu,sig=np.split(df['popt'].explode().values.reshape(-1,3)[:,:2],2,axis=1)
        E_i=mu+sig
        df['factors']=E_i/E_i.mean()
        data=self.data[dfile]
        data['factor']=1
        for i in range(len(df)):
            data.loc[(data[['fpga','channel']]==df.iloc[i][['fpga','channel']].values).all(axis=1),'factor']=df.iloc[i]['factors']
        data['corrected']=data['subtracted']/data['factor']
        self.data[dfile]=data
        factors=df.loc[:,['fpga','channel','factors']]
        factors.reset_index(inplace=True,drop=True)
        self.factors[dfile]=factors
        np.savetxt('corr_factors',factors)

        backg=self.data[bkg]
        backg['factor']=1
        for i in range(len(df)):
            backg.loc[(backg[['fpga','channel']]==df.iloc[i][['fpga','channel']].values).all(axis=1),'factor']=df.iloc[i]['factors']
        backg['corrected']=backg['subtracted']/(backg['factor'])
        self.data[bkg]=backg
        self.gen_df(dfile,binsize,maxval,coincidence_cut,bkg,corrected=True)
        
        self.refit_corr(dfile)


    def refit_corr(self,dfile):
        reg=self.dfs[dfile]
        corr=self.dfs[dfile+"_corrected"]
        def fact(n):
            return np.heaviside(n-1,1)-np.heaviside(-1*(n-1),0)
        ff=fact(reg['factors'].values.astype(float))
        regopt=reg['popt'].explode().values.reshape(-1,3)[:,:2].sum(axis=1).astype(float)
        corropt=corr['popt'].explode().values.reshape(-1,3)[:,:2].sum(axis=1).astype(float)
        refit=ff*regopt<ff*corropt
        refs=corr.reset_index(drop=True)
        
        mu0,sig0=reg.query("fpga==0")['popt'].explode().values.reshape(-1,3)[:,:2].mean(axis=0)
        mu1,sig1=reg.query("fpga==1")['popt'].explode().values.reshape(-1,3)[:,:2].mean(axis=0)
        fits=refs['fit'].values
        opts=refs['popt'].values
        covs=refs['pcov'].values
        residuals=refs['residuals'].values
        chisq=refs['chsq'].values
        
        for ind in refs.index:
            if refit[ind]:
                subhist=refs.loc[ind,'data']
                suberror=refs.loc[ind,'error']
                cent=refs.loc[ind,'centres']
                hist2=np.copy(subhist)
                err2=np.copy(suberror)
                hist2[:np.argmax(subhist)]=1
                err2[:np.argmax(subhist)]=1

                def half_gauss(x,mu,sigma,amp):
                    return np.heaviside(x-mu,1)*amp*np.exp(-1*(x-mu)**2/(2*sigma**2)) + np.heaviside(mu-x,0)        
                def chsq(obv,exp,error):
                    return np.sum((obv-exp)**2/(error)**2)/(len(cent)-3)

                if refs.iloc[ind]['fpga']==0:
                    mu=mu0
                    sig=sig0
                if refs.iloc[ind]['fpga']==1:
                    mu=mu1
                    sig=sig1
                p0=[mu,sig,np.max(subhist)]
                popt,pcov=curve_fit(half_gauss,cent,hist2,p0,sigma=err2)
                fit=half_gauss(cent,popt[0],popt[1],popt[2])
                resid=(hist2-fit)/err2
                i=np.where((corr['fpga'].values==refs.loc[ind,'fpga'])&(corr['channel'].values==refs.loc[ind,'channel']))[0][0]
                fits[i]=fit
                covs[i]=pcov
                opts[i]=popt
                residuals[i]=resid
                chisq[i]=chsq(hist2,fit,err2)
            
        corr['fit']=fits
        corr['popt']=opts
        corr['pcov']=covs
        corr['residuals']=residuals
        corr['chsq']=chisq

        self.dfs[dfile+'_corrected']=corr


    def corrected_gauss_fits(self,dfile):
        df=self.dfs[dfile]
        corr_df=self.dfs[dfile+"_corrected"]
        fig,ax=plt.subplots(2,2,gridspec_kw={'height_ratios':[3,1]})
        self.curr_pos=0
        f=df.iloc[self.curr_pos]
        g=corr_df.iloc[self.curr_pos]
        
        def key_event(e):
            if e.key=="right":
                self.curr_pos+=1
                self.curr_pos%=len(df)
                f=df.iloc[self.curr_pos]
                g=corr_df.iloc[self.curr_pos]
                
                ax[0,0].cla()
                ax[0,1].cla()
                ax[1,0].cla()
                ax[1,1].cla()
                
                fig.suptitle("FPGA {}, channel {}".format(f['fpga'],f['channel']),fontsize=24)
                
                ax[0,0].bar(f['centres'],f['data'],width=f['space'],alpha=0.3,label='Data')
                ax[0,0].plot(f['centres'],f['fit'],c='r',label='Fit')
                ax[0,0].set_xlabel('Peak value over pedestal',fontsize=20)
                ax[0,0].set_ylabel('Frequency',fontsize=20)
                ax[0,0].set_xlim(0,6000)
                ax[0,0].plot(np.repeat(int(f['popt'][:2].sum()),100),np.linspace(0,np.nanmax(f['data']),100),label=r'Fit $\mu$ + $\sigma$')
                ax[0,0].legend(loc='upper right')
                ax[0,0].text(int(f['popt'][:2].sum())+200,int(np.nanmax(f['fit'])),r"$\mu$ + $\sigma%$ = "+str(int(f['popt'][:2].sum())))
                ax[0,0].set_title("Original data")
                
                ax[0,1].bar(g['centres'],g['data'],width=g['space'],alpha=0.3,label='Data')
                ax[0,1].plot(g['centres'],g['fit'],c='r',label='Fit')
                ax[0,1].set_xlabel('Peak value over pedestal',fontsize=20)
                ax[0,1].set_ylabel('Frequency',fontsize=20)
                ax[0,1].set_xlim(0,6000)
                ax[0,1].plot(np.repeat(int(g['popt'][:2].sum()),100),np.linspace(0,np.nanmax(g['data']),100),label=r'Fit $\mu$ + $\sigma$')
                ax[0,1].legend(loc='upper right')
                ax[0,1].text(int(g['popt'][:2].sum())+200,int(np.nanmax(g['fit'])),r"$\mu$ + $\sigma%$ = "+str(int(g['popt'][:2].sum())))
                ax[0,1].set_title("Corrected data")
                
                ax[1,0].scatter(f['centres'],f['residuals'],marker='+')
                ax[1,0].set_ylabel(r'Residuals \ $\sigma$',fontsize=20)
                ax[1,0].set_xlim(0,6000)
                ma=np.nanmax(np.abs(f['residuals']))+1
                ax[1,0].set_ylim(-1*ma,ma)
                ax[1,0].fill_between((0,6000),(-1,-1),(1,1),alpha=0.3,label=r'1 $\sigma$')
                ax[1,0].legend(loc='upper right')
                
                ax[1,1].scatter(g['centres'],g['residuals'],marker='+')
                ax[1,1].set_ylabel(r'Residuals \ $\sigma$',fontsize=20)
                ax[1,1].set_xlim(0,6000)
                ma=np.nanmax(np.abs(g['residuals']))+1
                ax[1,1].set_ylim(-1*ma,ma)
                ax[1,1].fill_between((0,6000),(-1,-1),(1,1),alpha=0.3,label=r'1 $\sigma$')
                ax[1,1].legend(loc='upper right')
                
                fig.canvas.draw()
                
            elif e.key=="left":
                self.curr_pos-=1
                self.curr_pos%=len(df)
                f=df.iloc[self.curr_pos]
                g=corr_df.iloc[self.curr_pos]
                
                ax[0,0].cla()
                ax[0,1].cla()
                ax[1,0].cla()
                ax[1,1].cla()
                
                fig.suptitle("FPGA {}, channel {}".format(f['fpga'],f['channel']),fontsize=24)
                
                ax[0,0].bar(f['centres'],f['data'],width=f['space'],alpha=0.3,label='Data')
                ax[0,0].plot(f['centres'],f['fit'],c='r',label='Fit')
                ax[0,0].set_xlabel('Peak value over pedestal',fontsize=20)
                ax[0,0].set_ylabel('Frequency',fontsize=20)
                ax[0,0].set_xlim(0,6000)
                ax[0,0].plot(np.repeat(int(f['popt'][:2].sum()),100),np.linspace(0,np.nanmax(f['data']),100),label=r'Fit $\mu$ + $\sigma$')
                ax[0,0].legend(loc='upper right')
                ax[0,0].text(int(f['popt'][:2].sum())+200,int(np.nanmax(f['fit'])),r"$\mu$ + $\sigma%$ = "+str(int(f['popt'][:2].sum())))
                ax[0,0].set_title("Original data")
                
                ax[0,1].bar(g['centres'],g['data'],width=g['space'],alpha=0.3,label='Data')
                ax[0,1].plot(g['centres'],g['fit'],c='r',label='Fit')
                ax[0,1].set_xlabel('Peak value over pedestal',fontsize=20)
                ax[0,1].set_ylabel('Frequency',fontsize=20)
                ax[0,1].set_xlim(0,6000)
                ax[0,1].plot(np.repeat(int(g['popt'][:2].sum()),100),np.linspace(0,np.nanmax(g['data']),100),label=r'Fit $\mu$ + $\sigma$')
                ax[0,1].legend(loc='upper right')
                ax[0,1].text(int(g['popt'][:2].sum())+200,int(np.nanmax(g['fit'])),r"$\mu$ + $\sigma%$ = "+str(int(g['popt'][:2].sum())))
                ax[0,1].set_title("Original data")
                
                ax[1,0].scatter(f['centres'],f['residuals'],marker='+')
                ax[1,0].set_ylabel(r'Residuals \ $\sigma$',fontsize=20)
                ax[1,0].set_xlim(0,6000)
                ma=np.nanmax(np.abs(f['residuals']))+1
                ax[1,0].set_ylim(-1*ma,ma)
                ax[1,0].fill_between((0,6000),(-1,-1),(1,1),alpha=0.3,label=r'1 $\sigma$')
                ax[1,0].legend(loc='upper right')
                
                ax[1,1].scatter(g['centres'],g['residuals'],marker='+')
                ax[1,1].set_ylabel(r'Residuals \ $\sigma$',fontsize=20)
                ax[1,1].set_xlim(0,6000)
                ma=np.nanmax(np.abs(g['residuals']))+1
                ax[1,1].set_ylim(-1*ma,ma)
                ax[1,1].fill_between((0,6000),(-1,-1),(1,1),alpha=0.3,label=r'1 $\sigma$')
                ax[1,1].legend(loc='upper right')
                
                fig.canvas.draw()
                
        
        fig.canvas.mpl_connect('key_press_event',key_event)
                
                
        fig.suptitle("FPGA {}, channel {}".format(f['fpga'],f['channel']),fontsize=24)
                
        ax[0,0].bar(f['centres'],f['data'],width=f['space'],alpha=0.3,label='Data')
        ax[0,0].plot(f['centres'],f['fit'],c='r',label='Fit')
        ax[0,0].set_xlabel('Peak value over pedestal',fontsize=20)
        ax[0,0].set_ylabel('Frequency',fontsize=20)
        ax[0,0].set_xlim(0,6000)
        ax[0,0].plot(np.repeat(int(f['popt'][:2].sum()),100),np.linspace(0,np.nanmax(f['data']),100),label=r'Fit $\mu$ + $\sigma$')
        ax[0,0].legend(loc='upper right')
        ax[0,0].text(int(f['popt'][:2].sum())+200,int(np.nanmax(f['fit'])),r"$\mu$ + $\sigma%$ = "+str(int(f['popt'][:2].sum())))
        ax[0,0].set_title("Original data")
                
        ax[0,1].bar(g['centres'],g['data'],width=g['space'],alpha=0.3,label='Data')
        ax[0,1].plot(g['centres'],g['fit'],c='r',label='Fit')
        ax[0,1].set_xlabel('Peak value over pedestal',fontsize=20)
        ax[0,1].set_ylabel('Frequency',fontsize=20)
        ax[0,1].set_xlim(0,6000)
        ax[0,1].plot(np.repeat(int(g['popt'][:2].sum()),100),np.linspace(0,np.nanmax(g['data']),100),label=r'Fit $\mu$ + $\sigma$')
        ax[0,1].legend(loc='upper right')
        ax[0,1].text(int(g['popt'][:2].sum())+200,int(np.nanmax(g['fit'])),r"$\mu$ + $\sigma%$ = "+str(int(g['popt'][:2].sum())))
        ax[0,1].set_title("Original data")
                
        ax[1,0].scatter(f['centres'],f['residuals'],marker='+')
        ax[1,0].set_ylabel(r'Residuals \ $\sigma$',fontsize=20)
        ax[1,0].set_xlim(0,6000)
        ma=np.nanmax(np.abs(f['residuals']))+1
        ax[1,0].set_ylim(-1*ma,ma)
        ax[1,0].fill_between((0,6000),(-1,-1),(1,1),alpha=0.3,label=r'1 $\sigma$')
        ax[1,0].legend(loc='upper right')
                
        ax[1,1].scatter(g['centres'],g['residuals'],marker='+')
        ax[1,1].set_ylabel(r'Residuals \ $\sigma$',fontsize=20)
        ax[1,1].set_xlim(0,6000)
        ma=np.nanmax(np.abs(g['residuals']))+1
        ax[1,1].set_ylim(-1*ma,ma)
        ax[1,1].fill_between((0,6000),(-1,-1),(1,1),alpha=0.3,label=r'1 $\sigma$')
        ax[1,1].legend(loc='upper right')
                
        plt.show()


    def ReconstructCubes(self,datafile):
        events=self.data[datafile]
        # Find all events of coincidence 3
        threes=np.unique(events.index.array[np.where(events['coincidence']==3)[0]])

        # Get first index of each set of 3 coincident events
        StartIndex=np.where(np.isin(events.index.array,threes))[0][::3]

        # When dealing with 2 and 3 coincidence events, need to make sure that the coincident fibres 
        # are orthogonal for cube reconstruction. These snippets of code find any 'bad' coincident events 
        # and exclude them from cube reconstruction.

        # First, find all events of coincidence = 3 & identify the direction with the most hits in each event
        xy_threes=np.array(events.loc[threes][['x','y']]).reshape([-1,3,2])
        modvals,modes=mode(xy_threes,axis=1)[0][:,0,:],mode(xy_threes,axis=1)[1][:,0,:]
        for i in range(modes.shape[0]):
            if (modes[i]==np.array([2,2])).all():
                modes[i][np.where(modvals[i]!=-1)[0]]=1
            elif (modes[i]==np.array([3,3])).all():
                modes[i][np.where(modvals[i]!=-1)[0]]=1
        duplicate_dir=np.argmax(modes,axis=1)

        duplicate_col_vals=np.zeros([xy_threes.shape[0],3])
        for i in range(xy_threes.shape[0]):
            duplicate_col_vals[i] = xy_threes[i,:,duplicate_dir[i]]

        # Remove events where all 3 events are in the same fibre direction i.e. all x or all y.
        begone=np.where((duplicate_col_vals==-1).all(axis=1))[0]
        events.loc[threes[begone],'inWindow']=False
        
        duplicate_col_vals=np.delete(duplicate_col_vals,begone,axis=0).reshape([-1,3])
        StartIndex_pruned=np.delete(StartIndex,begone,axis=0)

        # Find indices of 2 hits in the same direction in each event
        Starts_to_compare=np.concatenate([StartIndex_pruned,StartIndex_pruned])
        Starts_to_compare.sort()
        Starts_to_compare+=np.where(duplicate_col_vals==-1)[1]

        # Find which hit of the 2 needs to be disregarded in reconstruction, by removing 
        # the one with the lowest totValue.
        three_to_go=np.argmin(np.array(events['totValue'].iloc[Starts_to_compare]).reshape([-1,2]),axis=1)
        Starts_to_compare=Starts_to_compare.reshape([-1,2])
        to_go=np.zeros(Starts_to_compare.shape[0],dtype=int)

        for i in range(Starts_to_compare.shape[0]):
            to_go[i]=Starts_to_compare[i,three_to_go[i]]
        
        # Find all events of coincidence = 2        
        twos=np.where(events['coincidence']==2)[0]
        twos_index=np.unique(events.index.array[twos])

        # Find all coincidence 2 events where both signals are in the same direction
        xminus1=(np.array(events.loc[twos_index,'x']).reshape([-1,2])==-1).all(axis=1)
        yminus1=(np.array(events.loc[twos_index,'y']).reshape([-1,2])==-1).all(axis=1)

        # Find indices of all hits in events with both signals in the same direction 
        # & identify them to remove
        twos_check=[None]*(xminus1.shape[0]+yminus1.shape[0])
        twos_check[::2]=xminus1+yminus1
        twos_check[1::2]=xminus1+yminus1
        bad_twos=twos[np.where(twos_check)[0]]
        events.loc[np.unique(events.index.array[bad_twos]),'inWindow']=False

        # Create a boolean array to identify events to calculate cube coordinates for
        check=np.zeros(len(events),dtype=bool)
        check[np.concatenate([StartIndex,StartIndex+1,StartIndex+2])]=True

        go=np.concatenate([StartIndex[begone],StartIndex[begone]+1,StartIndex[begone]+2])
        go.sort()

        # Remove all coincidence 3 events where all hits are in the same direction 
        check[go]=False
        # Remove the smallest totValue hit in coincidence 3 events with 2 hits in the same direction
        check[to_go]=False
        
        # Remove coincidence 2 events with all hits in the same direction
        check[np.where(events['coincidence']==2)]=True
        check[bad_twos]=False

        events['Combine']=check

        events.loc[events['Combine']]

        events['CubeX']=-1
        events['CubeY']=-1
        events['CubeZ']=-1

        comb=events.loc[events['Combine']]
        # Adding 1 to both the x and y events as for each pair of coincident events, 
        # one is in an x-fibre and one is in a y-fibre and x/y takes a value of -1 
        # when it is in the opposite plane of fibres - e.g. if for one event x = 3, 
        # for the y fibre event x = -1 and so need to add 1 to offset this when summing. 
        # Kept summing as it will be faster than some max method
        events.loc[events['Combine'],'CubeX']=comb['x'][::2]+comb['x'][1::2]+1
        events.loc[events['Combine'],'CubeY']=comb['y'][::2]+comb['y'][1::2]+1
        events.loc[events['Combine'],'CubeZ']=(comb['z'][::2]+comb['z'][1::2])/2

        events['CubeZ']=events['CubeZ'].astype(int)
        comb=events.loc[events['Combine']]

        # Compensates for left-handed coordinate system on old detector
        events['CubeX']=events['CubeX']*-1 + 3
        
        events.loc[events['Combine'],'ZCubeID']=comb['CubeX'] + 4*comb['CubeY'] + 16*comb['CubeZ']
        events.loc[events['Combine'],'XCubeID']=comb['CubeY'] + 4*comb['CubeZ'] + 16*comb['CubeX']
        events.loc[events['Combine'],'YCubeID']=comb['CubeZ'] + 4*comb['CubeX'] + 16*comb['CubeY']
        events['XCubeID'].fillna(-1,inplace=True)
        events['YCubeID'].fillna(-1,inplace=True)
        events['ZCubeID'].fillna(-1,inplace=True)
        events['XCubeID']=events['XCubeID'].astype(int)
        events['YCubeID']=events['YCubeID'].astype(int)
        events['ZCubeID']=events['ZCubeID'].astype(int)
        
        events['EventNum'] = (~events['inWindow']).cumsum()
        
        events = events.set_index('EventNum')
        self.reconstructed[datafile]=events[events['Combine']]


    def ThresholdEfficiency(self,datafile,compton_E_i,PA_E=12,ADCthresh=200):
        df=self.dfs[datafile]
        df.reset_index(inplace=True,drop=True)
        musig=pd.DataFrame(df['popt'].explode().values.reshape(-1,3)[:,:2].sum(axis=1),columns=['mu + sigma'])
        calib=pd.concat([df.loc[:,['fpga','channel']],musig],axis=1)
        Co60_E=963
        
        calib['ADC/keV'] = calib['mu + sigma']/Co60_E
        calib['ADC/PA'] = calib['ADC/keV']*PA_E        
        calib['PA threshold'] = ADCthresh/calib['ADC/PA']
        PA_i = compton_E_i/PA_E

        convs={}
        for i in range(len(calib)):
            convs['fpga {}, channel{}'.format(calib.loc[i,'fpga'],calib.loc[i,'channel'])]=calib['ADC/keV'].values[i]

        self.channel_convs[datafile]=convs

        def poisson(x,lamb):
            try:
                return (lamb**x*np.exp(-1*lamb))/math.factorial(x)
            except ValueError:
                print("x should be an integer")

        def det_efficiency(lamb,threshold):
            val1,val2=0,0
            for i in range(math.floor(threshold)):
                val1+=poisson(i,lamb)
            for i in range(math.ceil(threshold)):
                val2+=poisson(i,lamb)
            return 1 - val1, 1 - val2

        f=uproot.open(datafile)
        settings=f['settings'].pandas.df().loc[:,['fpgaId','chId','x','y','z']]
        ze=np.zeros([4,4,4])
        cubecoords=np.argwhere(ze==0)

        cubedict={}
        for coord in cubecoords:
            channels=[settings.query('x==@coord[0]&z==@coord[2]').loc[:,['fpgaId','chId']].values,
                      settings.query('y==@coord[1]&z==@coord[2]').loc[:,['fpgaId','chId']].values]
            
            thresh1=calib.query('fpga==@channels[0][0][0]&channel==@channels[0][0][1]')['PA threshold'].values[0]
            thresh2=calib.query('fpga==@channels[1][0][0]&channel==@channels[1][0][1]')['PA threshold'].values[0]

            eff1=det_efficiency(PA_i,thresh1)
            eff2=det_efficiency(PA_i,thresh2)

            cubedict[str(coord)]=[eff1,eff2]
        self.cubedicts[datafile] = cubedict