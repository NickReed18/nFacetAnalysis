import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
from scipy.stats import mode
import matplotlib.colors as mcolor
import os
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import itertools
import ndap
import scipy.stats as sps

class Plotter:
    """
    Wrapper class for producing plots from data run analysis. Can either open root files directly 
    and convert them to the desired h5 format, or can work with already processed files. 
    If the conversion is done, the processed files are saved.

    ATTRIBUTES
    ------------------------
    data : dict
        Dictionary of datafiles that have been added to the Plotter.

    coincidence_window : int
        Timing window for calculating event coincidences, if converting from a root file.

    METHODS
    ------------------------
    add_data(self,folder,filename,name,cut=True,cut_thresh=350,raw=False)
        Adds a data file to the Plotter. If the passed data file is a root file, 
        performs coincidence finding analysis and saves the resulting DataFrame as a .h5 file.
        If the file has already been processed into a .h5 file in the same folder, load that file
        if raw is False.

    get_rate(self,dfile,bkg=None)
        Calculates the rate of neutrons in the detector. 

    color_maps(self,df,background,projection)
        Plots 2D planes of cube triggers with background subtracted, for a specified direction (x, y or z).

    projection(self,df,background,axis,source,nrot=0,background=True)
        Plots the projection of the 3D detector onto a given 2D plane, along an axis of choice. 

    benchmark(self,data,ref,axis,source,refname,refrot=True,databg=None,refbg=None)
        Plots the distribution of events across planes in a chosen direction, 
        for a source and a reference. Background subtraction is also supported.

    CubeIDplot(self,data,bg,source)
        Plots a cube count rate by cube ID, and a background subtracted cube count rate.

    all_color_maps(self,df,source,background=None)
        Plots all 12 2D plane hitmaps, with optional background subtraction.

    benchall(self,data,bg,ref,refbg,cddat,chpbdat,axis,source,bgsource,refrot=True)
        Plots the background-subtracted distribution of events across planes in a chosen direciton,
        for a source with different shieldings (bare, Cd, and CH2+Pb) and a reference.

    PIDplot(self,data,source)
        Plots the particle identification plot for the given dataset, of signal amplitude 
        against time over threshold. 

    peakValueplot(self,data,source,n_bins,background=None)
        Plots a histogram of the waveform peak value for the dataset. If background is not None, 
        performs background subtraction.

    fit_projection(self,data,source,fitconst,bkg=None,plot=False)
        Calculates the 1D distributions of count rate across planes in the X, Y and Z directions.
        Also fits an exponential to any identified as decaying, and returns the lambda 
        parameter of each exponential. Plots these distributions & exponentials if plot is True.

    comp_proj_fits(self,ds,bkgs,sources,fitconst)
        Calculates plane projection exponential fit parameters for a list of datasets, and plots
        a bar chart comparing them, alongside some reference sources.

    View_3D(self,data,source,bkg=None)
        Plots the 3D view of the dataset provided, with optional background subtraction.

    cmaps(self,data,source,bkg=None)
        Plots the projected 2D hitmaps, summed along each of the X, Y and Z axes, with 
        optional background subtraction.

    fit1(self,counts)
        Fits direction to source from a list of counts in each cube, ordered by increasing x, 
        then y, then z. Returns the vertical angle, the xy plane angle, and the covariance of 
        fitted parameters.

    single_dir_fit(self,data,bkg=None,nfits=10)
        Calculates a direction fit with a standard deviation by taking the data as a Poisson mean 
        and drawing from that Poisson distribution, to calculate a standard error. Can optionally
        include background subtraction. Returns list of the two fitted angle results for each fit.

    plot_directions(self,ds,bkgs,nfits,colors,labels,fillcols,markers)
        Calls the single_dir_fit function to fit directions to a list of datasets, and then plots them
        all on one polar plot, with a shaded region indicated 3 standard deviations.

    frac_fit(self,data,npoints,bkg=None,nfits=10)
        Calculates a direction fit on a fraction of a given dataset, where the number of neutrons taken
        for the fit is given by npoints. Returns fitted xy plane and vertical angles, their means, standard
        deviations, upper and lower bounds.

    err_develop(self,ds,bkgs,nfits,colors,labels,fillcols)
        Calculates fractional fits using the frac_fit function for a number of neutrons ranging from 10 
        to 100,000, incrementing in powers of 10. The error region for each fit is plotted, and the final fit
        point is also indicated.  

    face_counts(self,data,bkg=None,plot=False)
        Calculates the counts on each face and in the core 8 cubes. If plot is True, also plot these 
        counts in a bar chart.
    """

    
    def __init__(self,coincidence_window=100):
        """
        Initialises the data dictionaries and defines the coincidence window for data processing.

        PARAMETERS
        ------------------------
        coincidence_window : int
            The timing window for determining coincident events. Default is 100 samples.
        """
        self.data={}
        self.coincidence_window=coincidence_window

    def add_data(self,folder,filename,name,cut=True,cut_thresh=350,raw=False):
        """
        Takes in a data file of either .root or .h5 format and adds them to a dictionary of data files.
        If a root file is taken in, calculates coincidences and reconstructs cubes from all events with 2 or 3 coincidences.
        This reconstructed dataframe is also saved as a .h5 file.

        PARAMETERS
        ------------------------
        folder : str
            The folder to look for data in.

        filename : str
            Name of the data file to load.

        name : str
            Name to reference the data file as.
        """
        # Check if provided file is of the right type, check if the file is a .h5 file, 
        # check if file has already been converted to a .h5 file
        if filename[-5:]!=".root" and filename[-3:]!=".h5":
            raise TypeError("Incorrect file type; please make sure you are passing a root file or a h5 file.")
        
        if filename[-3:]==".h5":
            if os.path.isfile(folder + filename):
                self.data[name]=pd.read_hdf(folder+filename)
                return

        if os.path.isfile(folder + filename[:-5]+".h5"):
            if not name in self.data:
                if not raw:
                    self.data[name]=pd.read_hdf(folder+filename[:-5]+".h5")
                    return

        # Load events and calculate the peak value over pedestal for each event
        f=uproot.open('/home/nr1315/Documents/Project/NPL_2021/vdg_1um_Li7target19A_433cm_0deg.root')

        events=f['events'].pandas.df(flatten=False)
        settings=f['settings'].pandas.df(flatten=False)
        events['x'],events['y'],events['z'],events['subtracted']=-2,-2,-2,0

        events['subtracted']=0
        for chann in range(len(settings)):
            events.loc[(events[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'subtracted']=events.loc[(events[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'peakValue']-settings.loc[chann,'pedestal']
            events.loc[(events[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'x']=settings.loc[chann,'x']  
            events.loc[(events[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'y']=settings.loc[chann,'y']  
            events.loc[(events[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'z']=settings.loc[chann,'z']  
            events.loc[(events[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'subtracted']=events.loc[(events[['fpga','channel']]==settings.loc[chann,['fpgaId','chId']].values).all(axis=1),'peakValue']-settings.loc[chann,'pedestal']

        if cut:
            events = events.loc[events['subtracted']>cut_thresh]

        if len(events)==0:
            print("No events remaining")
            return


        events.sort_values(['time'],inplace=True)
        events['deltaT'] = events['time'].diff()

        events['inWindow'] = events['deltaT'] < 100
        events['EventNum'] = (~events['inWindow']).cumsum()

        events = events.set_index('EventNum')


        # Groups events by coincidence count
        events['coincidence']=events.groupby('EventNum')['totValue'].count()
        events['CubeX'],events['CubeY'],events['CubeZ']=-1,-1,-1


        # ----------------------------------------------
        # COINCIDENCE 3 EVENT RECONSTRUCTION BEGINS HERE
        # ----------------------------------------------

        nunique=events.query('coincidence==3').groupby('EventNum')['fpga'].nunique()



        # Ignores any events of coincidence 3 that have all 3 triggers on the same FPGA
        events.loc[(events['coincidence']==3).index].loc[nunique.index[np.where(nunique==1)]]['inWindow']=False
        events.loc[(events['coincidence']==3).index].loc[nunique.index[np.where(nunique==1)]]['coincidence']=1

        # Looks at all remaining coincidence 3 events
        coinc3=events.query('coincidence==3').loc[nunique.index[np.where(nunique>1)]]
        mod=coinc3.groupby('EventNum')['fpga'].agg(pd.Series.mode)
        invmod = np.heaviside(-1*mod,1).astype(int)

        # Identifies and ignores coincidence 3 events where the individual triggers do not have matching z coordinates, i.e. 
        # the signal on the FPGA with 1 count has a different Z coordinate to either trigger on the other FPGA, 
        # as this cannot be reconstructed to a single cube
        bad_pair = coinc3.index.unique()[np.where(((coinc3.query('fpga==@mod')['z'][::2]==coinc3.query('fpga==@invmod')['z'])!=(coinc3.query('fpga==@mod')['z'][1::2]==coinc3.query('fpga==@invmod')['z'])))[0]]
        events.loc[bad_pair,'inWindow'] = False
        events.loc[bad_pair,'coincidence'] = 1

        
        nunique=events.query('coincidence==3').groupby('EventNum')['fpga'].nunique()
        coinc3=events.query('coincidence==3').loc[nunique.index[np.where(nunique>1)]]
        mod=coinc3.groupby('EventNum')['fpga'].agg(pd.Series.mode)
        invmod = np.heaviside(-1*mod,1).astype(int)


        # Finds all coincidence 3 events where only 1 trigger on the FPGA with 2 triggers has the correct z coordinate to pair
        # with the trigger on the other FPGA, and reconstructs a cube from these two triggers.
        single_pair = coinc3.index.unique()[np.where(((coinc3.query('fpga==@mod')['z'][::2]==coinc3.query('fpga==@invmod')['z'])!=(coinc3.query('fpga==@mod')['z'][1::2]==coinc3.query('fpga==@invmod')['z'])))[0]]
        spair_df = coinc3.loc[single_pair]


        zs = coinc3.loc[single_pair].groupby(['EventNum'])['z'].agg(pd.Series.mode)

        spair=coinc3.loc[single_pair]

        spair.loc[spair.eval('z==@zs'),'CubeX'] = (spair.loc[spair.eval('z==@zs'),'x'][::2] + spair.loc[spair.eval('z==@zs'),'x'][1::2] +1)
        spair.loc[spair.eval('z==@zs'),'CubeY'] = (spair.loc[spair.eval('z==@zs'),'y'][::2] + spair.loc[spair.eval('z==@zs'),'y'][1::2] +1)
        spair.loc[spair.eval('z==@zs'),'CubeZ'] = zs

        coinc3.loc[single_pair,['CubeX','CubeY','CubeZ']]=spair.loc[:,['CubeX','CubeY','CubeZ']]




        # Handles coincidence 3 events where either of the two triggers on one FPGA could be viably paired with the trigger
        # on the other FPGA. First selects from the two based on the highest ToT value, and then on the highest peak over 
        # pedestal value if the ToT values are the same. 
        ambiguous_pair = coinc3.index.unique()[np.where(((coinc3.query('fpga==@mod')['z'][::2]==coinc3.query('fpga==@invmod')['z'])&(coinc3.query('fpga==@mod')['z'][1::2]==coinc3.query('fpga==@invmod')['z'])))[0]]
        ambi_pair = events.loc[ambiguous_pair]
        mod_ambiguous=mod.loc[ambiguous_pair]
        inv_mod_ambiguous=np.heaviside(-1*mod_ambiguous,1).astype(int)
        tot_max = ambi_pair.query('fpga==@mod_ambiguous').groupby('EventNum')['totValue'].max()
        sub_max = ambi_pair.query('fpga==@mod_ambiguous').groupby('EventNum')['subtracted'].max()
        ambi_c = ((ambi_pair.eval('fpga==@mod_ambiguous'))&ambi_pair.eval('totValue==@tot_max'))
        ambi_off = (ambi_pair.eval('fpga==@inv_mod_ambiguous'))
        sums = (ambi_c|ambi_off).groupby('EventNum').sum()
        ambiguous_combine = ((ambi_pair.eval('(fpga==@mod_ambiguous)')&ambi_pair.eval('totValue==@tot_max')&(sums==2))|(ambi_pair.eval('fpga==@inv_mod_ambiguous')))|(((sums!=2)&ambi_pair.eval('fpga==@mod_ambiguous')&ambi_pair.eval('subtracted==@sub_max')))





        ambi_pair.loc[ambiguous_combine,'CubeX']=ambi_pair.loc[ambiguous_combine,'x'][::2]+ambi_pair.loc[ambiguous_combine,'x'][1::2] + 1
        ambi_pair.loc[ambiguous_combine,'CubeY']=ambi_pair.loc[ambiguous_combine,'y'][::2]+ambi_pair.loc[ambiguous_combine,'y'][1::2] + 1
        ambi_pair.loc[ambiguous_combine,'CubeZ']=(ambi_pair.loc[ambiguous_combine,'z'][::2]+ambi_pair.loc[ambiguous_combine,'z'][1::2])/2
        ambi_pair.loc[:,'CubeZ']=ambi_pair.loc[:,'CubeZ'].astype(int)
        coinc3.loc[ambiguous_pair,['CubeX','CubeY','CubeZ']]=ambi_pair.loc[:,['CubeX','CubeY','CubeZ']]

        events.loc[coinc3.index.unique(),['CubeX','CubeY','CubeZ']]=coinc3.loc[:,['CubeX','CubeY','CubeZ']]

        # ----------------------------------------------
        # COINCIDENCE 2 EVENT RECONSTRUCTION BEGINS HERE
        # ----------------------------------------------

        nunique_2 = events.query('coincidence==2').groupby('EventNum')['fpga'].nunique()

        coinc2=events.query('coincidence==2')

        z_unique = events.query('coincidence==2').groupby('EventNum')['z'].nunique()

        # Ignores all coincidence 2 events where both signals are in different Z coordinates

        events.loc[coinc2.index.unique()[np.where(nunique_2!=2)[0]],'inWindow']=False
        events.loc[coinc2.index.unique()[np.where(z_unique!=1)],'inWindow']=False
        events.loc[coinc2.index.unique()[np.where(nunique_2!=2)[0]],'coincidence']=1
        events.loc[coinc2.index.unique()[np.where(z_unique!=1)],'coincidence']=1


        # Remaining valid coincidence 2 events are reconstructed.
        coinc2=events.query('coincidence==2')
        coinc2.loc[:,'CubeX'] = coinc2.loc[:,'x'][::2] + coinc2.loc[:,'x'][1::2] + 1
        coinc2.loc[:,'CubeY'] = coinc2.loc[:,'y'][::2] + coinc2.loc[:,'y'][1::2] + 1
        coinc2.loc[:,'CubeZ'] = (coinc2.loc[:,'z'][::2] + coinc2.loc[:,'z'][1::2])/2

        events.loc[coinc2.index.unique(),['CubeX','CubeY','CubeZ']] = coinc2.loc[:,['CubeX','CubeY','CubeZ']]

        # Need to reverse X direction of existing data from old system, as coordinate system definition was not
        # consistent with simulation (and was a left-handed coordinate system as opposed to a right-handed one.)
        # If the mapping order is changed for the new system in the DAQ, then the lines modifying CubeX here will 
        # not be necessary.
        events['CubeX'] = events['CubeX']*-1 + 3
        events.loc[events['CubeX']==4,'CubeX']=-1
        
        events['CubeZ'] = events['CubeZ'].astype(int)

        # Calculate Cube ID values, in order of increasing X, then Y, then Z.
        events['ZCubeID'] = events['CubeX'] + 4*events['CubeY'] + 16*events['CubeZ']

        # Set all events without a reconstructed cube to have a cube ID of -1.
        events.loc[events['ZCubeID']<0,'ZCubeID'] = -1

        # Recalculate total number of events after accidental events were dropped.
        events['EventNum'] = (~events['inWindow']).cumsum()
        events.set_index('EventNum',inplace=True,drop=True)

        events['tot']=events.groupby(events.index)['totValue'].max()
        events['peak_amplitude']=events.groupby(events.index)['peakValue'].max()

        events.to_hdf(folder+filename[:-5]+'.h5',key='events')

        self.data[name]=events
        return
        
    def get_rate(self,dfile,bkg=None):
        """
        Calculates the measured neutron and gamma rates, allowing for background subtraction.

        PARAMETERS
        ------------------------
        dfile : str
            The label of the desired datafile, as saved in self.data.
        
        bkg : str or NoneType
            The label of the desired background datafile, as saved in self.data.

        RETURNS
        ------------------------
        everate : float
            The calculated neutron rate.

        grate : float
            The calculated gamma rate.
        """
        events=self.data[dfile]
        # Number of unique indices in DataFrame where 2 <= coincidence < 4 is equal to # of 
        # neutrons (as each coincident event is one neutron)
        # 1e8 is count rate of system (100 MHz), time is in system counts 
        everate=len(np.unique(events.query('coincidence>=2&coincidence<4&totValue>60').index))/((events['time'].max()-events['time'].min())/1e8)
        grate=len(events.query('coincidence==1&totValue<60'))/((events['time'].max()-events['time'].min())/1e8)
        if bkg is not None:
            bg=self.data[bkg]
            brate=len(np.unique(bg.query('coincidence>=2&coincidence<4&totValue>60').index))/((bg['time'].max()-bg['time'].min())/1e8)
            bgrate=len(bg.query('coincidence==1&totValue<60'))/((bg['time'].max()-bg['time'].min())/1e8)
            return everate-brate,grate-bgrate
        else:
            return everate,grate
    
    def color_maps(self,df,projection,background=None):
        """
        Plots 2D planes of hitmaps of cube counts. There are 4 planes for 
        each direction and the user's choice of direction is plotted.

        PARAMETERS
        ------------------------
        df : str
            Name of dataset stored in self.data for the source data.

        background : str
            Name of background dataset stored in self.data.

        projection : int
            Index of which planes to plot. Choose 0, 1 or 2 for x, y or z respectively.

        RAISES
        ------------------------
        ValueError
            The projection parameter is required to be 0, 1 or 2. ValueError is raised if a different value is given.
        """
        if projection!=0 and projection!=1 and projection!=2:
            raise ValueError("Invalid projection, please choose a valid axis (0, 1 or 2).")

        data_arr=np.zeros([4,4,4])
        data=self.data[df]

        if background is not None:
            bg=self.data[background]
            bkg_arr=np.zeros([4,4,4])
            bg_norm=(bg['time'].max()-bg['time'].min())/1e8
            dat_norm=(data['time'].max() - data['time'].min())/1e8

        for k in range(4):
            for j in range(4):
                for i in range(4):
                    data_arr[i,j,k]=data.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
                    if background is not None:
                        bkg_arr[i,j,k]=bg.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
        
        

        
        data_arr/=2
        # Background subtraction, if background dataset is provided
        if background is not None:
            bkg_arr/=2
            norm = bg_norm/dat_norm
            if norm < 1:
                plot_arr=data_arr*norm - bkg_arr
            else:
                plot_arr=data_arr - bkg_arr/norm   
        else:
            plot_arr=data_arr     


        vmin=plot_arr.min()
        vmax=plot_arr.max()
        
        fig,ax=plt.subplots(2,2,figsize=(20,20))
        if projection==0:
            fig.suptitle('X planes', fontsize=24)

            fig0=ax[0,0].imshow(plot_arr[0,:,:].T,interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[0,0].set_xlabel('CubeY',fontsize=18)
            ax[0,0].set_ylabel('CubeZ',fontsize=18)
            ax[0,0].set_title('CubeX = 0',fontsize=20)

            fig1=ax[0,1].imshow(plot_arr[1,:,:].T,interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[0,1].set_xlabel('CubeY',fontsize=18)
            ax[0,1].set_ylabel('CubeZ',fontsize=18)
            ax[0,1].set_title('CubeX = 1',fontsize=20)


            fig2=ax[1,0].imshow(plot_arr[2,:,:].T,interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[1,0].set_xlabel('CubeY',fontsize=18)
            ax[1,0].set_ylabel('CubeZ',fontsize=18)
            ax[1,0].set_title('CubeX = 2',fontsize=20)


            fig3=ax[1,1].imshow(plot_arr[3,:,:].T,interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[1,1].set_xlabel('CubeY',fontsize=18)
            ax[1,1].set_ylabel('CubeZ',fontsize=18)
            ax[1,1].set_title('CubeX = 3',fontsize=20)

            
            fig.colorbar(fig3,ax=ax.ravel().tolist())

        if projection==1:
            fig.suptitle('Y planes', fontsize=24)

            fig0=ax[0,0].imshow(plot_arr[:,0,:],interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[0,0].set_xlabel('CubeZ',fontsize=18)
            ax[0,0].set_ylabel('CubeX',fontsize=18)
            ax[0,0].set_title('CubeY = 0',fontsize=20)

            fig1=ax[0,1].imshow(plot_arr[:,1,:],interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[0,1].set_xlabel('CubeZ',fontsize=18)
            ax[0,1].set_ylabel('CubeX',fontsize=18)
            ax[0,1].set_title('CubeY = 1',fontsize=20)


            fig2=ax[1,0].imshow(plot_arr[:,2,:],interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[1,0].set_xlabel('CubeZ',fontsize=18)
            ax[1,0].set_ylabel('CubeX',fontsize=18)
            ax[1,0].set_title('CubeY = 2',fontsize=20)


            fig3=ax[1,1].imshow(plot_arr[:,3,:],interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[1,1].set_xlabel('CubeZ',fontsize=18)
            ax[1,1].set_ylabel('CubeX',fontsize=18)
            ax[1,1].set_title('CubeY = 3',fontsize=20)

            fig.subplots_adjust(wspace=0.05,hspace=0.5)
        
        if projection==2:
            fig.suptitle('Z planes', fontsize=24)

            fig0=ax[0,0].imshow(plot_arr[:,:,0].T,interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[0,0].set_xlabel('CubeX',fontsize=18)
            ax[0,0].set_ylabel('CubeY',fontsize=18)
            ax[0,0].set_title('CubeZ = 0',fontsize=20)

            fig1=ax[0,1].imshow(plot_arr[:,:,1].T,interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[0,1].set_xlabel('CubeX',fontsize=18)
            ax[0,1].set_ylabel('CubeY',fontsize=18)
            ax[0,1].set_title('CubeZ = 1',fontsize=20)


            fig2=ax[1,0].imshow(plot_arr[:,:,2].T,interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[1,0].set_xlabel('CubeX',fontsize=18)
            ax[1,0].set_ylabel('CubeY',fontsize=18)
            ax[1,0].set_title('CubeZ = 2',fontsize=20)


            fig3=ax[1,1].imshow(plot_arr[:,:,3].T,interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax,origin='lower')
            ax[1,1].set_xlabel('CubeX',fontsize=18)
            ax[1,1].set_ylabel('CubeY',fontsize=18)
            ax[1,1].set_title('CubeZ = 3',fontsize=20)

            fig.subplots_adjust(wspace=0.05,hspace=0.5)
        
        plt.show()

    def projection(self,df,axis,source,nrot=0,background=None):
        """
        Plots a 2D hitmap of a given plane, summed over the direction given by axis. 
        For example, passing axis = 0 gives a summation over the x direction and thus the y-z plane.

        PARAMETERS
        ------------------------
        df : str
            Name of source dataset stored in self.data.

        background : str
            Name of background dataset stored in self.data.

        axis : int
            The axis along which to sum. Takes values of 0, 1 or 2. 

        source : str
            Name of the source used in the source dataset.

        nrot : int
            Number of rotations by 90 degrees to perform. Choose based on the dataset, 
            as some have the detector rotated.

        backg : bool
            Whether to subtract the background or not. Defaults to True.

        RAISES
        ------------------------
        ValueError
            Raised when the value of axis is not 0, 1 or 2. 
        """
        if axis!=0 and axis!=1 and axis!=2:
            raise ValueError("Incompatible axis value. Please pass either 0, 1 or 2.")

        Zdatamap=np.zeros([4,4,4])
        Zbackmap=np.zeros([4,4,4])

        data=self.data[df]
        if background is not None:
            bg=self.data[background]

        for k in range(4):
            for j in range(4):
                for i in range(4):
                    Zdatamap[i,j,k]=data.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
                    if background is not None:
                        Zbackmap[i,j,k]=bg.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
        
        if background is not None:
            bgnorm=(bg['time'].max()-bg['time'].min())/1e8
            Zbackmap/=2
        datnorm=(data['time'].max() - data['time'].min())/1e8
        
        Zdatamap/=2
        if background is not None:
            Zdmap=Zdatamap*bgnorm/datnorm-Zbackmap   
        else:
            Zdmap=Zdatamap

        dat=Zdmap.sum(axis=axis)
        dat=dat.T
        dat=np.rot90(dat,nrot)
        axes=np.array(['x','y','z'])
        a=~(axes==axes[axis])
        axs=axes[a]
        
        fig,ax=plt.subplots(1,1)
        orig='lower'
        fi=ax.imshow(dat,interpolation='nearest',cmap='viridis',origin=orig,norm=mcolor.Normalize(vmin=dat.min(),vmax=dat.max()))
        plt.colorbar(fi)
        plt.title('{}'.format(source),fontsize=22)
        if axis==0:
            plt.xlabel('CubeY',fontsize=18)
            plt.ylabel('CubeZ',fontsize=18)
        if axis==1:
            plt.xlabel('CubeX',fontsize=18)
            plt.ylabel('CubeZ',fontsize=18)
        if axis==2:
            plt.xlabel('CubeX',fontsize=18)
            plt.ylabel('CubeY',fontsize=18)
        
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        for label in ax.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
            
        ax.tick_params(labelsize=20)

        plt.show()

    def benchmark(self,data,ref,axis,source,refname,refrot=True,databg=None,refbg=None):
        """
        Compares plane distributions of one source with those of a given reference source. 

        PARAMETERS
        ------------------------
        data : str
            Name of source dataset saved in self.data.

        databg : str
            Name of background for source dataset saved in self.data.

        ref : str
            Name of reference source dataset saved in self.data.
        
        refbg : str
            Name of reference source background dataset saved in self.data

        axis : int
            Specifies the axis along which to show the benchmark. 
            0, 1 and 2 correspond to x, y and z axes respectively.

        source : str
            Name of source used for source dataset

        refname : str
            Name of source used for reference dataset

        refrot : bool
            If True, rotates reference data by 180 degrees. Defaults to True.
        """
        data=self.data[data]
        if databg is not None:
            bg=self.data[databg]
        ref=self.data[ref]
        if refbg is not None:
            refbg=self.data[refbg]


        datamap=np.zeros([4,4,4])
        bgmap=np.zeros([4,4,4])
        refmap=np.zeros([4,4,4])
        refbgmap=np.zeros([4,4,4])
        
        
        for k in range(4):
            for j in range(4):
                for i in range(4):
                    datamap[i,j,k]=data.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
                    if databg is not None:
                        bgmap[i,j,k]=bg.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
                    refmap[i,j,k]=ref.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
                    if refbg is not None:
                        refbgmap[i,j,k]=refbg.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()

        if databg is not None:
            datnormt=max((data['time'].max()-data['time'].min()),(bg['time'].max()-bg['time'].min()))
            da=(datamap/(data['time'].max()-data['time'].min())-bgmap/(bg['time'].max()-bg['time'].min()))*datnormt
        else:
            da=datamap


        if refbg is not None:
            refnormt=max((ref['time'].max()-ref['time'].min()),(refbg['time'].max()-refbg['time'].min()))
            refe=(refmap/(ref['time'].max()-ref['time'].min())-refbgmap/(refbg['time'].max()-refbg['time'].min()))*refnormt
        else:
            refe=refmap        
        
        da/=2
        refe/=2                
        datsum=da.sum()
        refsum=refe.sum()    
        
        norm=min(datsum,refsum)
        
        axes=np.array([0,1,2])
        axes=axes[~(axes==axis)]
        
        if refrot:
            refe=np.rot90(refe,2)
        
        da*=(norm/datsum)
        refe*=(norm/refsum)
        
        
        dataplanes=da.sum(axis=tuple(axes))
        refplanes=refe.sum(axis=tuple(axes))
        
        nu=np.array([0,1,2,3],dtype=int)

        plt.bar(nu,dataplanes,width=1,edgecolor='b',alpha=0.5,color='xkcd:cerulean',label='DATA')
        plt.bar(nu,refplanes,width=1,alpha=0.5,color='xkcd:deep blue',label='REF')
        
        
        a=['X','Y','Z']
        
        plt.xlabel('{}'.format(a[axis]),fontsize=18)
        plt.ylabel('Count',fontsize=18)
        plt.title('{} vs {} reference'.format(source,refname),fontsize=22)
        plt.legend()
        
        plt.show()

    def CubeIDplot(self,data,source,bg=None):
        """
        Plots count rate as a function of cube ID, and the same count with the background subtracted.

        PARAMETERS
        ------------------------
        data : str
            Name of the source dataset saved in self.data.

        bg : str
            Name of the background dataset saved in self.data

        source : str
            Name of the source used for the dataset. 
        """
        data=self.data[data]
        if bg is not None:
            bg=self.data[bg]

            df=pd.DataFrame(np.concatenate([np.unique(bg['ZCubeID'],return_counts=True)[1][1:][:,np.newaxis]/2,np.unique(data['ZCubeID'],return_counts=True)[1][1:][:,np.newaxis]/2],axis=1),columns=['background','data'])
            plt.ion()
            df=df.astype(int)
            bgnorm=bg['time'].max()-bg['time'].min()
            datnorm=data['time'].max() - data['time'].min()
            norm=max(bgnorm,datnorm)
        
            fig,ax=plt.subplots(2,1,figsize=(10,20))
        
        
            ax[0].bar(df.index.array,df['data'].values*norm/datnorm,width=1,alpha=0.5,yerr=np.sqrt(df['data'].values),ec='b',color='xkcd:cerulean',label='Source')
            ax[0].bar(df.index.array,df['background'].values*norm/bgnorm,width=1,alpha=0.5,yerr=np.sqrt(df['background'].values),color='xkcd:deep blue',label='Background')
            
            ax[0].set_xlabel('Cube ID',fontsize=18)
            
            ax[0].set_ylabel('Count',fontsize=18)
            ax[0].set_title('{} count by Cube ID'.format(source),fontsize=22)
            
            ax[0].legend(fontsize=20)
            
            ax[1].bar(df.index.array,df['data'].values*norm/datnorm - df['background'].values*norm/bgnorm,width=1,alpha=0.5,yerr=np.sqrt(df['data'].values/2*norm/datnorm - df['background'].values/2*norm/bgnorm),ec='b',color='xkcd:cerulean')
            ax[1].set_xlabel('Cube ID',fontsize=18)
            ax[1].set_ylabel('Count',fontsize=18)
            ax[1].set_title('{} count, background subtracted'.format(source),fontsize=22)
            
            
            fig.tight_layout(pad=10)
            
            plt.show()
        else:
            df=(np.unique(data['ZCubeID'],return_counts=True)[1][1:][:,np.newaxis]/2).astype(int)
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.bar(np.arange(0,64,1),df[:,0],width=1,alpha=0.5,yerr=np.sqrt(df[:,0]),ec='b',color='xkcd:cerulean')
            ax.set_xlabel('Cube ID',fontsize=18)
            ax.set_ylabel('Count',fontsize=18)
            ax.set_title('{} count by Cube ID'.format(source),fontsize=22)
            plt.show()

    def all_color_maps(self,df,source,background=None):
        """
        Plots all of the plane hitmaps, 4 for each direction. If background is not None,
        performs a background subtraction.

        PARAMETERS
        ------------------------
        df : str
            Label of the desired dataset as saved in self.data.

        source : str
            Name of the source used in the chosen dataset.

        background : str or NoneType
            Label of the desired background dataset as saved in self.data.
        """
        Zdatamap=np.zeros([4,4,4])
        data=self.data[df]

        if background is not None:
            bg=self.data[background]
            Zbackmap=np.zeros([4,4,4])
            bgnorm=(bg['time'].max()-bg['time'].min())/1e8
            datnorm=(data['time'].max() - data['time'].min())/1e8

        for k in range(4):
            for j in range(4):
                for i in range(4):
                    Zdatamap[i,j,k]=data.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
                    if background is not None:
                        Zbackmap[i,j,k]=bg.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
            
        Zdatamap/=2
        if background is not None:
            Zbackmap/=2
            Zdmap=Zdatamap*bgnorm/datnorm-Zbackmap   
        else:
            Zdmap=Zdatamap     
        Zmin=Zdmap.min()
        Zvmax=Zdmap.max()

        fig,ax=plt.subplots(3,4,figsize=(40,30))
        ax[0,0].imshow(Zdmap[0,:,:].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[0,0].set_xlabel('CubeY',fontsize=18)
        ax[0,0].set_ylabel('CubeZ',fontsize=18)
        ax[0,0].set_title('CubeX = 0',fontsize=20)

        ax[0,1].imshow(Zdmap[1,:,:].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[0,1].set_xlabel('CubeY',fontsize=18)
        ax[0,1].set_ylabel('CubeZ',fontsize=18)
        ax[0,1].set_title('CubeX = 1',fontsize=20)

        ax[0,2].imshow(Zdmap[2,:,:].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[0,2].set_xlabel('CubeY',fontsize=18)
        ax[0,2].set_ylabel('CubeZ',fontsize=18)
        ax[0,2].set_title('CubeX = 2',fontsize=20)

        ax[0,3].imshow(Zdmap[3,:,:].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[0,3].set_xlabel('CubeY',fontsize=18)
        ax[0,3].set_ylabel('CubeZ',fontsize=18)
        ax[0,3].set_title('CubeX = 3',fontsize=20)

        ax[1,0].imshow(Zdmap[:,0,:].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[1,0].set_xlabel('CubeZ',fontsize=18)
        ax[1,0].set_ylabel('CubeX',fontsize=18)
        ax[1,0].set_title('CubeY = 0',fontsize=20)

        ax[1,1].imshow(Zdmap[:,0,:].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[1,1].set_xlabel('CubeZ',fontsize=18)
        ax[1,1].set_ylabel('CubeX',fontsize=18)
        ax[1,1].set_title('CubeY = 1',fontsize=20)
        
        ax[1,2].imshow(Zdmap[:,0,:].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[1,2].set_xlabel('CubeZ',fontsize=18)
        ax[1,2].set_ylabel('CubeX',fontsize=18)
        ax[1,2].set_title('CubeY = 2',fontsize=20)

        ax[1,3].imshow(Zdmap[:,0,:].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[1,3].set_xlabel('CubeZ',fontsize=18)
        ax[1,3].set_ylabel('CubeX',fontsize=18)
        ax[1,3].set_title('CubeY = 3',fontsize=20)

        ax[2,0].imshow(Zdmap[:,:,0].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[2,0].set_xlabel('CubeX',fontsize=18)
        ax[2,0].set_ylabel('CubeY',fontsize=18)
        ax[2,0].set_title('CubeZ = 0',fontsize=20)

        ax[2,1].imshow(Zdmap[:,:,0].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[2,1].set_xlabel('CubeX',fontsize=18)
        ax[2,1].set_ylabel('CubeY',fontsize=18)
        ax[2,1].set_title('CubeZ = 1',fontsize=20)

        ax[2,2].imshow(Zdmap[:,:,0].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[2,2].set_xlabel('CubeX',fontsize=18)
        ax[2,2].set_ylabel('CubeY',fontsize=18)
        ax[2,2].set_title('CubeZ = 2',fontsize=20)

        ax[2,3].imshow(Zdmap[:,:,0].T,interpolation='nearest',cmap='viridis',vmin=Zmin,vmax=Zvmax,origin='lower')
        ax[2,3].set_xlabel('CubeX',fontsize=18)
        ax[2,3].set_ylabel('CubeY',fontsize=18)
        ax[2,3].set_title('CubeZ = 3',fontsize=20)

        fig.suptitle("{} projections".format(source),fontsize=24)
        fig.tight_layout(pad=15,h_pad=15,w_pad=5)
        plt.show()

    def benchall(self,data,ref,cddat,chpbdat,axis,source,bgsource,refrot=True,bg=None,refbg=None):
        """
        Compares plane distributions against a reference distribution, with different 
        shieldings (bare, Cd, CH2+Pb), in all three directions.

        PARAMETERS
        ------------------------
        data : str
            Name of the source dataset saved in self.data.

        bg : str
            Name of the background dataset saved in self.data.

        ref : str
            Name of the reference dataset saved in self.data.

        refbg : str
            Name of the reference background dataset saved in self.data.

        cddat : str
            Name of reference dataset with Cd shielding saved in self.data.

        chpbdat : str
            Name of reference dataset with CH2 + Pb shielding saved in self.data.

        axis : int
            Axis of planes to plot. 0, 1 and 2 refer to x, y and z planes respectively. 

        source : str
            Name of source used for the source dataset

        bgsource : str
            Name of source used for the reference dataset

        refrot : bool
            If True, rotate the reference source by 180 degrees. 
        """
        datamap=np.zeros([4,4,4])
        bgmap=np.zeros([4,4,4])
        refmap=np.zeros([4,4,4])
        refbgmap=np.zeros([4,4,4])
        cdmap=np.zeros([4,4,4])
        chpbmap=np.zeros([4,4,4])
        cdbgmap=np.zeros([4,4,4])
        chpbbgmap=np.zeros([4,4,4])

        data=self.data[data]
        ref=self.data[ref]
        cddat=self.data[cddat]
        chpbdat=self.data[chpbdat]
        if bg is not None:
            bg=self.data[bg]
        if refbg is not None:
            refbg=self.data[refbg]

        
        for k in range(4):
            for j in range(4):
                for i in range(4):
                    datamap[i,j,k]=data.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
                    refmap[i,j,k]=ref.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
                    cdmap[i,j,k]=cddat.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
                    chpbmap[i,j,k]=chpbdat.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
                    if bg is not None:
                        bgmap[i,j,k]=bg.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()
                    if refbg is not None:
                        refbgmap[i,j,k]=refbg.query('CubeX=={} & CubeY=={} & CubeZ=={}'.format(i,j,k))['CubeX'].count()


        if refbg is not None:
            cdbgmap=bgmap
            chpbbgmap=bgmap
            cdbg=bg
            chpbbg=bg

            datnormt=max((data['time'].max()-data['time'].min()),(bg['time'].max()-bg['time'].min()))
            refnormt=max((ref['time'].max()-ref['time'].min()),(refbg['time'].max()-refbg['time'].min()))
            cdnormt=max((cddat['time'].max()-cddat['time'].min()),(cdbg['time'].max()-cdbg['time'].min()))
            chpbnormt=max((chpbdat['time'].max()-chpbdat['time'].min()),(chpbbg['time'].max()-chpbbg['time'].min()))
        
        
            da=(datamap/(data['time'].max()-data['time'].min())-bgmap/(bg['time'].max()-bg['time'].min()))*datnormt
            refe=(refmap/(ref['time'].max()-ref['time'].min())-refbgmap/(refbg['time'].max()-refbg['time'].min()))*refnormt
            cdt=(cdmap/(cddat['time'].max()-cddat['time'].min())-cdbgmap/(cdbg['time'].max()-cdbg['time'].min()))*cdnormt
            chpbt=(chpbmap/(chpbdat['time'].max()-chpbdat['time'].min())-chpbbgmap/(chpbbg['time'].max()-chpbbg['time'].min()))*chpbnormt
        else:
            da = datamap
            refe=refmap
            cdt=cdmap
            chpbt = chpbmap
        da/=2
        refe/=2
        cdt/=2
        chpbt/=2
        
        datsum=da.sum()
        refsum=refe.sum()
        cdsum=cdt.sum()
        chpbsum=chpbt.sum()
        norm=min(datsum,refsum,cdsum,chpbsum)

        
        axes=np.array([0,1,2])
        axes=axes[~(axes==axis)]
        
        if refrot:
            refe=np.rot90(refe,2)
        
        da*=(norm/datsum)
        refe*=(norm/refsum)
        cdt*=(norm/cdsum)
        chpbt*=(norm/chpbsum)
        
        dataplanes=da.sum(axis=tuple(axes))
        refplanes=refe.sum(axis=tuple(axes))
        cdplanes=cdt.sum(axis=tuple(axes))
        chpbplanes=chpbt.sum(axis=tuple(axes))
        
        
        histdat=np.concatenate([np.repeat(0,dataplanes[0]),np.repeat(1,dataplanes[1]),np.repeat(2,dataplanes[2]),np.repeat(3,dataplanes[3])])
        histref=np.concatenate([np.repeat(0,refplanes[0]),np.repeat(1,refplanes[1]),np.repeat(2,refplanes[2]),np.repeat(3,refplanes[3])])
        histcd=np.concatenate([np.repeat(0,cdplanes[0]),np.repeat(1,cdplanes[1]),np.repeat(2,cdplanes[2]),np.repeat(3,cdplanes[3])])
        histchpb=np.concatenate([np.repeat(0,chpbplanes[0]),np.repeat(1,chpbplanes[1]),np.repeat(2,chpbplanes[2]),np.repeat(3,chpbplanes[3])])
        
        
        #lim=min(dataplanes.min(),refplanes.min(),cdplanes.min(),chpbplanes.min())
        lim=1000
        nu=np.array([0,1,2,3],dtype=int)
        
        a=['X','Y','Z']
        
        #plt.bar(nu,dataplanes*datsum/norm,width=1,alpha=1,edgecolor='blue',yerr=np.sqrt(dataplanes),fill=False,linewidth=2,label='DATA')
        #plt.bar(nu,refplanes*refsum/norm,width=1,alpha=1,edgecolor='darkblue',linestyle='--',yerr=np.sqrt(refplanes),fill=False,linewidth=2,label='REF')
        
        fig,ax=plt.subplots()
        
        ax.hist(histref,bins=4,range=[-0.5,3.5],label='Cf reference',histtype='step',alpha=0.5,linestyle='--',color='xkcd:grey blue',linewidth=2)

        ax.hist(histcd,bins=4,range=[-0.5,3.5],label='Cd',histtype='step',alpha=0.5,color='xkcd:dark blue',linewidth=4)
        ax.hist(histchpb,bins=4,range=[-0.5,3.5],label='CH2-Pb',histtype='step',color='xkcd:magenta',linewidth=4)
        ax.hist(histdat,bins=4,range=[-0.5,3.5],label='Bare',histtype='step',color='black',linewidth=4)    
        
        plt.xlabel('{} plane'.format(a[axis]),fontsize=26)
        plt.ylabel('Count',fontsize=26)
        plt.title('{} vs {} reference'.format(source,bgsource),fontsize=32)
        plt.legend(fontsize=26,loc='upper left')
        plt.ylim(bottom=lim-1000)
        
        for label in ax.xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
            
        plt.tick_params(labelsize=26)
        
        plt.show()

    def PIDplot(self,data,source):
        """
        Plots the particle identification plot, of signal amplitude 
        against time over threshold. Also groups events by coincidence numbers.

        PARAMETERS
        ------------------------
        data : str
            Name of the dataset saved in self.data to use to plot.

        source : str
            Name of the source used in the dataset. 
        """
        data=self.data[data]

        fig,ax=plt.subplots(dpi=130)
        
        ax.scatter(data.query("coincidence==1")['peak_amplitude'].values,
                    data.query("coincidence==1")['tot'].values,
                alpha=0.05,c='xkcd:magenta',marker='.',label='1')
        ax.scatter(data.query("coincidence==2")['peak_amplitude'].values,
                    data.query("coincidence==2")['tot'].values,
                    alpha=0.05,c='xkcd:dark blue',marker='.',label='2')
        ax.scatter(data.query("coincidence>2")['peak_amplitude'].values,
                    data.query("coincidence>2")['tot'].values,
                    alpha=0.05,c='green',marker='.',label=r'$\\>$ 2') 
    
        ax.set_xlabel('Peak Amplitude',fontsize=30)
        ax.set_ylabel('Discrimination metric',fontsize=30)
        leg=ax.legend(frameon=0,scatterpoints=3,title='Coincidence',fontsize=24)
        ax.set_title('{} PID'.format(source),fontsize=30)
        
        for l in leg.legendHandles:
            l.set_alpha(0.5)
        
        ax.minorticks_on()
        ax.tick_params(direction='in',which='both',labelsize=30)
        
        plt.show()

    def peakValueplot(self,data,source,n_bins,background=None):
        """
        Plots a histogram of the waveform peak values. 

        PARAMETERS
        ------------------------
        data : str
            Label of desired dataset as saved in self.data.

        source : str
            Name of the source for the specified dataset.

        n_bins : int
            Number of bins to calculate for the histogram

        background : str or NoneType
            Label of desired background dataset as saved in self.data. 
            If NoneType, background is not subtracted.
        """
        dat=self.data[data]
        bins=np.linspace(0,16400,n_bins)
        dattmax=dat['time'].max()
        Ndat,bin_edges=np.histogram(dat['peakValue'],bins=bins)
        
        if background is not None:
            bkg=self.data[background]
            bkgtmax=bkg['time'].max()
            Nbkg,bin_edges=np.histogram(bkg['peakValue'],bins=bins)

       

            if dattmax>bkgtmax:
                Nplot=(Ndat/dattmax)*bkgtmax - Nbkg
            else:
                Nplot=Ndat - (Nbkg/bkgtmax)*dattmax

        else:
            Nplot=Ndat
        plt.bar(bin_edges[:-1]+(16400/n_bins)/2,Nplot,width=16400/n_bins)
        plt.show()

    def alex_projection(self, data):
        cubs=self.data[data].query('Combine').loc[:,['CubeX','CubeY','CubeZ']]
        return pd.DataFrame(np.array([np.unique(cubs['CubeX'],return_counts=True)[1],
                            np.unique(cubs['CubeY'],return_counts=True)[1],
                            np.unique(cubs['CubeZ'],return_counts=True)[1]]).T/2,
                            columns=['Xdiff','Ydiff','Zdiff'])

    def alex_projection2(self, data):
        cubs=self.data[data].query('Combine').loc[:,['CubeX','CubeY','CubeZ']]
        data = pd.DataFrame(np.array([np.unique(cubs['CubeX'],return_counts=True)[1],
                            np.unique(cubs['CubeY'],return_counts=True)[1],
                            np.unique(cubs['CubeZ'],return_counts=True)[1]]).T/2,
                            columns=['Xdiff','Ydiff','Zdiff'])

        def func(x, a, b, c):
            return a * sps.expon.pdf(x, b, c)

        bins = np.arange(0., 5., 1.)
        xdata = (bins[:-1] + bins[1:]) / 2
        ydata, _ = np.histogram(data["Ydiff"].tolist(), bins=bins)
        po, po_cov = curve_fit(func, xdata, data["Ydiff"], [data["Ydiff"][0], 0, 1])
        po_x, _ = curve_fit(func, xdata, data["Xdiff"], [data["Xdiff"][0], 0, 1])
        po_z, _ = curve_fit(func, xdata, data["Zdiff"], [data["Zdiff"][0], 0, 1])
        print("A = ", po[0])
        print("B = ", po[1])
        print("C = ", po[2])
        sps_fit = po
        print("sps fit", sps_fit)
        fit_data =  func(xdata, *po)
        data["Yfit"] = fit_data
        data["Xfit"] = func(xdata, *po_x)
        data["Zfit"] = func(xdata, *po_z)
        print("Y data:", data["Ydiff"].tolist())
        print("Fit data:", fit_data)
        print(sps.chisquare(ydata, fit_data, 3))
        return data


    def fit_projection(self,data,source,fitconst,bkg=None,plot=False):
        """
        Calculates 1D projections of rates in X, Y and Z planes, and fits an exponential function
        to any directions identified as having a gradient across the planes. Can also optionally 
        plot these distributions and exponential fits.

        PARAMETERS
        ------------------------
        data : str
            Label for the desired dataset, as saved in self.data.

        source : str
            Name of the dataset used, for plot title.

        fitconst : int
            Constant added to stabilise fitting.

        bkg : str or NoneType
            Label for the desired background dataset. If NoneType, background is not subtracted.

        plot : bool
            If True, plot the calculated distributions and exponentials.

        RETURNS
        ------------------------
        parameters : ndarray
            Array of fit parameters and their uncertainties. Any fits that failed have np.nan 
            listed for all parameters.
        """
        data=self.data[data]
        cubs=data.query('Combine').loc[:,['CubeX','CubeY','CubeZ']]
        if bkg is not None:
            bkg=self.data[bkg]
            bcubs=bkg.query('Combine').loc[:,['CubeX','CubeY','CubeZ']]
            planes=pd.DataFrame(np.array([np.unique(cubs['CubeX'],return_counts=True)[1],np.unique(cubs['CubeY'],return_counts=True)[1],np.unique(cubs['CubeZ'],return_counts=True)[1],np.unique(bcubs['CubeX'],return_counts=True)[1],np.unique(bcubs['CubeY'],return_counts=True)[1],np.unique(bcubs['CubeZ'],return_counts=True)[1]]).T/2,columns=['X','Y','Z','bkgX','bkgY','bkgZ'])
            datnorm=(data['time'].max()-data['time'].min())/1e8
            bgnorm=(bkg['time'].max()-bkg['time'].min())/1e8
            norm=bgnorm/datnorm
            if norm<=1:
                planes['Xdiff']=norm*planes['X']-planes['bkgX']
                planes['Ydiff']=norm*planes['Y']-planes['bkgY']
                planes['Zdiff']=norm*planes['Z']-planes['bkgZ']
            else:
                planes['Xdiff']=planes['X']-planes['bkgX']/norm
                planes['Ydiff']=planes['Y']-planes['bkgY']/norm
                planes['Zdiff']=planes['Z']-planes['bkgZ']/norm
        else:
            planes=pd.DataFrame(np.array([np.unique(cubs['CubeX'],return_counts=True)[1],np.unique(cubs['CubeY'],return_counts=True)[1],np.unique(cubs['CubeZ'],return_counts=True)[1]]).T/2,columns=['Xdiff','Ydiff','Zdiff'])

        
        def en(x,a,b):
            return a*np.exp(-1*x/b) + fitconst

        def ep(x,a,b):
            return a*np.exp(x/b) + fitconst

        xyz=['Xdiff','Ydiff','Zdiff']
        diffs=planes.loc[:,['Xdiff','Ydiff','Zdiff']].diff().iloc[1:]
        parameters=np.zeros([3,4])
        for i in range(3):
            plane=xyz[i]
            if planes.loc[planes[plane]==max(planes[plane])].index[0]==0:
                if (diffs[plane]<0).all(axis=0):
                    ma=max(planes[plane])
                    popt,pcov=curve_fit(en,planes.index.values,planes[plane].values,p0=[ma,2],sigma=np.sqrt(planes[plane].values))
                    if pcov[1,1] < 1:
                        parameters[i,0],parameters[i,2]=popt[0],popt[1]
                        parameters[i,1],parameters[i,3]=pcov[0,0],pcov[1,1]
                    else:
                        parameters[i]=np.nan
                else:
                    parameters[i]=np.nan
            elif planes.loc[planes[plane]==max(planes[plane])].index[0]==1:
                parameters[i]=np.nan
                """
                if (diffs[plane].iloc[1:]<0).all():
                    ma=max(planes[plane])
                    popt,pcov=curve_fit(en,planes.index.values[1:],planes[plane].values[1:],p0=[ma,2],sigma=np.sqrt(planes[plane].values[1:]))
                    if pcov[1,1]<1:
                        parameters[i,0],parameters[i,2]=popt[0],popt[1]
                        parameters[i,1],parameters[i,3]=pcov[0,0],pcov[1,1]
                    else:
                        parameters[i,0],parameters[i,1],parameters[i,2],parameters[i,3]=np.nan,np.nan,np.nan,np.nan
                else:
                    parameters[i]=np.nan
                """
            elif planes.loc[planes[plane]==max(planes[plane])].index[0]==2:
                parameters[i]=np.nan
                """
                if (diffs[plane].iloc[:-1]>0).all():
                    ma=max(planes[plane])
                    popt,pcov=curve_fit(ep,planes.index.values[:-1],planes[plane].values[:-1],p0=[ma,2],sigma=np.sqrt(planes[plane].values[:-1]))
                    if pcov[1,1]<1:
                        parameters[i,0],parameters[i,2]=popt[0],popt[1]
                        parameters[i,1],parameters[i,3]=pcov[0,0],pcov[1,1]
                    else:
                        parameters[i,0],parameters[i,1],parameters[i,2],parameters[i,3]=np.nan,np.nan,np.nan,np.nan
                else:
                    parameters[i]=np.nan
                """
            elif planes.loc[planes[plane]==max(planes[plane])].index[0]==3:
                if (diffs[plane]>0).all(axis=0):
                    ma=max(planes[plane])
                    popt,pcov=curve_fit(ep,planes.index.values,planes[plane].values,p0=[ma,2],sigma=np.sqrt(planes[plane].values))
                    if pcov[1,1]<1:
                        parameters[i,0],parameters[i,2]=popt[0],popt[1]
                        parameters[i,1],parameters[i,3]=pcov[0,0],pcov[1,1]
                    else:
                        parameters[i]=np.nan
                else:
                    parameters[i]=np.nan

        if plot:
            space=np.linspace(-0.2,3.2,100)
            fig,ax=plt.subplots(1,3,figsize=(30,10))
            ax[0].bar(planes.index,planes['Xdiff'].values,width=1,edgecolor='b',alpha=0.5,yerr=np.sqrt(planes['Xdiff'].values),color='xkcd:cerulean')
            if not np.isnan(parameters[0,0]):
                if np.argmax(planes['Xdiff'])<=1:
                    func=en
                    ax[0].plot(space,func(space,parameters[0,0],parameters[0,2]),c='r',label=r'$\lambda$ = {}'.format(-1*round(parameters[0,2],2)))   
                else:
                    func=ep
                    ax[0].plot(space,func(space,parameters[0,0],parameters[0,2]),c='r',label=r'$\lambda$ = {}'.format(round(parameters[0,2],2)))   
            
            ax[1].bar(planes.index,planes['Ydiff'].values,width=1,edgecolor='b',alpha=0.5,yerr=np.sqrt(planes['Ydiff'].values),color='xkcd:cerulean')
            if not np.isnan(parameters[1,0]):
                if np.argmax(planes['Ydiff'])<=1:
                    func=en
                    ax[1].plot(space,func(space,parameters[1,0],parameters[1,2]),c='r',label=r'$\lambda$ = {}'.format(-1*round(parameters[1,2],2)))
                else:
                    func=ep
                    ax[1].plot(space,func(space,parameters[1,0],parameters[1,2]),c='r',label=r'$\lambda$ = {}'.format(round(parameters[1,2],2)))

            ax[2].bar(planes.index,planes['Zdiff'].values,width=1,edgecolor='b',alpha=0.5,yerr=np.sqrt(planes['Zdiff'].values),color='xkcd:cerulean')
            if not np.isnan(parameters[2,0]):
                if np.argmax(planes['Zdiff'])<=1:
                    func=en
                    ax[2].plot(space,func(space,parameters[2,0],parameters[2,2]),c='r',label=r'$\lambda$ = {}'.format(-1*round(parameters[2,2],2)))
                else:
                    func=ep
                    ax[2].plot(space,func(space,parameters[2,0],parameters[2,2]),c='r',label=r'$\lambda$ = {}'.format(round(parameters[2,2],2)))
                
                

            pl=['X planes','Y planes','Z planes']
            for i in range(3):
                ax[i].set_xlabel('Plane number',fontsize=26)
                ax[i].set_ylabel('Background corrected count',fontsize=26)
                ax[i].set_title(pl[i],fontsize=30)
                if not np.isnan(parameters[i,0]):
                    ax[i].legend(loc='upper right',fontsize=36)
            
                ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[i].tick_params(labelsize=26)
                
            #fig.set_tight_layout(True)
            fig.suptitle(source,fontsize=30)
            #fig.subplots_adjust(left=0.05,bottom=0.09,right=0.99,top=0.31,wspace=0.17,hspace=0.2)
            plt.show()

        return parameters

    def comp_proj_fits(self,ds,bkgs,sources,fitconst,refs=None,crefs=None,labels=None):
        """
        Calculates exponential fit to 1D planes counts for a list of datasets,
        then plots a bar chart comparing them, with an error on each bar equal to 
        3 standard deviations.

        PARAMETERS
        ------------------------
        ds : list of str
            List of labels of desired datasets to compare, as saved in self.data

        bkgs : list of str or NoneType
            List of labels of desired background datasets as saved in self.data. For NoneType entries,
            the corresponding dataset is not background subtracted.

        sources : list of str
            List of names of datasets, used for labelling.

        fitconst : float
            Constant used to stabilize exponential fit.

        refs : list of int or NoneType
            List of reference lambda values to plot, if NoneType then no references.

        crefs : list of colours or NoneType
            List of colours for plotting the reference lambda values.

        labels : list of str or NoneType.
            List of labels for the reference data points.
        """
        lamb,std=[],[]
        fig=plt.figure()
        ax=fig.add_subplot(111)
        for i in range(len(ds)):
            params=self.fit_projection(ds[i],sources[i],fitconst,bkgs[i],plot=False)
            lamb.append(params[1,2])
            std.append(params[1,3])
        
        ax.bar(np.linspace(0,len(ds),len(ds)),lamb,yerr=3*std)
        ss=['']+sources
        ax.set_xticklabels(ss,fontsize=24)
        if refs is not None:
            for i in range(len(refs)):
                ax.plot(np.linspace(-1,len(ds)+1,len(ds)+3),np.repeat(refs[i],len(ds)+3),c=crefs[i],lable=labels[i])
        ax.set_ylabel(r'$\lambda$',fontsize=30)
        ax.set_xlim(-0.5,len(ds)-0.5)
        ax.tick_params(labelsize=20)
        plt.rcParams['legend.title_fontsize']=24
        plt.legend(loc='upper right',title='Reference',fontsize=20,bbox_to_anchor=(1,0.9))
        fig.set_tight_layout(True)
        plt.show()

    def View_3D(self,data,source,bkg=None):
        """
        Plots a 3D view of the given dataset, with optional background subtraction.

        PARAMETERS
        ------------------------
        data : str
            The desired dataset to visualise, as saved in self.data.

        source : str
            The name of the dataset used, to label the plot

        bkg : str or NoneType
            The desired background dataset for background subtraction, as saved in self.data.
            If NoneType, no background subtraction is done.
        """
        dd=self.data[data]
        cc=np.unique(dd[dd['Combine']].loc[:,['CubeX','CubeY','CubeZ']],axis=0,return_counts=True)
        c=cc[1]
        if bkg is not None:
            db=self.data[bkg]
            cb=np.unique(db[db['Combine']].loc[:,['CubeX','CubeY','CubeZ']],axis=0,return_counts=True)
            datnorm=(dd['time'].max()-dd['time'].min())/1e8
            bgnorm=(db['time'].max()-db['time'].min())/1e8
            norm=bgnorm/datnorm
            if norm < 1:
                c=c*norm
                c=c-cb[1]
            else:
                c=c-cb[1]/norm
        arr=np.zeros((4,4,4))
        arr[cc[0][:,0],cc[0][:,1],cc[0][:,2]]=c
        arr/=2
        sc=(50.888-50)/50
        p=ndap.NDArrayPlotter(arr,spacing=("even",sc,sc,sc))
        p.set_alpha(0.05)

        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        fig.set_tight_layout(True)
        cmap=cm.jet
        norm=mcolor.Normalize(vmin=0,vmax=np.max(arr))
        p.colors=cmap(norm(arr))
        alph=norm(arr)*0.95
        alph+=0.04
        p.alphas=alph 
        sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
        sm.set_array(arr)
        p.render(azim=-56,elev=25,ax=ax,text=None,space=0.5,labels=True)
        ax.quiver(-0.4,-0.4,-0.4,1,0,0,length=5,arrow_length_ratio=0.05,color='black')
        ax.quiver(-0.4,-0.4,-0.4,0,1,0,length=5,arrow_length_ratio=0.05,color='black')
        ax.quiver(-0.4,-0.4,-0.4,0,0,1,length=5,arrow_length_ratio=0.05,color='black')
        fig.suptitle(source,fontsize=30)
        cbar=plt.colorbar(sm,ax=ax)
        cbar.set_label('Event count',rotation=270,fontsize=30,labelpad=30)
        plt.show()
    
    def cmaps(self,data,source,bkg=None):
        """
        Plots 2D hitmaps summed along each axis, with optional background subtraction.

        PARAMETERS
        ------------------------
        data : str
            The desired dataset to plot, as saved in self.data.

        source : str
            The label for the desired dataset to plot, to label the plot.

        bkg : str or NoneType
            The desired background dataset for background subtraction. If NoneType, 
            no background subtraction is done.
        """
        dd=self.data[data]
        if bkg is not None:
            db=self.data[bkg]
        counts=[]
        bcounts=[]
        ze=np.zeros([4,4,4])
        coords=np.argwhere(ze==0)
        for i in coords:
            counts.append(np.count_nonzero((dd[dd['Combine']].loc[:,['CubeX','CubeY','CubeZ']].values==i).all(axis=1)))
            if bkg is not None:
                bcounts.append(np.count_nonzero((db[db['Combine']].loc[:,['CubeX','CubeY','CubeZ']].values==i).all(axis=1)))
        if bkg is not None:
            datnorm=(dd['time'].max()-dd['time'].min())/1e8
            bgnorm=(db['time'].max()-db['time'].min())/1e8
            norm =bgnorm/datnorm
            counts=np.array(counts)
            if norm < 1:
                counts=counts*norm - bcounts
            else:
                counts=counts - bcounts/norm
        
        ze[coords[:,0],coords[:,1],coords[:,2]]=counts
        ze/=2
        X=ze.sum(axis=0)
        Y=ze.sum(axis=1)
        Z=ze.sum(axis=2)
        X=X.T
        Y=Y.T
        Z=Z.T

        fig,ax=plt.subplots(1,3,figsize=(30,10))
        xp=ax[0].imshow(X,interpolation='nearest',cmap='viridis',origin='lower',norm=mcolor.Normalize(vmin=X.min(),vmax=X.max()))
        yp=ax[1].imshow(Y,interpolation='nearest',cmap='viridis',origin='lower',norm=mcolor.Normalize(vmin=Y.min(),vmax=Y.max()))
        zp=ax[2].imshow(Z,interpolation='nearest',cmap='viridis',origin='lower',norm=mcolor.Normalize(vmin=Z.min(),vmax=Z.max()))
        xcbar=plt.colorbar(xp,ax=ax[0])
        ycbar=plt.colorbar(yp,ax=ax[1])
        zcbar=plt.colorbar(zp,ax=ax[2])
        xcbar.ax.tick_params(labelsize=26)
        ycbar.ax.tick_params(labelsize=26)
        zcbar.ax.tick_params(labelsize=26)

        ax[0].set_xlabel('CubeY',fontsize=28)
        ax[0].set_ylabel('CubeZ',fontsize=28)

        ax[1].set_xlabel('CubeX',fontsize=28)
        ax[1].set_ylabel('CubeZ',fontsize=28)

        ax[2].set_xlabel('CubeX',fontsize=28)
        ax[2].set_ylabel('CubeY',fontsize=28)

        xyz=['X','Y','Z']
        for i in range(3):
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))
            ax[i].tick_params(labelsize=26)
            ax[i].set_title(xyz[i],fontsize=32)

        fig.suptitle(source,fontsize=36)
        fig.set_tight_layout(True)
        plt.show()

    def fit1(self,counts):
        """
        Find the direction of the source by fitting an arrow pointing from the 
        center of a 4 by 4 detector.
        
        PARAMETERS
        ------------------------
        counts : 1D array
            The array of counts ordered in increasing x, y and z.

        RETURNS
        ------------------------
        theta : float
            The vertical angle to the x-y plane of the reconstructed direction vector. 

        phi : float
            The angle in the x-y plane of the reconstructed direction vector.
        
        """    
        def N(xyz,nx,ny,nz,p):
            n=nx*xyz[:,0]+ny*xyz[:,1]+nz*xyz[:,2]+p
            return n
        xyz=[]
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    xyz.append([x,y,z])
        
        popt,pcov=curve_fit(N,xyz,counts,p0=[0,0,0,0],sigma=(counts+1)**0.5)
        r=(popt[0]**2+popt[1]**2+popt[2]**2)**0.5
        theta=np.rad2deg(np.arccos(popt[2]/r))
        phi=np.rad2deg(np.arctan2(popt[1], popt[0]))
        return theta,phi,pcov

    def single_dir_fit(self,data,bkg=None,nfits=10):
        """
        Performs a direction fit on a given dataset, based on Poisson draws based on the raw data. 
        
        PARAMETERS
        ------------------------
        data : str
            The desired dataset for the fit, as saved in self.data.

        bkg : str or NoneType
            The desired background dataset for background subtraction, as saved in self.data. 
            If NoneType, no background subtraction is performed.

        nfits : int
            Number of draws taken from the Poisson distribution to calculate mean and standard deviation.

        RETURNS
        ------------------------
        thets : list of floats
            List of vertical angles of the fitted direction vector.

        phis : list of floats
            List of x-y plane angles of the fitted direction vector.
        """
        dd=self.data[data]
        dcounts=[]
        if bkg is not None:
            db=self.data[bkg]
            bcounts=[]
        ze=np.zeros([4,4,4])
        coords=np.argwhere(ze==0)
        for i in coords:
            dcounts.append(np.count_nonzero((dd[dd['Combine']].loc[:,['CubeX','CubeY','CubeZ']].values==i).all(axis=1)))
            if bkg is not None:
                bcounts.append(np.count_nonzero((db[db['Combine']].loc[:,['CubeX','CubeY','CubeZ']].values==i).all(axis=1)))
        if bkg is not None:
            datnorm=(dd['time'].max()-dd['time'].min())/1e8
            bgnorm=(db['time'].max()-db['time'].min())/1e8
            norm =bgnorm/datnorm
            dcounts=np.array(dcounts)
            if norm < 1:
                dcounts=dcounts*norm - bcounts
            else:
                dcounts=dcounts - bcounts/norm

        ze[coords[:,0],coords[:,1],coords[:,2]]=dcounts
        ze/=2

        counts=[]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    counts.append(ze[k,j,i])
        counts=np.array(counts)
        phis,thets,covs=[],[],[]
        for i in range(nfits):
            sim=np.random.poisson(counts)
            thet,phi,cov=self.fit1(sim)
            thets.append(thet)
            phis.append(phi)
            covs.append(covs)
        
        thets=90-np.array(thets)
        phis=np.array(phis)
        #covs=np.array(covs)

        return thets,phis#,covs

    def plot_directions(self,ds,bkgs,nfits,colors,labels,fillcols,markers):
        """
        Plots fitted directions for a list of datasets, on a polar plot. The radius indicates the 
        vertical angle, whilst the angle is the x-y plane angle. Directions are plotted as points 
        with a shaded region indicating 3 standard deviations.

        PARAMETERS
        ------------------------
        ds : list of str
            List of datasets to fit and plot, as saved in self.data.

        bkgs : list of str or NoneType
            List of background datasets for subtraction, as saved in self.data. 
            If NoneType, no background subtraction is performed.

        nfits : int
            Number of draws from the Poisson defined by the dataset, for generating 
            mean & standard deviation.

        colors : list of colors
            Colors to plot each dataset in.

        labels : list of str
            Labels for each dataset.

        fillcols : list of colors
            Colors to plot standard deviation region with.

        markers : list of markers
            Marker choice for each dataset.

        """

        fig=plt.figure()
        ax=fig.add_subplot(111,projection='polar')
        degree_sign = u"\N{DEGREE SIGN}"
        plt.grid(True)
        def ellipse(phi,phis,thets,phi0,theta0):
            sqrt=np.sqrt(1-((phi-phi0)/phis)**2)
            return np.rad2deg(theta0-thets*sqrt),np.rad2deg(theta0+thets*sqrt)

        for i in range(len(ds)):
            thet,phi=self.single_dir_fit(ds[i],bkgs[i],nfits)
            mphi,mthet=np.mean(phi),np.mean(thet)
            stdphi,stdthet=np.std(phi),np.std(thet)
            tphi,tthet=np.deg2rad(mphi+3*stdphi),np.deg2rad(mthet+3*stdthet)
            bphi,bthet=np.deg2rad(mphi-3*stdphi),np.deg2rad(mthet-3*stdthet)
            mphi,mthet=np.deg2rad(mphi),np.deg2rad(mthet)
            stdphi,stdthet=np.deg2rad(stdphi),np.deg2rad(stdthet)
            
            phirang=np.linspace(bphi,tphi,100)
            bthets,tthets=ellipse(phirang,3*stdphi,3*stdthet,mphi,mthet)
            
            ax.fill_between(phirang,bthets,tthets,color=fillcols[i],alpha=0.3,edgecolor=mcolor.to_rgba(colors[i]))
            ax.scatter(mphi,np.rad2deg(mthet),c=colors[i],label=labels[i],s=15,marker=markers[i])

        sourcethet=np.rad2deg(np.arctan2(-1*(91-27+12.5-37),95))
        source90thet=sourcethet
        sourcephi=np.deg2rad(90)
        source90phi=np.deg2rad(0)

        ax.scatter(sourcephi,sourcethet,marker='x',color='xkcd:blue',label='Source location, Position 1',s=100)
        ax.scatter(source90phi,source90thet,marker='x',color='xkcd:red',label='Source location, Position 2',s=100)

        labels=np.array([-80,'',-40,'',0,'',40,'',80])
        
        ll=np.array([str(label)+degree_sign for label in labels])
        ll[1::2]=''
        ll[4]=''
        ax.set_yticklabels(ll)
        ax.tick_params(labelsize=22)
        lgnd=plt.legend(loc='upper right',bbox_to_anchor=(0.5,0.5),fontsize=24,framealpha=1)
        for i in range(len(ds)+2):
            lgnd.legendHandles[i]._sizes=[40]
        
        ax.tick_params(pad=20)
        ax.set_ylim(-90,90)
        fig.subplots_adjust(top=0.85)
        ax.set_rlabel_position(135)
        label_pos=ax.get_rlabel_position()
        ax.text(np.radians(label_pos+10),10,r'$\varphi$',ha='center',va='center',fontsize=24)
        ax.text(np.radians(label_pos-5),5,'0'+degree_sign,ha='center',va='center',fontsize=22)

        ax.text(0,135,r'$\theta$',ha='center',va='center',fontsize=24)
        ax.plot(np.linspace(0,2*np.pi,100),np.repeat(0,100),c='black',lw=2)
        #ax.set_xlim(np.deg2rad(-90),np.deg2rad(180))

        fig.set_tight_layout(True)
        plt.show()

    def frac_fit(self,data,npoints,bkg=None,nfits=10):
        """
        Performs a directional fit to a given dataset, taking a specific 
        number of neutrons from the dataset.

        PARAMETERS
        ------------------------
        data : str
            The desired dataset for the fit, as saved in self.data.

        npoints : int
            Number of neutrons to fit to.

        bkg : str or NoneType
            The desired background dataset for background subtraction. If NoneType, 
            no background subtraction is performed.

        nfits : int
            Number of draws from the Poisson distribution for statistics on the direction fit.

        RETURNS
        ------------------------
        thets : float
            List of vertical angles of the fitted direction vectors.

        phis : float
            List of x-y plane angles of the fitted direction vectors.

        mthet : float
            Mean vertical angle of the fitted direction vectors.

        mphi : float
            Mean x-y plane angle of the fitted direction vectors.

        stdthet : float
            Standard deviation of the vertical angles of the fitted direction vectors.

        stdphi : float
            Standard deviation of the x-y plane angles of the fitted direction vectors.

        tthet : float
            Upper bound of the vertical angle for the fitted direction vectors, mean + 3*standard deviation

        tphi : float
            Upper bound of the x-y plane angle for the fitted direction vectors, mean + 3*standard deviation

        bthet : float
            Lower bound of the vertical angle for the fitted direction vectors, mean - 3*standard deviation

        bphi : float
            Lower bound of the x-y plane angle for the fitted direction vectors, mean - 3*standard deviation
        """
        dd=self.data[data].query('Combine').iloc[:int(2*npoints)]
        dcounts=[]
        if bkg is not None:
            db=self.data[bkg]
            bcounts=[]
        ze=np.zeros([4,4,4])
        coords=np.argwhere(ze==0)
        for i in coords:
            dcounts.append(np.count_nonzero((dd[dd['Combine']].loc[:,['CubeX','CubeY','CubeZ']].values==i).all(axis=1)))
            if bkg is not None:
                bcounts.append(np.count_nonzero((db[db['Combine']].loc[:,['CubeX','CubeY','CubeZ']].values==i).all(axis=1)))
        if bkg is not None:
            datnorm=(dd['time'].max()-dd['time'].min())/1e8
            bgnorm=(db['time'].max()-db['time'].min())/1e8
            norm=bgnorm/datnorm
            dcounts=np.array(dcounts)
            bcounts=np.array(bcounts)
            if norm < 1:
                #print("dcounts ",dcounts*norm)
                #print("bouncs ",bcounts)
                dcounts=dcounts*norm - bcounts
                
            else:
                #print("dcounts ",dcounts)
                #print("bcounts ",bcounts/norm)
                dcounts=dcounts - bcounts/norm
                #print("counts ",dcounts)

        ze[coords[:,0],coords[:,1],coords[:,2]]=dcounts
        ze/=2
        #print(ze)

        counts=[]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    counts.append(ze[k,j,i])
        counts=np.array(counts)
        #print('counts ',counts)
        counts[counts<0]=0
        #print("zeros counts ",counts)

        
        phis,thets,covs=[],[],[]
        for i in range(nfits):
            sim=np.random.poisson(counts)
            thet,phi,cov=self.fit1(sim)
            thets.append(thet)
            phis.append(phi)
            covs.append(covs)
        
        thets=90-np.array(thets)
        phis=np.array(phis)
        #covs=np.array(covs)

        mphi,mthet=np.mean(phi),np.mean(thet)
        stdphi,stdthet=np.std(phi),np.std(thet)
        tphi,tthet=np.deg2rad(mphi+3*stdphi),np.deg2rad(mthet+3*stdthet)
        bphi,bthet=np.deg2rad(mphi-3*stdphi),np.deg2rad(mthet-3*stdthet)
        mphi,mthet=np.deg2rad(mphi),np.deg2rad(mthet)
        stdphi,stdthet=np.deg2rad(stdphi),np.deg2rad(stdthet)


        return thets,phis,mthet,mphi,stdthet,stdphi,tthet,tphi,bthet,bphi

    def err_develop(self,ds,bkgs,nfits,colors,labels,fillcols):
        """
        Plots the development of the statistics of the fitted direction angles, 
        by increasing the number of neutrons used to fit sequentially. The shaded regions 
        indicate 3 standard deviations, and are coloured to match the dataset point. The point 
        for the fitted direction is only plotted on the final fit.

        PARAMETERS
        ------------------------
        ds : list of str
            List of datsets for fitting, as defined in self.data.

        bkgs : list of str or NoneType
            List of background datasets for background subtraction. If any element 
            is NoneType, the corresponding dataset is not background subtracted.

        nfits : int 
            Number of Poisson draws to do to calculate statistics of the fitted directions.

        colors : list of colours
            List of colours to for the plots for each dataset.

        labels : list of str
            List of labels for each dataset.

        fillcols : list of colours
            List of colours to use for the shaded standard deviation regions.


        """
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='polar')
        degree_sign = u"\N{DEGREE SIGN}"

        def ellipse(phi,phis,thets,phi0,theta0):
            sqrt=np.sqrt(1-((phi-phi0)/phis)**2)
            return np.rad2deg(theta0-thets*sqrt),np.rad2deg(theta0+thets*sqrt)

        ns=np.logspace(1,5,5)

        for i in range(len(ds)):
            for n in ns:
                thets,phis,mthet,mphi,stdthet,stdphi,tthet,tphi,bthet,bphi=self.frac_fit(ds[i],n,bkgs[i],nfits)
                """
                mphi,mthet=np.mean(phi),np.mean(thet)
                stdphi,stdthet=np.std(phi),np.std(thet)
                tphi,tthet=np.deg2rad(mphi+3*stdphi),np.deg2rad(mthet+3*stdthet)
                bphi,bthet=np.deg2rad(mphi-3*stdphi),np.deg2rad(mthet-3*stdthet)
                mphi,mthet=np.deg2rad(mphi),np.deg2rad(mthet)
                stdphi,stdthet=np.deg2rad(stdphi),np.deg2rad(stdthet)
                """
                phirang=np.linspace(bphi,tphi,100)
                bthets,tthets=ellipse(phirang,3*stdphi,3*stdthet,mphi,mthet)
                ax.fill_between(phirang,bthets,tthets,color=fillcols[i],alpha=0.15,edgecolor=mcolor.to_rgba(colors[i]))
                if n==ns[-1]:
                    ax.scatter(mphi,np.rad2deg(mthet),c=colors[i],label=labels[i],s=20)
        
        labels=np.array([-80,'',-40,'',0,'',40,'',80])
        
        sourcethet=np.rad2deg(np.arctan2(-1*(91-27+12.5-37),95))
        source90thet=sourcethet
        sourcephi=np.deg2rad(90)
        source90phi=np.deg2rad(0)

        ax.scatter(sourcephi,sourcethet,marker='x',color='xkcd:blue',label='Source location, Position 1',s=100)
        #ax.scatter(source90phi,source90thet,marker='x',color='xkcd:red',label='Source location, Position 2',s=100)

        ll=np.array([str(label)+degree_sign for label in labels])
        ll[1::2]=''
        ll[4]=''
        ax.set_yticklabels(ll)
        ax.tick_params(labelsize=22)
        lgnd=plt.legend(loc='upper right',bbox_to_anchor=(1.6,1.15),fontsize=24)
        for i in range(len(ds)):
            lgnd.legendHandles[i]._sizes=[40]
        
        ax.tick_params(pad=20)
        ax.set_ylim(-90,90)
        fig.subplots_adjust(top=0.85)
        ax.set_rlabel_position(135)
        label_pos=ax.get_rlabel_position()
        ax.text(np.radians(label_pos+10),10,r'$\varphi$',ha='center',va='center',fontsize=24)
        ax.text(np.radians(label_pos-5),5,'0'+degree_sign,ha='center',va='center',fontsize=22)

        ax.text(0,135,r'$\theta$',ha='center',va='center',fontsize=24)
        ax.plot(np.linspace(0,2*np.pi,100),np.repeat(0,100),c='black',lw=2)
        #ax.set_xlim(np.deg2rad(-90),np.deg2rad(180))

        fig.set_tight_layout(True)
        plt.show()
            
    def face_counts(self,data,bkg=None,plot=False):
        """
        Calculates the count on each face and in the core 8 cubes. If plot is True, also plot a bar chart of this.

        PARAMETERS
        ------------------------
        data : str
            Label of the desired dataset, as saved in self.data.

        bkg : str or NoneType
            Label of the desired background dataset, as saved in self.data.

        plot : bool
            If True, plot the calculated counts as a bar chart.
        """
        data=self.data[data]
        cubes=data.query('Combine').loc[:,['CubeX','CubeY','CubeZ']]
        bars=np.array([len(cubes.query('CubeX==0'))/2,
                       len(cubes.query('CubeX==3'))/2,
                       len(cubes.query('CubeY==0'))/2,
                       len(cubes.query('CubeY==3'))/2,
                       len(cubes.query('CubeZ==0'))/2,
                       len(cubes.query('CubeZ==3'))/2,
                       len(cubes.query('CubeX>0&CubeX<3&CubeY>0&CubeY<3&CubeZ>0&CubeZ<3'))/2]).astype(int)
        if bkg is not None:
            bkg=self.data[bkg]
            bcubes=bkg.query('Combine').loc[:,['CubeX','CubeY','CubeZ']]
            bbars=np.array([len(bcubes.query('CubeX==0'))/2,
                            len(bcubes.query('CubeX==3'))/2,
                            len(bcubes.query('CubeY==0'))/2,
                            len(bcubes.query('CubeY==3'))/2,
                            len(bcubes.query('CubeZ==0'))/2,
                            len(bcubes.query('CubeZ==3'))/2,
                            len(bcubes.query('CubeX>0&CubeX<3&CubeY>0&CubeY<3&CubeZ>0&CubeZ<3'))/2]).astype(int)
            
            bgnorm=(bkg['time'].max()-bkg['time'].min())/1e8
            datnorm=(data['time'].max()-data['time'].min())/1e8

            norm=bgnorm/datnorm
            if norm < 1:
                bars=bars*norm-bbars
            else:
                bars=bars-bbars/norm

        if plot:
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.bar(np.arange(0,len(bars),1),bars,width=0.8)
            ss=['']+['X = 0','X = 3','Y = 0','Y = 3','Z = 0','Z = 3','Core']
            ax.set_xticklabels(ss,fontsize=24)
            ax.set_ylabel('Count',fontsize=24)
            plt.show()
        
        return bars




def en(x,a,b):
    e=abs(a)*np.exp(-x/b)
    return e
def ep(x,a,b):
    x=np.array(x)-3
    e=abs(a)*np.exp(x/b)
    return e