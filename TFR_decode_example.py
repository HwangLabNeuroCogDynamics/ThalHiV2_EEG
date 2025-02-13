####################################################################
# Script to run time frequency decomp then linear discrimnation analysis
####################################################################
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, cross_val_predict, KFold
from sklearn.model_selection import LeaveOneOut
from scipy.stats import zscore
from scipy.special import logit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datetime import datetime
import mne
from mne import io
from mne.time_frequency import tfr_morlet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#keily notes: next 2 lines are global variables:
n_jobs = 4
included_subjects = input()

classes = ['far','fab','fsr','fsb', 'dar','dsr','dab','dsb'] #conditions or categories
#----------setup trial ordrs (switch and stay trials)---------
#### potential hiearchical cue trees
#
#-- TEXTURE
#			  filled                        donut          # texture (filled, donut) dermine 2nd feature of interest
#			/	     \	                  /       \
#	   asterisk       star             red        blue
#      /      \      /     \           /  \       /   \
#     red    blue   red    blue   astrx  star  astrx  star  # s = star, a = asterisk
#      |       |     |       |        |    |     |     |
#     far     fab   fsr     fsb      dar  dsr   dab   dsb
#      |       |     |       |        |    |     |     |
#     face   face   scene  scene      F    F     S     S    # f = face task, s = scene task
#
#
#-- SHAPE
#			 asterisk                       star          # shape (star, asterisk) dermine 2nd feature of interest
#			/	     \	                 /        \
#	    filled        donut            red         blue
#      /      \      /     \          /   \       /    \
#     red    blue   red    blue    fill  donut  fill  donut   # s = star, a = asterisk
#      |       |     |       |       |    |       |     |
#     far     fab   dar     dab     fsr  dsr     fsb   dsb
#      |       |     |       |       |    |       |     |
#     face   face   scene   scene    F    F       S     S    # f = face task, s = scene task
#
#
#-- COLOR
#			   red                          blue          # color (red, blue) dermine 2nd feature of interest
#			/	     \	                 /        \
#	   asterisk       star           filled       donut
#      /     \       /     \          /   \      /    \
#    fill  donut   fill  donut    astrx  star  astrx  star      # s = star, a = asterisk
#     |       |     |       |        |    |      |     |
#    far     dar   fsr     dsr      fab  fsb    dab   dsb
#     |       |     |       |        |    |      |     |
#    face   face   scene   scene     F    F      S     S    # f = face task, s = scene task
#
#
#
# Total 8 cues, a short hand will be a three letter code
# 1st letter: texture, f= filled, d=donut, if cue is filled, focus on shape, if donut, focus on color
# 2nd letter: shape, s = star, a = asterisk. If texture is filled and shape is __, do the face task. If texture is filled & shape is __, do the scene task
# 3rd letter: color, r = red, b = blue. If texture is donut and color is red, do the face task. If texture is dont and color is blue, do the scene task
#
# For ease of tracking, we will give each individual cue an integer code
# 1: far  = filled asterisk with red   
# 2: fab  = filled asterisk with blue  
# 3: fsr  = filled star with red      
# 4: fsb  = filled star with blue      
# 5: dar  = donut asterisk with red    
# 6: dsr  = donut star with red        
# 7: dab  = donut asterisk with blue   
# 8: dsb  = donut star with blue    
# 
# - - - - - TRIAL STRUCTURE
#     cue ........ 0 ms (make cue onset the start of the trial)
#     delay1 ..... 500 ms
#     retro cue .. 1300 ms
#     delay2 ..... 1800 ms
#     probe ...... 3800 ms
#     feedback ... 5300 ms (lasts for 500 ms... trl end is 5800 ms)

#keily notes:
ROOT = '/data/backed_up/shared/ThalHiV2/EEG_data/'  #path where data files are stored. 
output_dir = '/data/backed_up/shared/ThalHiV2/EEG_data/tfr' # path to where to output data after analysis

# #plot the time frecuency representation:
# # Load the raw EEG data (adjust path based on actual data location)
# raw = mne.io.read_raw_fif('/data/backed_up/shared/ThalHiV2/EEG_data/subject1_raw.fif', preload=True)

# # Define events (adjust according to your actual event structure)
# events = mne.find_events(raw)

# # Define the event_id for the different conditions or categories
# event_id = {'far': 1, 'fab': 2, 'fsr': 3, 'fsb': 4, 'dar': 5, 'dsr': 6, 'dab': 7, 'dsb': 8}

# # Create epochs based on your event structure
#epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.5, tmax=2.0, baseline=(-0.5, 0), detrend=1, preload=True)

# Perform time-frequency decomposition using Morlet wavelets
# frequencies = np.arange(6, 30, 2)  # Define frequency bands (e.g., 6-30 Hz)
# time_bandwidth = 2.0  # Trade-off between time and frequency resolution
# n_cycles = frequencies / 2  # Number of cycles for each frequency

# # Perform the time-frequency decomposition on the epochs
# power = tfr_morlet(epochs, freqs=frequencies, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=False)

# # Plot the time-frequency representation (TFR)
# # Option 1: Topographic plot of the power (e.g., for a specific time)
# time_range = (0.5, 2.0)  # Define the time window to plot
# power.plot_topo(baseline=(-0.5, 0), mode='logratio', title="TFR Topo Plot", time_mask=time_range)

# # Option 2: Joint plot showing time-frequency data across electrodes (for a specific frequency)
# power.plot_joint(baseline=(-0.5, 0), mode='logratio', title="TFR Joint Plot")


#-------------------------------------------------------------------------------------
def mirror_evoke(ep):
#in short, this funtion , takes epoch data(ep) as input and returns the modfied epochs(e) and returns then
	e = ep.copy()
	nd = np.concatenate((np.flip(e._data[:,:,e.time_as_index(0)[0]:e.time_as_index(1.5)[0]], axis=2), e._data, np.flip(e._data[:,:,e.time_as_index(e.tmax-1.5)[0]:e.time_as_index(e.tmax)[0]],axis=2)),axis=2)
	#defined new time boundaries for the modified epoch
	tnmin = e.tmin - 1.5
	tnmax = e.tmax + 1.5
	e._set_times(np.arange(tnmin,tnmax+e.times[2]-e.times[1],e.times[2]-e.times[1]))
	e._data = nd

	return e


def run_TFR(sub): #sub stand fot subject number
    ''' run frequency decomp and save, sub by sub'''

    this_sub_path = ROOT + 'preproc/' # '/data/backed_up/shared/ThalHiV2/EEG_data/preproc/' points to where subject's data are stored 
    all_trl = mne.read_epochs(this_sub_path+"sub-"+sub+"_task-ThalHiV2_trl_eeg-epo.fif") # line reads the epoch data for the specified subject 
    all_trl.baseline = None
    all_trl = all_trl.crop(tmin = -1, tmax = 3.0) # crop from [-1  5.8] TO [-1  3]
    ###data is being saved to a CSV file 
    
    all_trl.metadata.to_csv((ROOT+'tfr/%s_metadata.csv' %sub))

    freqs = np.logspace(*np.log10([1, 40]), num=30) 
    n_cycles = np.logspace(*np.log10([3, 12]), num=30)

    tfr = tfr_morlet(mirror_evoke(all_trl), freqs=freqs,  n_cycles=n_cycles, 
                    average=False, use_fft=True, return_itc=False, decim=5, n_jobs=n_jobs)
    
    tfr = tfr.crop(tmin = -.8, tmax = 1.5) #trial by chn by freq by time, chop at 1.5s
    np.save((ROOT+'tfr/%s_times' %sub), tfr.times) #this saves the time points associated with the TFR
    tfr.save((ROOT+'tfr/%s_tfr.h5' %sub), overwrite=True)#this saves the TFR data to an HDF5 file

    return tfr # an object is returned as an object containing time and frecuency information for each epoch.





# - - - FOR DECODING ... still need to be modified
def run_classification(x_data, y_data, tfr_data, permutation=False):
	''' clasification analysis with LDA, using all freqs as features, so this is temporal prediction analysis (Fig 3A)'''

	# do this 10 times then average?
	lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto') #

	n_scores = np.zeros((y_data.shape[0],8,10))
	for n in np.arange(10):
		cv = KFold(n_splits=4, shuffle=True, random_state=6*n)

		if permutation:
			permuted_order = np.arange(x_data.shape[0])
			np.random.shuffle(permuted_order)
			xp_data = x_data[permuted_order,:,:]  #x_data is trial by chn by freq, we permute the trial order
			#need to vectorize data. Feature space is trial by ch by freq = 4xx * 64 * 20, need to wrap it into 4xx * 1280 (collapsing chn by freq)
			xp_data = zscore(np.reshape(xp_data, (tfr_data.shape[0], tfr_data.shape[1]*tfr_data.shape[2])))
			scores = cross_val_predict(lda, xp_data, y_data, cv=cv, method='predict_proba', pre_dispatch = n_jobs) #logit the probability
		else:
			#need to vectorize data. Feature space is trial by ch by freq = 4xx * 64 * 20, need to wrap it into 4xx * 1280 (collapsing chn by freq)
			xp_data = zscore(np.reshape(x_data, (tfr_data.shape[0], tfr_data.shape[1]*tfr_data.shape[2])))
			scores = cross_val_predict(lda, xp_data, y_data, cv=cv, method='predict_proba', pre_dispatch = n_jobs) #logit the probability
			n_scores[:,:,n] = scores

	#logit transform prob.
	n_scores = np.mean(n_scores,axis=2) # average acroos random CV runs
	n_scores = logit(n_scores) #logit transform probability
	n_scores[n_scores==np.inf]=36.8 #float of .9999999999xx
	n_scores[n_scores==np.NINF]=-36.8 #float of -.9999999999xx

	return n_scores


def run_full_TFR_classification(x_data, y_data, classes, permutation = False):
	''' clasification analysis with LDA, inputing one frequency at a time. Time-frequency prediction (Figure 3B)
	Results then feed to RSA regression (Figure 4)
	'''

	lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')

	if permutation:
		cv = KFold(n_splits=4, shuffle=True, random_state=np.random.randint(9)+1)
		n_scores = np.zeros((y_data.shape[0],len(classes)))
		permuted_order = np.arange(x_data.shape[0])
		np.random.shuffle(permuted_order)
		xp_data = x_data[permuted_order,:] # permuate trial, first dim
		scores = cross_val_predict(lda, xp_data, y_data, cv=cv, method='predict_proba', n_jobs = 1, pre_dispatch = 1)
		n_scores[:,:] = scores
	else:
		# do this 10 times then average?
		n_scores = np.zeros((y_data.shape[0],len(classes),10))
		for n in np.arange(10):
			cv = KFold(n_splits=4, shuffle=True, random_state=n*6)
			scores = cross_val_predict(lda, x_data, y_data, cv=cv, method='predict_proba', n_jobs = 1, pre_dispatch = 1)
			n_scores[:,:,n] = scores
		n_scores = np.mean(n_scores,axis=2) # average acroos random CV runs

	n_scores = logit(n_scores) #logit transform probability
	n_scores[n_scores==np.inf]=36.8 #float of .9999999999xx
	n_scores[n_scores==np.NINF]=-36.8 #float of -.9999999999xx

	return n_scores


def run_cue_prediction(tfr, permutation=False, full_TFR=True):
    # Cue classes for prediction
    cue_classes = ['far', 'fab', 'fsr', 'fsb', 'dar', 'dsr', 'dab', 'dsb']
    tfr_data = tfr.data

    # Initialize trial_prob based on conditions
    if permutation:
        num_permutations = 1000
        trial_prob = np.zeros((tfr_data.shape[0], tfr_data.shape[3], len(cue_classes), num_permutations))
    elif not full_TFR:
        trial_prob = np.zeros((tfr_data.shape[0], tfr_data.shape[3], len(cue_classes)))  # Trial x time x labels
    else:
        trial_prob = np.zeros((tfr_data.shape[0], tfr_data.shape[2], tfr_data.shape[3], len(cue_classes)))  # Trial x freq x time x labels

    # Iterate over time points
    for t in np.arange(tfr.times.shape[0]):
        y_data = tfr.metadata.cue.values.astype('str')  # Ensure proper indentation
        x_data = tfr_data[:, :, :, t]

        if permutation:
            for n_p in np.arange(num_permutations):
                n_scores = run_classification(x_data, y_data, tfr_data, permutation=True)
                trial_prob[:, t, :, n_p] = n_scores
        elif not full_TFR:
            n_scores = run_classification(x_data, y_data, tfr_data, permutation=False)
            trial_prob[:, t, :] = n_scores
        else:
            for f in np.arange(tfr_data.shape[2]):
                x_data = tfr_data[:, :, f, t]
                n_scores = run_full_TFR_classification(x_data, y_data, cue_classes)
                trial_prob[:, f, t, :] = n_scores

    # Save posterior probabilities
    if permutation:
        np.save(f"{ROOT}/decoding/{sub}_prob_permutation", trial_prob)
    else:
        np.save(f"{ROOT}/decoding/{sub}_tfr_prob", trial_prob)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - -    NOW ACTUALLY RUN THE CODE   - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
for sub in [included_subjects]:

	# datetime object containing current date and time
	now = datetime.now()
	print("starting time: ", now)
	# dd/mm/YY H:M:S
	# dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	# print("date and time =", dt_string)

	print('-------')
	print(('running subject %s' %sub))
	print('-------')

	#tfr = run_TFR(sub) # uncomment if need to regenerate the tfr 
	tfr = mne.time_frequency.read_tfrs((ROOT+'tfr/%s_tfr.h5' %sub))[0] 
	print("Metadata Columns:", tfr.metadata.columns)
	print("Metadata Preview:")
	print(tfr.metadata.head())

	now = datetime.now()
	print("TFR done at ", now)

	time_range = (0.5, 2.0)
	tfr.average().plot_topo(baseline=(-0.5, 0), mode='logratio', title="TFR Topo Plot")

	fig = tfr.average().plot_topo(baseline=(-0.5, 0), mode='logratio', title="TFR Topo Plot")
	fig.savefig('tfr_topo_plot.png')
	plt.close(fig)    
    
    # - - INSERT CODE FOR PLOTTING TFR DATA TO MAKE SURE IT WORKED
	#Plotting code:

	#def plot_avg_tfr(tfr, title='Average TFR'):

	###### Code for interactive power plots to visualize the data for one sensor. 
	##### You can also select a portion in the time-frequency plane to obtain a 
	##### topomap for a certain time-frequency region.
		
	"""
	Plot the average Time-Frequency Representation (TFR).

	Parameters:
	- tfr: MNE TFR object containing the time-frequency data.
	"""
	#tfr.avr.topo.plot()
	# Average across trials (assuming trials are the first dimension)
	
	if False: # just making it skip code temporarily... change to True to run again
		avg_tfr = tfr.average()  #createing average tfr object shape: (n_channels, n_freqs, n_times)

		power = avg_tfr.copy()

		power.plot_topo(baseline=(-0.5, 0), mode="logratio", title="Average power")

##########################Individual cue plots code#########################################

	import matplotlib.pyplot as plt
	import numpy as np

	# cue 
	#import tha data in the correct format. 
	# Replace with actual time array if available, or generate a placeholder
	# tfr.

	# assert probabilities.shape[0] == len(time),

	# cue_names = []  # cue names

	# # Plots: 
	# for i in range(probabilities.shape[1]):
	#     plt.figure(figsize=(6, 4))
	#     plt.plot(time, probabilities[:, i], label=cue_names[i]) 
	#     plt.xlabel('Time (ms)')
	#     plt.ylabel('Probability')
	#     plt.title(f'Probability of {cue_names[i]} Over Time')
	#     plt.grid(True)
	#     plt.legend()
	#     plt.show()



		#power.plot(picks=[64], baseline=(-0.5, 0), mode="logratio", title=power.ch_names[82])

	# 	fig, axes = plt.subplots(1, 2, figsize=(7, 4), layout="constrained")
	# 	topomap_kw = dict(
	# 		ch_type="grad", tmin=0.5, tmax=1.5, baseline=(-0.5, 0), mode="logratio", show=False
	# )
	# 	plot_dict = dict(Alpha=dict(fmin=8, fmax=12), Beta=dict(fmin=13, fmax=25))
	# 	for ax, (title, fmin_fmax) in zip(axes, plot_dict.items()):
	# 		power.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
	# 		ax.set_title(title)

	# 	plt.show()

	#########################################################################################################
	##### linear discrimination analysis on individual features from evoke data
	#########################################################################################################
	# run_evoke_dim_prediction(sub)
	# now = datetime.now()
	# print("feature prediction done at:", now)
	
	#########################################################################################################
	##### linear discrimination analysis on individual cues
	#########################################################################################################
	run_cue_prediction(tfr, permutation = False, full_TFR=False) #
	
	# plotting cue decoding results:
	now = datetime.now()
	print("Cue Prediction Done at:", now)
	trial_prob = np.load((ROOT+'/decoding/%s_tfr_prob' %sub)) #output form decoding 
	df = tfr.metadata

	print(df.columns)  # Check all column names in the metadata
	print(df.head())
	
	cue_epo_list = df['cue'] # NAME MIGHT NOT BE CORRECT
	cue_names = ['far','fab','fsr','fsb', 'dar','dsr','dab','dsb']
	for freq_i in range(probabilities.shape[1]): # loop through freq bins
		fig, axes = plt.subplot(1, 8, sharey=True)
		for cue_i in range(probabilities.shape[3]): # loop through cue objects
			current_trial_list = cue_epo_list == cue_names[cue_i] #trials for the current cue ; find lists of trials corresponding to the object
			avg_prob = np.mean(trial_prob[current_trial_list, :, :, :], axis=0) # freq x time x prob ;  for each cue pull out the probability for that specific cue. 
			axes[cue_i].plot(avg_prob[freq_i,:,cue_i], 'r') #, avg_prob[0,:,1], 'b') # indivisula plt plot being plotted, can be changed to plot all other individual cues as needed to see the decoding in action. 
		plt.show() 
	#goals is to plot all the 8 cues
	# pull out tfr.metadata (all information in epocs) - use to figure out which trial is which cue object. 
	# make this a for loop once we find the trials 
	#########################################################################################################
	##### linear discrimination analysis on texture, feature (color and shape), and task dimensions
	#########################################################################################################
	#run_dim_prediction(tfr, permutation = False)
	#run_dim_prediction(tfr, permutation = True)
	# now = datetime.now()
	# print("Dimension Prediction Done at:", now)

	# DoPermute = False
	# if DoPermute:
	# 	## run permtuations
	# 	now = datetime.now()
	# 	print("Starting cue permutation at:", now)
	# 	run_cue_prediction(tfr, permutation = True, full_TFR=False)
	# 	now = datetime.now()
	# 	print("Permute Cue Prediction Done at:", now)

	# DoDimPermute = True
	# if DoDimPermute:		
	# now = datetime.now()
	# print("Starting dimension permutation at:", now)	
	# run_dim_prediction(tfr, permutation = True)
	# now = datetime.now()
	# print("Dimension permutation done at:", now)	