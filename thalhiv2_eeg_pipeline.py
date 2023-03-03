"""
Quantum EEG Preprocessing script
    authors: Stephanie C Leach, Juniper Hollis, and Kai Hwang
    affiliations: University of Iowa, IA, Dept. of Psychological and Brain Sciences
    contact information:
        lab phone: (319) 467 - 0590
        lab email: psyc-hwang-lab@uiowa.edu
Overview of sections
    setup - generate paths and subject list and then loop through subjects
    1. load raw data
    2. high and low pass filter the data + re-reference
        * high pass of 0.1 Hz
        * low pass of 50 Hz
    3. plot and inspect filtered data for bad channels
        * manually mark bad channels to remove at this point
        * manually mark bad chunks of data as bad at this point
    4. run ICA on the copy of the data (task data only)
        * will view ICs and manually reject artefactual ICs
    5. re-reference (avg. re-ref) again and then epoch data
    6. inspect and reject remaining bad epochs (blinks, saccades, muscle, etc.)

dylan script link: https://github.com/HwangLabNeuroCogDynamics/TaskRep/blob/main/preprocess.py

"""
# Start by importing functions we will need
import mne
from mne.datasets import sample
from mne import io
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing import ICA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
import glob
import pickle
import fnmatch
import argparse
import datetime
from df2gspread import df2gspread as d2g
plt.ion()

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess EEG data from subject",
        usage="[subject] [OPTIONS] ... ",
    )
    parser.add_argument("subject", help="5-digit subject id if you only want to run 1 subject ... ALL if you want to run all subjects")
    #parser.add_argument("--subj", help="5-digit subject number if you only want to run 1 subject")
    parser.add_argument("--preproc",
                        help="preprocess EEG data, default is false",
                        default=False, action="store_true")
    parser.add_argument("--reinspect_epochs",
                        help="load and re-plot epochs for further inspection, default is false",
                        default=False, action="store_true")
    parser.add_argument("--gen_vis_erp_plots",
                        help="create and display primary visual erp plots, default is false",
                        default=False, action="store_true")
    parser.add_argument("--get_epoch_nums", 
                        help="get epoch numbers and add to google sheets file",
                        default=False, action="store_true")
    return parser

def generate_subj_list(subj_opt, data_path, filename_end_pattern):
    if subj_opt == "ALL":
        subjects = glob.glob( os.path.join( data_path,("sub-*_task-ThalHiV2_"+filename_end_pattern) ) )
    else:
        subjects = glob.glob( os.path.join( data_path,("sub-"+str(subj_opt)+"*task-ThalHiV2_"+filename_end_pattern) ) )
    return subjects

# ------------ Set Options --------------
#generate_plots = False
parser = init_argparse()
args = parser.parse_args(sys.argv[1:])
#generate_plots = args.generate_plots
subj_opt = args.subject


high_pass = 0.15 # in Hz
low_pass = 50.0 # in Hz
pre_ica_reject_dict = {'eeg': 450e-6, 'eog':500e-6, 'emg': 1000e-6} # in Volts (e-6 converts from microvolts to volts)

epo_baseline = None #(-0.8, -0.3) # in seconds
epo_reject_dict = {'eeg': 125e-6, 'emg':500e-6} # in Volts (e-6 converts from microvolts to volts)


# --------------- SETUP -----------------
output_path = '/data/backed_up/shared/ThalHiV2/EEG_data/'
raw_bids = '/data/backed_up/shared/ThalHiV2/EEG_data/BIDS/'
raw_behav = '/mnt/cifs/rdss/rdss_kahwang/ThalHi_data/v2_EEG_data/'

resp_keys = {'yes_key':['num_1',1],'no_key':['num_2',2]}
trigDict = {'startSaveflag':bytes([201]), 'stopSaveflag':bytes([255]), 'blockStart':202, 'blockEnd':203, 
                    'cue':{'far':111,'fab':113,'fsr':121,'fsb':123, 'dar':211,'dsr':221,'dab':213,'dsb':223}, 
                    'delay_1':131,   'retrocue':{'texture': 141, 'shape': 143, 'color': 145}, 
                    'delay_2':133,   'stim':{'Face':151, 'Scene':153}, 
                    'resp':{'correct':{resp_keys['yes_key'][1]:171, resp_keys['no_key'][1]:173}, 
                            'incorrect':{resp_keys['yes_key'][1]:175, resp_keys['no_key'][1]:177}},
                    'feedback':{'correct':181, 'incorrect':185},
                    'ITI': 191}

cue_codes = {'far':111,'fab':113,'fsr':121,'fsb':123, 'dar':211,'dsr':221,'dab':213,'dsb':223}
retrocue_codes = {'texture': 141, 'shape': 143, 'color': 145}
stim_codes = {'Face':151, 'Scene':153}
resp_codes =  {'correct/yes':171, 'correct/no': 173, 'incorrect/yes':175, 'incorrect/no':177}
feed_codes = {'correct':181, 'incorrect':185}
all_codes = {'far':111,'fab':113,'fsr':121,'fsb':123, 'dar':211,'dsr':221,'dab':213,'dsb':223, 
            'texture': 141, 'shape': 143, 'color': 145, 
            'Face':151, 'Scene':153, 
            'correct/yes':171, 'correct/no': 173, 'incorrect/yes':175, 'incorrect/no':177, 
            'correct':181, 'incorrect':185}

# ----------------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------------- 
# - - - - - - - - - - - - - - - - - - Preprocess Data - - - - - - - - - - - - - - - - - - - - - - 
# ----------------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------------- 
if args.preproc:
    # -- generate raw subjects list
    subjects = generate_subj_list(subj_opt, os.path.join(raw_bids,"sub-*","ses-01","eeg"), 'eeg.bdf')
    print(subjects)
    for raw_file in subjects:
        # pull out subject id number from raw file string
        sid = re.search("[0-9]{5}",raw_file)
        if sid:
            sub = sid.group(0)
        print("\ncurrently working on subject ", str(sub), "...")
        # pull out session number
        sid = re.search("ses-[0-2]{2}",raw_file)
        if sid:
            session_num = sid.group(0)
            session_num = int(session_num[-2:])
        print("\tsession ", str(session_num), "...\n")

        
        # load and add subject behavioral data
        if int(sub) == 10162:
            beh_files = glob.glob(os.path.join(raw_behav, ("sub-"+str(sub)+"_task-ThalHi_v2_block-00[1-4]_date-*.csv")))
        elif int(sub) == 10263:
            beh_files = glob.glob(os.path.join(raw_behav,("sub-"+str(sub)+"_task-ThalHiV2_block-00[3-7]_*.csv"))) # only use blocks 3-7
        elif int(sub) ==10287:
            beh_files = glob.glob(os.path.join(raw_behav,("sub-"+str(sub)+"_task-ThalHiV2_block-00[1-2]_*.csv")))
        elif int(sub) ==10292:
            beh_files = glob.glob(os.path.join(raw_behav,("sub-"+str(sub)+"_task-ThalHiV2_block-00[1-2]_*.csv")))
        elif int(sub) ==10218:
            beh_files = glob.glob(os.path.join(raw_behav,("sub-"+str(sub)+"_task-ThalHiV2_block-00[1-3]_*.csv")))
        elif int(sub) ==10305:
            beh_files = glob.glob(os.path.join(raw_behav,("sub-"+str(sub)+"_task-ThalHiV2_block-00[1-5]_*.csv")))
        else:
            beh_files = glob.glob(os.path.join(raw_behav,("sub-"+str(sub)+"_task-ThalHiV2_block-00[1-7]_*.csv")))
        beh_files = sorted(beh_files)
        print(beh_files)
        beh_df = pd.DataFrame() # create empty df to fill in
        for bf in beh_files:
            if int(sub) == 10264:
                if bf == beh_files[1]:
                    tmp_beh_df = pd.read_csv(bf)
                    tmp_beh_df.drop(range(33,72), inplace=True) # drop last half of 2nd block
                    print(tmp_beh_df)
                    beh_df = beh_df.append(tmp_beh_df) # add each block to full df
                else:
                    beh_df = beh_df.append(pd.read_csv(bf)) # add each block to full df
            elif int(sub) == 10263:
                if bf == beh_files[0]:
                    tmp_beh_df = pd.read_csv(bf)
                    tmp_beh_df.drop([0,1,2], inplace=True)
                    beh_df = beh_df.append(tmp_beh_df) # add each block to full df
                else:
                    beh_df = beh_df.append(pd.read_csv(bf)) # add each block to full df
            elif int(sub) == 10273:
                if bf == beh_files[4]:
                    tmp_beh_df = pd.read_csv(bf)
                    print(tmp_beh_df.loc[46, ["rt"]])
                    tmp_beh_df.loc[46, ["rt"]] = -1
                    tmp_beh_df.loc[46, ["subj_resp"]] = -1
                    beh_df = beh_df.append(tmp_beh_df) # add each block to full df
                else:
                    beh_df = beh_df.append(pd.read_csv(bf)) # add each block to full df
            else: 
                beh_df = beh_df.append(pd.read_csv(bf)) # add each block to full df
        print("current behavioral output file looks like...\n", beh_df, "\n")
        resp_df = beh_df[beh_df['subj_resp']!=-1] # reduce to just rows with responses
        # make a data frame with preproc parameters so we can save out a csv with details
        cur_csv = {'Participant_ID': sub, 'low_pass_filt': low_pass, 'high_pass_filt': high_pass, 'erp_baseline': str(epo_baseline), 'trl_baseline': 'None'}


        if not(os.path.exists( os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_eeg-preICA.fif")) )):
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # 1) load raw data and set channel types and montage
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            raw = mne.io.read_raw_bdf(raw_file, preload = True)
            raw.set_channel_types({'EXG1':'emg', 'EXG2':'emg', 'EXG3':'eog', 'EXG4':'eog', 'EXG5':'eog', 'EXG6':'eog', 'EXG7':'ecg', 'EXG8':'emg'})
            if int(sub) != 10162:
                raw.drop_channels(['EXG8'])
            if int(sub) == 10263:
                raw.crop(tmin=64.5) # crop out first 64.5 seconds
            if int(sub) == 10264:
                raw_p1 = raw.copy().crop(tmax=871)
                raw_p2 = raw.copy().crop(tmin=1092) # crop out time where it crashed during block 2
                raw, events = mne.concatenate_raws(raws=[raw_p1, raw_p2], events_list = [mne.find_events(raw_p1), mne.find_events(raw_p2)])
            if int(sub) == 10287:
                raw=raw.crop(tmax=1193)
            if int(sub) == 10292:
                raw=raw.crop(tmax=1192)
            if int(sub) == 10218:
                raw=raw.crop(tmax=1784)
            if int(sub) == 10305:
                raw=raw.crop(tmax=2982)
            raw.set_montage(montage = "biosemi64")
            # make a note of which EXG electrodes are eog and ecg for later
            EOG_channels=[['EXG3', 'EXG4'], ['EXG5', 'EXG6', 'FP1', 'FP2']]
            ECG_channels=['EXG7']
            # -- find events in the raw data file
            if int(sub) == 10273:
                events = mne.find_events(raw, shortest_event=1) # pull out the events (triggers) from the data file
                print(events.shape)
                events = np.delete(events, (2684), axis=0)
                print(events.shape)
                print(events[2683:26866,:])
            elif int(sub) != 10264:
                events = mne.find_events(raw) # pull out the events (triggers) from the data file
            # -- save out some raw data variables into our preprocessing csv file
            cur_csv['sampling_rate'] = raw.info['sfreq'] # get sampling rate
            cur_csv['ica_method'] = "input a copy of the continuous data with 1Hz highpass and 35Hz lowpass with no segment rejection THEN applied back to raw after selecting artefactual ICs"


            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # 2) Filter data
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            print('\n\n - - - - - Filtering Data - - - - - -\n')
            # -- filter the data
            raw.filter(l_freq=high_pass, h_freq=low_pass)
            eeg_filt_reref, _ = mne.set_eeg_reference(inst=raw, ref_channels='average') #['EXG1','EXG2'])
            # -- draw a psd plot so we can make sure there is no weird freq. noise (e.g., line noise)
            eeg_filt_reref.plot_psd(fmin=0.01, fmax=65.0)


            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # 3) Annotate block breaks, inspect channels, reject bad channels, and interpolate
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # -- automatically mark breaks between blocks
            dur_list = []
            break_onsets = []
            descript_list = []
            start_blk_trigs = events[(events[:,2] == 202)] # find starts of task blocks
            stop_blk_trigs = events[(events[:,2] == 203)] # find stops of task blocks
            print(start_blk_trigs)
            print(stop_blk_trigs)
            if int(sub) == 10263:
                # code didn't start properly on block 3
                for ind, cblkt in enumerate(stop_blk_trigs):
                    if (ind) < len(start_blk_trigs[:,0]):
                        cur_dur = (start_blk_trigs[(ind),0] - cblkt[0])/eeg_filt_reref.info['sfreq']
                    else:
                        cur_dur = 0
                    dur_list.append(cur_dur + 2) # get time inbetween stop of one block and start of the next and add 2 seconds to this duration
                    break_onsets.append(cblkt[0]/eeg_filt_reref.info['sfreq'] - 0.5) 
                    descript_list.append('bad_break')
            elif int(sub) == 10264:
                # code crashed halfway through block 2 and we just moved onto the next block after fixing
                adj_num=1
                for ind, cblkt in enumerate(stop_blk_trigs):
                    if ind == 1:
                        # add missing break
                        cur_dur = 2 # just set at 3 seconds
                        dur_list.append(cur_dur) # get time inbetween stop of one block and start of the next and add 2 seconds to this duration
                        break_onsets.append(start_blk_trigs[(ind+adj_num),0]/eeg_filt_reref.info['sfreq'] - 1) # add onset of next block start and set it to start 13 seconds before
                        descript_list.append('bad_break')
                        adj_num=2 # change to 2 for following block breaks
                    if (ind+adj_num) < len(start_blk_trigs[:,0]):
                        cur_dur = (start_blk_trigs[(ind+adj_num),0] - cblkt[0])/eeg_filt_reref.info['sfreq']
                    else:
                        cur_dur = 2
                    dur_list.append(cur_dur + 2) # get time inbetween stop of one block and start of the next and add 2 seconds to this duration
                    break_onsets.append(cblkt[0]/eeg_filt_reref.info['sfreq'] - 0.5) # add onset of block end to list and set start as half a second before the block end
                    descript_list.append('bad_break')
            else:
                for ind, cblkt in enumerate(stop_blk_trigs):
                    if (ind+1) < len(start_blk_trigs[:,0]):
                        cur_dur = (start_blk_trigs[(ind+1),0] - cblkt[0])/eeg_filt_reref.info['sfreq']
                    else:
                        cur_dur = 0
                    dur_list.append(cur_dur + 2) # get time inbetween stop of one block and start of the next and add 2 seconds to this duration
                    break_onsets.append(cblkt[0]/eeg_filt_reref.info['sfreq'] - 0.5) # add onset of block end to list and set start as half a second before the block end
                    descript_list.append('bad_break')
            print('break_onsets: ', break_onsets, '\ndur_list: ', dur_list, '\ndescrpt_list: ', descript_list)
            block_annots = mne.Annotations(onset=break_onsets, duration=dur_list, description=descript_list, orig_time=eeg_filt_reref.info['meas_date'])
            eeg_filt_reref.set_annotations(block_annots)
            print('\n\n - - - - - Inspecting for bad channels and segments of data - - - - - -\n')
            # -- plot the filtered data with events visible so we can inspect for bad channels and note if any artefacts seem to regularly happen around certain events
            eeg_filt_reref.plot(events=events, n_channels = 71, scalings= {'eeg': 20e-6, 'emg': 40e-6, 'eog': 20e-6}, block=True )
            i = input("Press Enter to Continue if you have finished selecting all bad channels and bad segments (if any): ")
            if eeg_filt_reref.info['bads']: 
                preICA_eeg = eeg_filt_reref.copy().interpolate_bads()
                for title, data in zip(['orig.', 'interp.'], [eeg_filt_reref, preICA_eeg]): 
                    with mne.viz.use_browser_backend('matplotlib'):
                        fig = data.plot(butterfly = True, color = '#00000022', bad_color = 'r')
                    fig.subplots_adjust(top = 0.9)
                    fig.suptitle(title, size = 'xx-large', weight = 'bold')
                cur_csv['bad_channels'] = ' '.join(eeg_filt_reref.info['bads'])
                i = input("Press Enter to Continue if the interpolation plots look OK: ")
            else:
                preICA_eeg = eeg_filt_reref.copy()
                cur_csv['bad_channels'] = ""
            # -- save out data at this point so if we want to change later parameters we can
            print(" \n\tsaving pre-ica data ...\n")
            preICA_eeg.save(fname = os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_eeg-preICA.fif")), overwrite=True)
            print(cur_csv)
            tmp_df = pd.DataFrame(cur_csv, index=[0])
            tmp_df.to_csv(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_preprocessingParameters.csv")), index=False)
            # --- epoch the data to the trial period (7 second epochs)
            #     cue ........ 0 ms (make cue onset the start of the trial)
            #     delay1 ..... 500 ms
            #     retro cue .. 1300 ms
            #     delay2 ..... 1800 ms
            #     stim ....... 3800 ms
            #     feedback ... 5300 ms (lasts for 500 ms... trl end is 5800 ms)
            # cue_inds = ( ((np.asarray(events[:,2]) > 110) & (np.asarray(events[:,2]) < 124)) | ((np.asarray(events[:,2]) > 210) & (np.asarray(events[:,2]) < 224)) )
            # cue_events = events[cue_inds]
            # print("total of ", len(cue_events), " cue events found\n")
            # if raw.info['bads']: 
            #     trl_epochs = mne.Epochs(raw = eeg_data_interp, events = cue_events, event_id = cue_codes, tmin = -1.0, tmax = 6.0, reject=pre_ica_reject_dict,
            #                     baseline = None, on_missing = 'warn', event_repeated = 'drop', metadata = beh_df, preload = True)
            # else:    
            #     trl_epochs = mne.Epochs(raw = raw, events = cue_events, event_id = cue_codes, tmin = -1.0, tmax = 6.0, reject=pre_ica_reject_dict,
            #                     baseline = None, on_missing = 'warn', event_repeated = 'drop', metadata = beh_df, preload = True)
        else:
            # load preICA file
            preICA_eeg = mne.io.read_raw_fif(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_eeg-preICA.fif")), preload = True)
            if int(sub) == 10273:
                events = mne.find_events(preICA_eeg, shortest_event=1) # pull out the events (triggers) from the data file
                events = np.delete(events, (2684), axis=0)
            else:
                events = mne.find_events(preICA_eeg) # pull out the events (triggers) from the data file
            cur_csv = {'Participant_ID': sub, 'low_pass_filt': low_pass, 'high_pass_filt': high_pass, 'erp_baseline': str(epo_baseline), 'trl_baseline': 'None', 'sampling_rate': preICA_eeg.info['sfreq']}
            cur_csv['ica_method'] = "input a copy of the continuous data with 1Hz highpass and 35Hz lowpass with no segment rejection THEN applied back to raw after selecting artefactual ICs"
            tmp_df = pd.read_csv(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_preprocessingParameters.csv")))
            cur_csv['bad_channels'] = list(tmp_df['bad_channels'])
            print(cur_csv)
            EOG_channels=['EXG3', 'EXG4', 'EXG5', 'EXG6']
            ECG_channels=['EXG7']


        if not(os.path.exists( os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_eeg-postICA.fif")) )):
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # 4) run ICA on the data
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            print('\n\n - - - - - Running ICA - - - - - -\n')
            # -- now that we have removed bad channels and re-referenced, make a copy with a 1Hz high pass filter for ICA
            print('\nmaking a copy of the data and applying a ** 1 Hz ** high pass and ** 35 Hz ** low pass filter for ICA\n')
            eeg_copy = preICA_eeg.copy().filter(l_freq=1.0, h_freq=35.0)
            eeg_copy.plot(events=events, n_channels=71, scalings= {'eeg': 20e-6, 'emg': 40e-6, 'eog': 20e-6}, block=True )
            i = input("Press Enter to Continue if the copy of the eeg data looks OK: ")
            # -- set up to run ICA on the copied data
            our_picks = mne.pick_types(eeg_copy.info, meg=False, eeg=True) #exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'])
            ICA_method = mne.preprocessing.ICA(method = 'infomax', max_iter=500, fit_params=dict(extended=True))
            # -- fit ICA
            print("\nfitting ICA to the copy of the data\nstarted running ICA on the data at ", datetime.datetime.now().strftime('%I:%M:%S %p'),"\n")
            ica_output = ICA_method.fit(eeg_copy, picks=our_picks, reject_by_annotation=True)
            # -- manually inspect and select artifactual ICs
            print("\nplotting the properties using the data after running the copy of the data through ICA\n")
            still_inspecting = True
            while still_inspecting:
                for pick_groups in [range(0,20), range(20,40), range(40, min(60, ica_output.n_components_))]:
                    mne.viz.plot_ica_sources(ica_output, picks=pick_groups, inst=preICA_eeg)
                    mne.viz.plot_ica_components(ica_output, picks=pick_groups, inst=preICA_eeg)
                print('The ICs marked for rejection are: ' + str(ica_output.exclude))
                i1 = input('Are you ready to move on to double checking that the bad IC selection was saved? [y/n]: ')
                if i1 == 'y':
                    still_inspecting = False
            still_selecting = True
            while still_selecting:
                ica_output.plot_components() # here is where we actually mark for rejection
                mne.viz.plot_ica_overlay(ica_output, inst=preICA_eeg, title='Signal before (in red) and after (in black) rejecting selected ICs')
                print('The ICs marked for rejection are: ' + str(ica_output.exclude))
                i2 = input('Are you sure you want to proceed? [y/n]: ' )
                if i2 == 'y':
                    cur_csv['bad_ICs'] = str(ica_output.exclude)
                    still_selecting = False
            # -- apply ICA back to original data
            ica_output.apply(preICA_eeg) 
            postICA_eeg = preICA_eeg.copy()
            postICA_eeg.plot(events=events, n_channels=71, scalings= {'eeg': 20e-6, 'emg': 40e-6, 'eog': 20e-6}, block=True )
            i = input("Press Enter to Continue if the post-ica data looks OK: ")
            # -- save out data at this point so if we want to change later parameters we can
            print(" \n\tsaving post-ica data ...\n")
            postICA_eeg.save(fname = os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_eeg-postICA.fif")), overwrite=True)
            print(cur_csv)
            cur_df = pd.DataFrame(cur_csv, index=[0])
            cur_df.to_csv(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_preprocessingParameters.csv")), index=False)
        else:
            # load postICA file
            postICA_eeg = mne.io.read_raw_fif(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_eeg-postICA.fif")), preload = True)
            if int(sub) == 10273:
                events = mne.find_events(postICA_eeg, shortest_event=1) # pull out the events (triggers) from the data file
                events = np.delete(events, (2684), axis=0)
            else:
                events = mne.find_events(postICA_eeg) # pull out the events (triggers) from the data file
            sampling_rate = postICA_eeg.info['sfreq'] # get sampling rate
            cur_csv = {'Participant_ID': sub, 'low_pass_filt': low_pass, 'high_pass_filt': high_pass, 'erp_baseline': str(epo_baseline), 'trl_baseline': 'None', 'sampling_rate': preICA_eeg.info['sfreq']}
            cur_csv['ica_method'] = "input a copy of the continuous data with 1Hz highpass and 35Hz lowpass with no segment rejection THEN applied back to raw after selecting artefactual ICs"
            tmp_df = pd.read_csv(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_preprocessingParameters.csv")))
            cur_csv['bad_channels'] = list(tmp_df['bad_channels'])
            cur_csv['bad_ICs'] = tmp_df['bad_ICs'].to_list()
            print(cur_csv)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 5) re-reference data to average of all electrodes
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        print('\n\n - - - - - Re-referencing Data - - - - - -\n\n')
        eeg_reref, _ = mne.set_eeg_reference(inst=postICA_eeg, ref_channels='average', ch_type='eeg')
        eeg_reref.plot_psd(fmin=0.01, fmax=55.0) # plot psd again
        eeg_reref.plot(events=events, n_channels=71, scalings= {'eeg': 20e-6, 'emg': 40e-6, 'eog': 520e-6}, block=True )
        i = input("Press Enter to Continue if the re-referenced data looks OK: ")
        # add behavioral data to eeg file 
        #raw_reref.metadata = beh_df.reset_index()


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 6) epoch data
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # - - get events from the data file
        #     * epoching to the following (as of 7-10-2022)
        #        1) cue onset
        #        2) stimulus onset
        #        3) response/fixation onset (feedback happens right after resp is made)
        print('\n\n - - - - - Epoching Data - - - - - -\n')
        all_inds = (np.asarray(events[:,2]) < 255)
        all_events = events[all_inds]
        # get events (triggers) related to the cue (trial onset)
        # cue_codes = {'far':111,'fab':113,'fsr':121,'fsb':123, 'dar':211,'dsr':221,'dab':213,'dsb':223}
        cue_inds = ( ((np.asarray(events[:,2]) > 110) & (np.asarray(events[:,2]) < 124)) | ((np.asarray(events[:,2]) > 210) & (np.asarray(events[:,2]) < 224)) )
        cue_events = events[cue_inds]
        print("total of ", len(cue_events), " cue events found\n")

        # get events (triggers) related to the retro cue
        # retrocue_codes = {'texture': 141, 'shape': 143, 'color': 145}
        retrocue_inds = ((np.asarray(events[:,2]) > 140) & (np.asarray(events[:,2]) < 146))
        retrocue_events = events[retrocue_inds]
        print("total of ", len(cue_events), " retro cue events found\n")

        # get events (triggers) related to the stimulus (face/scene)
        # stim_codes = {'Face':151, 'Scene':153}
        stim_inds = (((np.asarray(events[:,2]) == 151) | (np.asarray(events[:,2]) == 153)))
        stim_events = events[stim_inds]
        print("total of ", len(stim_events), " stim events found\n")

        # get events (triggers) related to the response onset
        # resp_codes =  {'correct/yes':171, 'correct/no': 173, 'incorrect/yes':175, 'incorrect/no':177}
        if (int(sub) != 10106):
            resp_inds = ((np.asarray(events[:,2]) > 170) & (np.asarray(events[:,2]) < 180))
            resp_events = events[resp_inds]
        else:
            copy_of_events = mne.find_events(eeg_reref)
            print(copy_of_events) # tuple where (sampling_rate, ?, trigger_code)... [(),(),(),...]
            resp_inds = ((np.asarray(events[:,2]) > 170) & (np.asarray(events[:,2]) < 180))
            resp_inds_tmp = [idx for idx, x in enumerate(resp_inds) if x]
            for bindx, cur_ri in enumerate(resp_inds_tmp):
                # change sampling rate value to what it should be based on BEH file
                cur_sr = copy_of_events[cur_ri,0]
                cur_ts = resp_df['rt'].tolist()[bindx]
                mod_val = (1.5-cur_ts)*sampling_rate # calculate how much to subtract from original sample point value (and make sure it is in sample rate NOT seconds)
                copy_of_events[cur_ri,0] = cur_sr - mod_val
            print(copy_of_events)
            resp_inds = ((np.asarray(copy_of_events[:,2]) > 170) & (np.asarray(copy_of_events[:,2]) < 180))
            resp_events = copy_of_events[resp_inds]
        print("total of ", len(resp_events), " resp events found\n")

        if int(sub) != 10162:
            # get events (triggers) related to the feedback onset
            # feedback_codes = {'correct':181, 'incorrect':185}
            feed_inds = ((np.asarray(events[:,2]) > 180) & (np.asarray(events[:,2]) < 190))
            feed_events = events[feed_inds]
            print("total of ", len(feed_events), " feed events found\n")

        # --- epoch the data to the trial period (7 second epochs)
        #     cue ........ 0 ms (make cue onset the start of the trial)
        #     delay1 ..... 500 ms
        #     retro cue .. 1300 ms
        #     delay2 ..... 1800 ms
        #     stim ....... 3800 ms
        #     feedback ... 5300 ms (lasts for 500 ms... trl end is 5800 ms)
        checking_auto_rej = True
        while checking_auto_rej:
            print('top of while loop')
            trl_epochs = mne.Epochs(raw = eeg_reref, events = cue_events, event_id = cue_codes, tmin = -1.0, tmax = 6.0, reject=epo_reject_dict,
                                    baseline = None, on_missing = 'warn', event_repeated = 'drop', metadata = beh_df, preload = True)
            trl_epochs.plot_drop_log() # so we can see if a certain channel is causing lots of data loss
            try:
                num_interp_already = len(str(cur_csv['bad_channels'][0]).split(' '))
            except:
                num_interp_already = 0
            new_bad_num = 7 # set high here so it stays in the while loop
            print("\nReminder!!! You have already interpolated " + str(num_interp_already) + " channels (" + str(cur_csv['bad_channels'][0]) + ")\n")
            i4 = input("Is there a channel causing lots of data loss that we should interpolate? [y/n]: ")
            if i4 == 'y':
                eeg_reref.plot(events=events, n_channels = 71, scalings= {'eeg': 20e-6, 'emg': 50e-6, 'eog': 50e-6}, block=True) # plot continuous data
                i = input("Press Enter to Continue if you are finished selecting extra channels to interpolate (REMINDER: you cannot interpolate more than 6 in total): ")
                new_bad_num = len(eeg_reref.info['bads'])
                if (num_interp_already+new_bad_num)>6:
                    print("\nReminder!!! You have already interpolated " + str(num_interp_already) + " channels (" + str(cur_csv['bad_channels'][0]) + ")\n")
                    print("\n * * * * * The code will not interpolate the selected channels because more than 6 in total have been selected * * * * *\n")
                elif eeg_reref.info['bads']: 
                    eeg_reref_interp = eeg_reref.copy().interpolate_bads()
                    eeg_reref_interp.plot(events=events, n_channels = 71, scalings= {'eeg': 20e-6, 'emg': 50e-6, 'eog': 50e-6}, block=True) # plot continuous data
                    check_choice = True
                    while check_choice:
                        i5 = input("Should we use the interpolated version for epoching? [y/n]: ")
                        if i5 == 'y':
                            cur_csv['bad_channels'] = (str(cur_csv['bad_channels'][0]) + ' ' + ' '.join(eeg_reref.info['bads']))
                            eeg_reref = eeg_reref_interp # make this our reref file
                            check_choice = False
                        elif i5 == 'n':
                            eeg_reref.info['bads'] = []
                            check_choice = False
                        else:
                            print('Unrecognized response... Please try again')
            else:
                checking_auto_rej = False
        print(cur_csv)
        cur_df = pd.DataFrame(cur_csv, index=[0])
        cur_df.to_csv(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_preprocessingParameters.csv")), index=False)

        # --- epoch trl epochs further into their specific events
        cue_epochs = mne.Epochs(raw = eeg_reref, events = cue_events, event_id = cue_codes, tmin = -0.8, tmax = 1.2, reject=epo_reject_dict,
                                baseline = None, on_missing = 'warn', event_repeated = 'drop', metadata = beh_df, preload = True)
        
        retrocue_epochs = mne.Epochs(raw = eeg_reref, events = retrocue_events, event_id = retrocue_codes, tmin = -1.0, tmax = 2.0, reject=epo_reject_dict,
                                baseline = None, on_missing = 'warn', event_repeated = 'drop', metadata = beh_df, preload = True)
    
        stim_epochs = mne.Epochs(raw = eeg_reref, events = stim_events, event_id = stim_codes, tmin = -0.8, tmax = 1.7, reject=epo_reject_dict,
                                baseline = None, on_missing = 'warn', event_repeated = 'drop', metadata = beh_df, preload = True)
        
        resp_epochs = mne.Epochs(raw = eeg_reref, events = resp_events, event_id = resp_codes, tmin = -0.8, tmax = 1.2, reject=epo_reject_dict,
                                baseline = None, on_missing = 'warn', event_repeated = 'drop', metadata = resp_df, preload = True)
            
        if int(sub) != 10162:
            feed_epochs = mne.Epochs(raw = eeg_reref, events = feed_events, event_id = feed_codes, tmin = -0.85, tmax = 0.65, reject=epo_reject_dict,
                                    baseline = None, on_missing = 'warn', event_repeated = 'drop', metadata = beh_df, preload = True)
            
            epo_dict = {'trl':trl_epochs, 'cue':cue_epochs, 'retro_cue':retrocue_epochs, 'stim':stim_epochs, 'resp':resp_epochs, 'feedback':feed_epochs}
        else:
            epo_dict = {'trl':trl_epochs, 'cue':cue_epochs, 'retro_cue':retrocue_epochs, 'stim':stim_epochs, 'resp':resp_epochs}
        
        if int(sub)==10279:
            epochs_currently_wanted = ['stim','resp']
        else:
            epochs_currently_wanted = ['trl','stim','resp']
        for cur_epo in epochs_currently_wanted:
            cur_epo_obj = epo_dict[cur_epo] # pull out current epoch object
            #cur_epo_obj.plot_drop_log() # see if a particular channel results in most of the epoch loss 
            rej_epo = True
            while rej_epo:
                #epo_events = mne.find_events(cur_epo_obj)
                mne.viz.plot_epochs(cur_epo_obj, picks='all', events=events, n_channels=71, scalings= {'eeg': 20e-6, 'emg': 40e-6, 'eog': 20e-6}, block=True )
                ie = input('\nAre you sure you want to proceed? [y/n]: ')
                if ie == 'y':
                    rej_epo = False
            cur_epo_obj.save(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_"+cur_epo+"_eeg-epo.fif")), overwrite=False)




# ----------------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------------- 
# - - - - - - - - - - - - - - - -     Num Usable Epochs     - - - - - - - - - - - - - - - - - - -
# ----------------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------------- 
epo_plot_dict = {'Usable_stim_epochs': 'stim', 'Usable_trl_epochs': 'trl', 'Usable_resp_epochs': 'resp'}
if args.get_epoch_nums:
    # -- get list of epoched subjects
    epo_subjects = generate_subj_list(subj_opt, os.path.join(output_path,"preproc"), 'stim_eeg-epo.fif')
    sub_list = []
    for epo_file in epo_subjects:
        sid = re.search("[0-9]{5}",epo_file) # pull out subject id number from raw file string
        if sid:
            sub_list.append(sid.group(0))
    print(sub_list)

    # load google sheet as a data frame
    sheet_name = "Preprocessing"
    doc_info="" # would normally contain link info BUT removing for github upload
    url = f"https://docs.google.com/spreadsheets/{doc_info}=out:csv&sheet={sheet_name}"
    prepro_df = pd.read_csv(url)
    print(prepro_df) # check that it loaded properly by viewing
    
    # -- now grab file and get usable epochs for each participant
    #epo_nums_df = pd.DataFrame({'Subject_ID':sub_list, 'Usable_trl_epochs':np.zeros(len(sub_list)), 'Usable_stim_epochs':np.zeros(len(sub_list)), 'Usable_resp_epochs':np.zeros(len(sub_list))})
    #print(epo_nums_df)
    for sub in sorted(sub_list):
        print("\ncurrently working on subject ", str(sub), "...")
        sub_idx=list(prepro_df['Subject_ID']).index(int(sub))
        for cur_epo_type in epo_plot_dict.keys():
            print("currently getting epoch numbers for "+cur_epo_type)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # 1) load raw data and set channel types and montage
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            epo_eeg = mne.read_epochs(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_"+epo_plot_dict[cur_epo_type]+"_eeg-epo.fif")), proj=True, preload = False)
            usable_epos = epo_eeg.selection
            num_usable_epos = usable_epos.shape[0]
            prepro_df[cur_epo_type][sub_idx] = num_usable_epos
    print(prepro_df)
    
    # upload changes to google sheet
    prepro_df.to_csv(os.path.join(output_path,"scripts","temp.csv"))
    #d2g.upload(prepro_df, gfile=url, wks_name=sheet_name) # would need to add credentials to actually update google sheet
            




# ----------------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------------- 
# - - - - - - - - - - - - - - - -     Re-inspect Epochs     - - - - - - - - - - - - - - - - - - -
# ----------------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------------- 
epo_plot_dict = {'stimulus': 'stim', 'cue': 'trl', 'response': 'resp'}
if args.reinspect_epochs:
    # -- get list of epoched subjects
    epo_subjects = generate_subj_list(subj_opt, os.path.join(output_path,"preproc"), 'stim_eeg-epo.fif')
    sub_list = []
    for epo_file in epo_subjects:
        sid = re.search("[0-9]{5}",epo_file) # pull out subject id number from raw file string
        if sid:
            sub_list.append(sid.group(0))
    print(sub_list)

    # -- now grab file and re-inspect for each participant
    for sub in sorted(sub_list):
        print("\ncurrently working on subject ", str(sub), "...")
        for cur_epo_type in epo_plot_dict.keys():
            print("currently re-plotting epoch plots for "+cur_epo_type+" epochs")
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # 1) load raw data and set channel types and montage
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            epo_eeg = mne.read_epochs(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_"+epo_plot_dict[cur_epo_type]+"_eeg-epo.fif")), proj=True, preload = False)
            
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # 2) re-plot epochs for inspection
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            print(epo_eeg.drop_log)
            epo_eeg.drop_log
            epo_eeg.plot_drop_log()
 
            bad_epos = []
            for ind,CE in enumerate(epo_eeg.drop_log):
                try: 
                    if CE[0] == 'USER':
                        bad_epos.append(ind)
                        epo_eeg.drop_log[ind] = epo_eeg.drop_log[0]
                except:
                    print('')

            print(epo_eeg.drop_log)
            epo_eeg.drop_log
            epo_eeg.plot_drop_log()

            rej_epo = True
            while rej_epo:
                mne.Epochs.plot(epo_eeg, n_channels=64, scalings= {'eeg': 40e-6, 'emg': 100e-6, 'eog': 75e-6})
                i = input("Press Enter to Continue: ")
                ie = input('\nAre you sure you want to proceed? [y/n]: ')
                if ie == 'y':
                    rej_epo = False
            epo_eeg.save(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_"+epo_plot_dict[cur_epo_type]+"2_eeg-epo.fif")), overwrite=True)

            epo_eeg.drop_bad()
            #epo_eeg.set_montage(montage = "biosemi64")
            #epo_events = mne.find_events(epo_eeg) # pull out the events (triggers) from the data file
            print(epo_eeg.info)
            epo_eeg.apply_baseline(baseline=(None,0.0))
            # -- set up for plotting with new rejections
            epo_df = epo_eeg.to_data_frame() # time will be in milliseconds and channel measurements in microvolts now
            # -- grab face vs scene trials
            stim_conds = []
            for sc in list(epo_df.condition):
                stim_conds.append(sc.split("/")[0])
            epo_df["target"] = stim_conds
            # -- grab occipital channels since it is a visual erp
            channels = ["P3","P5","P7","PO3","PO7","O1","O2","PO4","PO8","P8","P6","P4"]
            vis_erp_data = pd.melt(epo_df[(["time","target","epoch"]+channels)], id_vars=["time","target","epoch"], value_vars=channels)
            print(vis_erp_data)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # 3) generate basic visual erp plots of the data
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # -- actually generate and display the erp plot
            fig = plt.figure() # might not need this...
            sns.lineplot(x='time', y='value', hue='target', data=vis_erp_data[((vis_erp_data.time>-300) & (vis_erp_data.time<1000))])
            fig.suptitle(("sub-" + sub + " " + cur_epo_type + " onset"))
            plt.draw()
            i = input("Press Enter to Continue: ")



# ----------------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------------- 
# - - - - - - - - - - - - - - - - Generate Basic ERP Plots - - - - - - - - - - - - - - - - - - -
# ----------------------------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------------------------- 
# cue_codes = {'far':111,'fab':113,'fsr':121,'fsb':123, 'dar':211,'dsr':221,'dab':213,'dsb':223}
# retrocue_codes = {'texture': 141, 'shape': 143, 'color': 145}
# stim_codes = {'Face':151, 'Scene':153}
# resp_codes =  {'correct/yes':171, 'correct/no': 173, 'incorrect/yes':175, 'incorrect/no':177}
# feed_codes = {'correct':181, 'incorrect':185}
epo_plot_dict = {'stimulus': 'stim' , 'response': 'resp', 'cue': 'trl'}

if args.gen_vis_erp_plots:
    # -- get list of epoched subjects
    epo_subjects = generate_subj_list(subj_opt, os.path.join(output_path,"preproc"), 'stim_eeg-epo.fif')
    sub_list = []
    for epo_file in epo_subjects:
        sid = re.search("[0-9]{5}",epo_file) # pull out subject id number from raw file string
        if sid:
            sub_list.append(sid.group(0))
    print(sub_list)

    # -- epoch the data to the trial period (7 second epochs)
    #     cue ........ 0 ms (make cue onset the start of the trial)
    #     delay1 ..... 500 ms
    #     retro cue .. 1300 ms
    #     delay2 ..... 1800 ms
    #     stim ....... 3800 ms
    #     feedback ... 5300 ms (lasts for 500 ms... trl end is 5800 ms)
    for sub in sorted(sub_list):
        print("\ncurrently working on subject ", str(sub), "...")
        for cur_epo_type in epo_plot_dict.keys():
            print("currently generating ERP plots for "+cur_epo_type+" epochs")
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # 1) load raw data and set channel types and montage
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            epo_eeg_all = mne.read_epochs(os.path.join(output_path,"preproc",("sub-"+sub+"_task-ThalHiV2_"+epo_plot_dict[cur_epo_type]+"_eeg-epo.fif")), preload = False)
            epo_eeg = epo_eeg_all.copy()
            epo_eeg.drop_bad()
            #epo_eeg.set_montage(montage = "biosemi64")
            #epo_events = mne.find_events(epo_eeg) # pull out the events (triggers) from the data file
            print(epo_eeg.info)
            if cur_epo_type == 'stimulus':
                epo_eeg.apply_baseline(baseline=(None,0.0))
                # epo_eeg.crop(tmin=3.0, tmax=4.5) #(tmin=-1.0, tmax=4.5)
                # epo_eeg.apply_baseline(baseline=(3.0,3.8))
            elif cur_epo_type == 'feedback':
                epo_eeg.crop(tmin=4.5, tmax=6.0) #(tmin=-1.0, tmax=4.5)
                epo_eeg.apply_baseline(baseline=(4.5,5.0))
            elif cur_epo_type == 'response':
                epo_eeg.apply_baseline(baseline=(-0.75,-0.25))
            else: 
                epo_eeg.apply_baseline(baseline=(-0.8,0.))
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # 2) convert epoched file to a pandas dataframe format
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            epo_df = epo_eeg.to_data_frame() # time will be in milliseconds and channel measurements in microvolts now
            # -- grab face vs scene trials
            stim_conds = []
            for sc in list(epo_df.condition):
                stim_conds.append(sc.split("/")[0])
            epo_df["target"] = stim_conds
            # -- grab occipital channels since it is a visual erp
            if cur_epo_type == 'response':
                channels = ["FCz", "Fz", "FC1", "FC2", "Cz"] 
                # -- create data frame with just time points, target conditions, epochs, and occipital electrode average
                vis_erp_data = pd.melt(epo_df[(["time","target","epoch"]+channels)], id_vars=["time","target","epoch"], value_vars=channels)
            elif cur_epo_type == 'stimulus':
                channels = ["P3","P5","P7","PO3","PO7","O1","O2","PO4","PO8","P8","P6","P4"]
                # -- create data frame with just time points, target conditions, epochs, and occipital electrode average
                # BELOW COMMENT MAKES COOL PLOTS BUT WE WON'T USE THEM FOR NOW ...
                # av1 = epo_eeg["stimulus == 'Face'"].average()
                # av2 = epo_eeg["stimulus == 'Scene'"].average()
                # joint_kwargs = dict(ts_args=dict(time_unit='s'),
                #                     topomap_args=dict(time_unit='s'))
                # av1.plot_joint(show=False, **joint_kwargs)
                # av2.plot_joint(show=False, **joint_kwargs)
                # evokeds = dict()
                # query = "stimulus == '{}'"
                # for cur_stim in epo_eeg.metadata['stimulus'].unique():
                #     evokeds[str(cur_stim)] = epo_eeg[query.format(cur_stim)].average()
                # mne.viz.plot_compare_evokeds(evokeds, cmap=('stimulus category', 'viridis'), picks=channels) 
                # i = input("Press Enter to Continue: ")
                vis_erp_data = pd.melt(epo_df[(["time","target","epoch"]+channels)], id_vars=["time","target","epoch"], value_vars=channels)
            else: 
                channels = ["FCz", "Fz", "FC1", "FC2", "Cz"] 
                # -- create data frame with just time points, target conditions, epochs, and occipital electrode average
                vis_erp_data = pd.melt(epo_df[(["time","target","epoch"]+channels)], id_vars=["time","target","epoch"], value_vars=channels)
            print(vis_erp_data)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # 3) generate basic visual erp plots of the data
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # -- actually generate and display the erp plot
            fig = plt.figure() # might not need this...
            sns.lineplot(x='time', y='value', hue='target', data=vis_erp_data[((vis_erp_data.time>-300) & (vis_erp_data.time<1000))])
            fig.suptitle(("sub-" + sub + " " + cur_epo_type + " onset"))
            plt.draw()
            i = input("Press Enter to Continue: ")
