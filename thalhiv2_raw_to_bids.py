import mne
from mne_bids import BIDSPath, write_raw_bids
import glob
import os
from thalpy import base
import numpy as np


def generate_events(raw):
    events_data = mne.find_events(raw, shortest_event=1)

    eligible_events = set(trigDict.values())
    ev_data_excludeIneligibleEvents = []
    print(events_data)
    for ev in events_data:
        # print(ev)
        if ev[2] in eligible_events:
            ev_data_excludeIneligibleEvents.append(ev)

    return np.asarray(ev_data_excludeIneligibleEvents)


raw_eeg_dir = '/mnt/cifs/rdss/rdss_kahwang/ThalHi_data/v2_EEG_data/Raw/'
bids_dir = '/data/backed_up/shared/ThalHiV2/EEG_data/BIDS/'

trigDict = {'startSaveflag':bytes([201]), 'stopSaveflag':bytes([255]), 
            'blockStart':202, 'blockEnd':203, 
            'far':111,'fab':113,'fsr':121,'fsb':123, 'dar':211,'dsr':221,'dab':213,'dsb':223, 
            'delay_1':131, 
            'texture': 141, 'shape': 143, 'color': 145, 
            'delay_2':133,   
            'Face':151, 'Scene':153, 
            'correct/yes':171, 'correct/no': 173, 'incorrect/yes':175, 'incorrect/no':177,
            'correct':181, 'incorrect':185,
            'ITI': 191}


# convert to bids format for eeg data not already in bids dir
os.chdir(raw_eeg_dir)
bdf_files = glob.glob('sub-*_task-ThalHi*_eeg_*.bdf')
print('\n\n- - - - - - - Converting EEG files to BIDS format - - - - - - -\n\n')
print('list of files: \n', bdf_files)
for bdf_file in bdf_files:
    subject = base.parse_sub_from_file(bdf_file, prefix='sub-')
    if int(subject) == 10263:
        bdf_file = glob.glob('sub-*_task-ThalHi*_eeg2_*.bdf')[0]

    print('\ncurrent file: ', bdf_file)
    raw = mne.io.read_raw_bdf(bdf_file)
    if 'task-ThalHiV2' in bdf_file:
        task = 'ThalHiV2'
    if 'session-001' in bdf_file:
        session = '01'
    if  'session-002' in bdf_file:
        session = '02'
    event_data = generate_events(raw)
    bids_path = BIDSPath(subject=subject, task=task,  datatype='eeg', session=session,
                         root=bids_dir)
    write_raw_bids(raw, bids_path=bids_path, overwrite=True,
                   events_data=event_data, event_id=trigDict)
