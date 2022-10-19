#!/usr/bin/env python
# -*- coding: utf-8 -*-

## June 17, 2022: adaption for BEH and EEG, adding retro cue (and delays) and two hierarchical trees

"""
Authors: Stephanie C. Leach, Kai Hwang.
University of Iowa, Iowa City, IA
Hwang Lab, Dpt. of Psychological and Brain Sciences
As of June. 15, 2022:
    Office:355N PBSB
    Office Phone:319-467-0610
    Fax Number:319-335-0191
    Lab:355W PBSB
Lab Contact Info: 
    Web - https://kaihwang.github.io/
    Email - kai-hwang@uiowa.edu

This version of the Psychopy script is designed to mimic and model functions
written from the psychopy builder

However, this script has been heavily edited with python code non-specific to the builder.
Take this into account when reading through this as script design here does not mirror builder
script layout. 

The goal of this script is to run a task ("ThalHi") that first presents a filled or not filled shape,
then presents a picture of a face or a scene. Subjects respond with either 1 or 0 on the keyboard. 
See decision tree below in the script for more insight on our task-switch.

EDITED FOR EEG ONLY
"""

from __future__ import absolute_import, division, print_function
from psychopy import locale_setup, sound, gui, visual, core, data, event, logging, clock, info
#from psychopy.event import globalKeys
from psychopy.hardware.emulator import launchScan # Add and customize this www.psychopy.org/api/hardware/emulator.html
from psychopy.hardware import keyboard

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy.random import random, randint, normal #, shuffle
from random import choice as randomchoice
from random import shuffle as Shuffle
import os  # handy system and path functions
import sys  # to get file system encoding
import glob # this pulls files in directories to create Dictionaries/Str
import pandas as pd #Facilitates data structure and analysis tools. prepend 'np.'
import pyglet as pyg
import copy #Incorporated for randomization of stimuli
import csv #for export purposes and analysis
from datetime import datetime, timedelta
import serial
from PIL import Image

#-----------Notes on Script Initiation----------
# This script calls on the directory it is housed in 
# as well as the folder containing the correct outputs
# Ensure this is addressed appropriatley before running the script
#-----------Change Directory------
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))#.decode(sys.getfilesystemencoding())
os.chdir(_thisDir)


testing_code = 0


######################################################################################################################
##################################################   Get Subj Info  ##################################################
# load subject info sheet to get and save gui info
subj_info_df = pd.read_csv("ThalHi_v2_Subject_Info.csv")
# find row (aka ID) to enter data (or pull version info from)
for ind, date_cell in enumerate(subj_info_df["SessionDate"]): # loop through the date column (see if date is empty or matches today's date)
    if type('str') != type(date_cell):
        cur_subid = subj_info_df.loc[ind, "Participant_ID"] # means no data here and we can use this row
        if np.isnan(cur_subid):
            cur_subid = 99999
        print('\tnan detected','\t entering data in row',ind)
        new_entry = 1
        break
    else: 
        if datetime.strptime(date_cell,'%m/%d/%Y') == datetime.today().strftime('%m/%d/%Y'): # if date matches today's date
            cur_ST = (datetime.strptime(subj_info_df.loc[ind,'StartTime'][0:5],'%I:%M') + timedelta(hours=2)).strftime('%I:%M')
            if datetime.now().time().strftime('%I:%M') < (): # check if time is within 2 hours
                cur_subid = int(subj_info_df.loc[ind, "Participant_ID"])
                break

#ind = len(subj_info_df["Participant_ID"]) # set row as length of rows plus one (avoid overwriting data)
version = subj_info_df.loc[ind, 'Version'] # pull out version for this subject
respord = subj_info_df.loc[ind, 'Response_Order']
# Store info about the experiment session
expName = 'ThalHiV2'  # from the Builder filename that created this script
expInfo = {'Participant_ID': int(cur_subid),  'Version': version, 'Response_Order': respord, 'Method': ['EEG','BEH'],
           'Gender': ['Male','Female','Non-Binary','Other','Prefer not to say'], 'Age': 0} 
# VERSION:  FS = Filled & Shape ... FC = Filled & Color ... SC = Shape & Color

dlg = gui.DlgFromDict(dictionary=expInfo, title=expName, sortKeys=False) # Gui grabs the dictionary to create a pre-experiment info deposit
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['expName'] = expName

if new_entry:
    # SAVE OUT ENTERED INFO BEFORE STARTING TASK
    subj_info_df.loc[ind, 'Participant_ID'] = expInfo['Participant_ID']
    subj_info_df.loc[ind, 'SessionDate'] = datetime.today().strftime('%m/%d/%Y') # add a simple timestamp
    subj_info_df.loc[ind, 'StartTime'] = datetime.today().strftime('%I:%M:%S %p')
    subj_info_df.loc[ind, 'Gender'] = expInfo['Gender']
    subj_info_df.loc[ind, 'Age'] = expInfo['Age']
    #print(subj_info_df)
    if expInfo['Participant_ID'] != 99999:
        subj_info_df.to_csv("ThalHi_v2_Subject_Info.csv", index=False)

######################################################################################################################
#----------------setup windows and display objects--------
##### Setup the display Window
if expInfo['Method']=='EEG':
    win = visual.Window(size=(1280, 800), fullscr=True, screen=0, allowGUI=False, allowStencil=False, units='deg',
                        monitor='testMonitor', color=[0,0,0], colorSpace='rgb', blendMode='avg', useFBO=True)
elif expInfo['Method']=='BEH':
    win = visual.Window(size=(1920, 1080), fullscr=True, screen=0, allowGUI=False, allowStencil=False, units='deg',
                        monitor='testMonitor', color=[0,0,0], colorSpace='rgb', blendMode='avg', useFBO=True)

##### Temporal parameters.
expInfo['frameRate'] = win.getActualFrameRate() # store frame rate of monitor if we can measure it
if expInfo['frameRate'] != None:
    frame_rate = round(expInfo['frameRate'])  # frame_rate_def = 1.0 / round(expInfo['frameRate'])
else:
    frame_rate = 60.0  # could not measure, so default 60 Hz   #frame_rate_def = 1.0 / 60.0 
# Add multiple shutdown keys "at once".
for key in ['q', 'escape']:
    event.globalKeys.add(key, func=core.quit)
    
timing_test = 0
skip_tutorial = 1

win.mouseVisible = False
######################################################################################################################
#------------Initialize Variables------
##### Response keys and keyboard
if expInfo['Response_Order'] == "yes=1":
    resp_keys = {'yes_key':['num_1',1],'no_key':['num_2',2]} # <- EEG room keyboard codes
    #resp_keys = {'yes_key':['end',1],'no_key':['down',2]} # <- BEH room keyboard IF version 3.2.0
elif expInfo['Response_Order'] == "yes=2":
    resp_keys = {'no_key':['num_1',1],'yes_key':['num_2',2]} # <- EEG room keyboard codes
kb = keyboard.Keyboard() # keyboard has better timing that other keypress functions

##### Task Parameters
# initialize retro cues
texture_text = visual.TextStim(win=win, text=u'texture', units='norm', font=u'Arial', 
    pos=(0, 0), height=0.09, wrapWidth=None, ori=0, color=u'black', colorSpace='rgb', opacity=1)
shape_text = visual.TextStim(win=win, text=u'shape', units='norm', font=u'Arial',
    pos=(0, 0), height=0.09, wrapWidth=None, ori=0, color=u'black', colorSpace='rgb', opacity=1)
color_text = visual.TextStim(win=win, text=u'color', units='norm', font=u'Arial',
    pos=(0, 0), height=0.09, wrapWidth=None, ori=0, color=u'black', colorSpace='rgb', opacity=1)

version_retrocues = {'FS':['texture', 'shape'], 'FC':['texture', 'color'], 'SC':['shape', 'color']}
retrocues = version_retrocues[version] # pull out retrocues based on version
retrocue_textobj = {'texture': texture_text, 'shape': shape_text, 'color': color_text}

#### Timing and trial keeping variables
# ----------- BEHAVIORAL --------------
# -- Cue Obj   500 ms
# -- Delay_1   800 ms
# -- Retrocue  500 ms
# -- Delay_2   1500 ms | 1750 ms | 2000 ms
# -- Stimuli   1500 ms
# -- ITI       1000 ms 
# -------------------------------------
# -------------  EEG  -----------------
# -- Cue Obj   500 ms
# -- Delay_1   800 ms
# -- Retrocue  500 ms
# -- Delay_2   2000 ms
# -- Stimuli   1500 ms
# -- ITI       2000 - 3000 ms (variable)
# -------------------------------------
if expInfo['Method']=='EEG':
    Task_Parameters = {'n_blocks': 7, 'n_trials': 72, 'retrocues':version_retrocues[version], 'retro_freq':[[.5, .5, .5, .5, .5, .5, .5, .5, .5], [.5, .5, .5, .5, .5, .5, .5, .5, .5]]}
    Prac_Task_Parameters = {'n_blocks': 1, 'n_trials': 24, 'retrocues':version_retrocues[version], 'retro_freq':[[.5, .5, .5], [.5, .5, .5]]}
    Shuffle(Task_Parameters['retro_freq'])
    
    Trl_Durs = {'cue': 0.5, 'delay_1': 0.8, 'retrocue': 0.5, 'delay_2': np.asarray([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]), 'stim': 1.5, 'ITI': np.asarray([2.0, 2.1, 2.15, 2.2, 2.25, 2.3, 2.4, 2.5, 2.6, 2.8, 3.0])}
    Prac_Trl_Durs = {'cue': 0.5, 'delay_1': 0.8, 'retrocue': 0.5, 'delay_2': np.asarray([3.5, 3.5, 2.50]), 'stim': 1.5, 'ITI': np.asarray([2.0, 2.1])}
    refresh_rate=expInfo['frameRate']
    port=serial.Serial('COM6',baudrate=115200)
    port.close()
    # cue triggers: ### = [1=f|2=d][1=a|2=s][1=r|3=b]
    trigDict = {'startSaveflag':bytes([201]), 'stopSaveflag':bytes([255]), 'blockStart':202, 'blockEnd':203, 
                    'cue':{'far':111,'fab':113,'fsr':121,'fsb':123, 'dar':211,'dsr':221,'dab':213,'dsb':223}, 
                    'delay_1':131,   'retrocue':{'texture': 141, 'shape': 143, 'color': 145}, 
                    'delay_2':133,   'stim':{'Face':151, 'Scene':153}, 
                    'resp':{'correct':{resp_keys['yes_key'][1]:171, resp_keys['no_key'][1]:173}, 
                            'incorrect':{resp_keys['yes_key'][1]:175, resp_keys['no_key'][1]:177}},
                    'feedback':{'correct':181, 'incorrect':185},
                    'ITI': 191}
 
elif expInfo['Method']=='BEH':
    Task_Parameters = {'n_blocks': 5, 'n_trials': 104, 'retrocues':version_retrocues[version], 'retro_freq':[[.5, .8, .8, .2, .2], [.5, .2, .2, .8, .8]]}
    Prac_Task_Parameters = {'n_blocks': 1, 'n_trials': 24, 'retrocues':version_retrocues[version], 'retro_freq':[[.5, .5, .5], [.5, .5, .5]]}
    # shuffle list of retro_freq so participants get diff order
    Shuffle(Task_Parameters['retro_freq'])
    
    Trl_Durs = {'cue': 0.5, 'delay_1': 0.8, 'retrocue': 0.5, 'delay_2': np.asarray([1.5, 1.75, 2.0, 1.5, 1.75]), 'stim': 1.5, 'ITI': np.asarray([2.0, 2.1, 2.2, 2.3, 2.4, 2.5])}
    Prac_Trl_Durs = {'cue': 0.5, 'delay_1': 0.8, 'retrocue': 1.0, 'delay_2': np.asarray([4.50, 4.50, 2.00]), 'stim': 1.5, 'ITI': np.asarray([2.0, 2.1])}


################################################################################################################################
##################################################       Set up trials       ##################################################
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

##### Setup Cue stim objects
#size=(0.5)
#filled star 
filled_star_red = visual.ImageStim(
    win=win, image=os.getcwd()+'/cue_objects/filled_star_red_3by3.png',name='filled_star_red', units='norm', pos=(0,0))#, size=size) 
filled_star_blue = visual.ImageStim(
    win=win, image=os.getcwd()+'/cue_objects/filled_star_blue_3by3.png',name='filled_star_blue', units='norm', pos=(0,0))#, size=size) 

#filled asterisk
filled_asterisk_red = visual.ImageStim(
    win=win, image=os.getcwd()+'/cue_objects/filled_asterisk_red_3by3.png',name='filled_asterisk_red', units='norm', pos=(0,0))#,size=size) #size=[vis_deg_poly,vis_deg_poly],
filled_asterisk_blue = visual.ImageStim(
    win=win, image=os.getcwd()+'/cue_objects/filled_asterisk_blue_3by3.png',name='filled_asterisk_blue', units='norm', pos=(0,0))#,size=size) #size=[vis_deg_poly,vis_deg_poly],

#donut star
donut_star_red = visual.ImageStim(
    win=win,  image=os.getcwd()+'/cue_objects/donut_star_red_3by3.png',name='donut_star_red', units='norm', pos=(0,0))#,size=size)
donut_star_blue = visual.ImageStim(
    win=win,  image=os.getcwd()+'/cue_objects/donut_star_blue_3by3.png',name='donut_star_blue', units='norm', pos=(0,0))#,size=size ) 

#donut asterisk
donut_asterisk_blue = visual.ImageStim(
    win=win,image=os.getcwd()+'/cue_objects/donut_asterisk_blue_3by3.png', name='donut_asterisk_blue', units='norm',  pos=(0,0))#,size=size) #size=[vis_deg_poly,vis_deg_poly],
donut_asterisk_red = visual.ImageStim(
    win=win,image=os.getcwd()+'/cue_objects/donut_asterisk_red_3by3.png', name='donut_asterisk_red', units='norm',  pos=(0,0))#,size=size) #size=[vis_deg_poly,vis_deg_poly],

white_box = visual.Polygon(win, edges=4, radius=0.5, units='deg', lineWidth=1.5, lineColor='white', fillColor='white', pos=(-25, 15), size=5.0, contrast=1.0, colorSpace='rgb')
horz_line = visual.Rect(win=win, units='deg', pos=(0,0), size=[1.5, 0.5], color='black', fillColor='black', colorSpace='rgb')
vert_line = visual.Rect(win=win, units='deg', pos=(0,0), size=[0.5, 1.5], color='black', fillColor='black', colorSpace='rgb')

##### create a attribute dictionary saving all cue types
# note, copying the shapestim and visual.circle object using copy.copy() turns out to be critical, otherwise can't change colors on the fly
tree_imgs = {'texture':{'fsr':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/texture/fsr.png', units='norm', pos=(0,0)),
                        'fsb':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/texture/fsb.png', units='norm', pos=(0,0)),
                        'far':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/texture/far.png', units='norm', pos=(0,0)),
                        'fab':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/texture/fab.png', units='norm', pos=(0,0)),
                        'dsr':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/texture/dsr.png', units='norm', pos=(0,0)),
                        'dsb':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/texture/dsb.png', units='norm', pos=(0,0)),
                        'dar':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/texture/dar.png', units='norm', pos=(0,0)),
                        'dab':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/texture/dab.png', units='norm', pos=(0,0)),},
            'shape':{'fsr':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/shape/fsr.png', units='norm', pos=(0,0)),
                        'fsb':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/shape/fsb.png', units='norm', pos=(0,0)),
                        'far':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/shape/far.png', units='norm', pos=(0,0)),
                        'fab':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/shape/fab.png', units='norm', pos=(0,0)),
                        'dsr':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/shape/dsr.png', units='norm', pos=(0,0)),
                        'dsb':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/shape/dsb.png', units='norm', pos=(0,0)),
                        'dar':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/shape/dar.png', units='norm', pos=(0,0)),
                        'dab':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/shape/dab.png', units='norm', pos=(0,0)),},
            'color':{'fsr':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/color/fsr.png', units='norm', pos=(0,0)),
                        'fsb':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/color/fsb.png', units='norm', pos=(0,0)),
                        'far':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/color/far.png', units='norm', pos=(0,0)),
                        'fab':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/color/fab.png', units='norm', pos=(0,0)),
                        'dsr':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/color/dsr.png', units='norm', pos=(0,0)),
                        'dsb':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/color/dsb.png', units='norm', pos=(0,0)),
                        'dar':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/color/dar.png', units='norm', pos=(0,0)),
                        'dab':visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/color/dab.png', units='norm', pos=(0,0)),}}

Prac_Cue_types = {'fsr': {'cue':'fsr', 'Color':'red',  'Texture':'Filled',  'Shape':'Star',     'Task':{'texture':'Scene', 'shape':'Face',  'color':'Scene'}, 'cue_stim': copy.copy(filled_star_red) },
                 'fsb': {'cue':'fsb', 'Color':'blue', 'Texture':'Filled',   'Shape':'Star',     'Task':{'texture':'Scene', 'shape':'Scene', 'color':'Face'}, 'cue_stim': copy.copy(filled_star_blue) },
                 'far': {'cue':'far', 'Color':'red',  'Texture':'Filled',   'Shape':'Asterisk', 'Task':{'texture':'Face',  'shape':'Face',  'color':'Face'}, 'cue_stim': copy.copy(filled_asterisk_red) },
                 'fab': {'cue':'fab', 'Color':'blue', 'Texture':'Filled',   'Shape':'Asterisk', 'Task':{'texture':'Face',  'shape':'Face',  'color':'Face'}, 'cue_stim': copy.copy(filled_asterisk_blue) },
                 'dsr': {'cue':'dsr', 'Color':'red',  'Texture':'Outline',  'Shape':'Star',     'Task':{'texture':'Face',  'shape':'Face',  'color':'Scene'}, 'cue_stim': copy.copy(donut_star_red) },
                 'dsb': {'cue':'dsb', 'Color':'blue', 'Texture':'Outline',  'Shape':'Star',     'Task':{'texture':'Scene', 'shape':'Scene', 'color':'Scene'}, 'cue_stim': copy.copy(donut_star_blue) },
                 'dab': {'cue':'dab', 'Color':'blue', 'Texture':'Outline',  'Shape':'Asterisk', 'Task':{'texture':'Scene', 'shape':'Scene', 'color':'Scene'}, 'cue_stim': copy.copy(donut_asterisk_blue) },
                 'dar': {'cue':'dar', 'Color':'red',  'Texture':'Outline',  'Shape':'Asterisk', 'Task':{'texture':'Face',  'shape':'Scene', 'color':'Face'}, 'cue_stim': copy.copy(donut_asterisk_red) }}

cue_list = ['fsr', 'fsb', 'far', 'fab', 'dsr', 'dsb', 'dab', 'dar']
Cue_types = {'fsr': {'cue':'fsr', 'Color':'red',  'Texture':'Filled', 'Shape':'Star',     'Task':{'texture':'Scene', 'shape':'Face',  'color':'Scene'}, 
                     'Stay': {'texture': ['fsr','fsb'],            'shape': ['fsr','dsr'],             'color': ['dsr','fsr']}, 
                     'IDS': {'texture': ['far','fab'],             'shape': ['fsb','dsb'],             'color': ['far','dar']}, 
                     'EDS': {'texture': ['dsr','dsb','dab','dar'], 'shape': ['far','fab','dar','dab'], 'color': ['fab','fsb','dab','dsb']}, 'cue_stim': copy.copy(filled_star_red) },
             
             'fsb': {'cue':'fsb', 'Color':'blue', 'Texture':'Filled', 'Shape':'Star',     'Task':{'texture':'Scene', 'shape':'Scene', 'color':'Face'},  
                     'Stay': {'texture': ['fsr','fsb'],            'shape': ['fsb','dsb'],             'color': ['fsb','fab']}, 
                     'IDS': {'texture': ['far','fab'],             'shape': ['fsr','dsr'],             'color': ['dab','dsb']}, 
                     'EDS': {'texture': ['dsr','dsb','dab','dar'], 'shape': ['far','fab','dar','dab'], 'color': ['far','dar','fsr','dsr']}, 'cue_stim': copy.copy(filled_star_blue) },
             
             'far': {'cue':'far', 'Color':'red',  'Texture':'Filled', 'Shape':'Asterisk', 'Task':{'texture':'Face',  'shape':'Face',  'color':'Face'},  
                     'Stay': {'texture': ['far','fab'],            'shape': ['far','fab'],              'color': ['far','dar']}, 
                     'IDS': {'texture': ['fsr','fsb'],             'shape': ['dar','dab'],              'color': ['dsr','fsr']}, 
                     'EDS': {'texture': ['dsr','dsb','dab','dar'], 'shape': ['fsr','dsr','fsb','dsb'],  'color': ['fab','fsb','dab','dsb']}, 'cue_stim': copy.copy(filled_asterisk_red) },
             
             'fab': {'cue':'fab', 'Color':'blue', 'Texture':'Filled', 'Shape':'Asterisk', 'Task':{'texture':'Face',  'shape':'Face',  'color':'Face'},  
                     'Stay': {'texture': ['far','fab'],            'shape': ['far','fab'],             'color': ['fsb','fab']}, 
                     'IDS': {'texture': ['fsr','fsb'],             'shape': ['dar','dab'],             'color': ['dab','dsb']}, 
                     'EDS': {'texture': ['dsr','dsb','dab','dar'], 'shape': ['fsr','dsr','fsb','dsb'], 'color': ['far','dar','fsr','dsr']}, 'cue_stim': copy.copy(filled_asterisk_blue) },
             
             'dsr': {'cue':'dsr', 'Color':'red',  'Texture':'Donut',  'Shape':'Star',     'Task':{'texture':'Face',  'shape':'Face',  'color':'Scene'}, 
                     'Stay': {'texture': ['dar','dsr'],            'shape': ['fsr','dsr'],             'color': ['dsr','fsr']}, 
                     'IDS': {'texture': ['dab','dsb'],             'shape': ['fsb','dsb'],             'color': ['far','dar']}, 
                     'EDS': {'texture': ['fsr','fsb','far','fab'], 'shape': ['far','fab','dar','dab'], 'color': ['fab','fsb','dab','dsb']}, 'cue_stim': copy.copy(donut_star_red) },
             
             'dsb': {'cue':'dsb', 'Color':'blue', 'Texture':'Donut',  'Shape':'Star',     'Task':{'texture':'Scene', 'shape':'Scene', 'color':'Scene'}, 
                     'Stay': {'texture': ['dab','dsb'],            'shape': ['fsb','dsb'],             'color': ['dab','dsb']}, 
                     'IDS': {'texture': ['dar','dsr'],             'shape': ['fsr','dsr'],             'color': ['fsb','fab']}, 
                     'EDS': {'texture': ['fsr','fsb','far','fab'], 'shape': ['far','fab','dar','dab'], 'color': ['far','dar','fsr','dsr']}, 'cue_stim': copy.copy(donut_star_blue) },
             
             'dab': {'cue':'dab', 'Color':'blue', 'Texture':'Donut',  'Shape':'Asterisk', 'Task':{'texture':'Scene', 'shape':'Scene', 'color':'Scene'}, 
                     'Stay': {'texture': ['dab','dsb'],            'shape': ['dar','dab'],             'color': ['dab','dsb']}, 
                     'IDS': {'texture': ['dar','dsr'],             'shape': ['far','fab'],             'color': ['fsb','fab']}, 
                     'EDS': {'texture': ['fsr','fsb','far','fab'], 'shape': ['fsr','dsr','fsb','dsb'], 'color': ['far','dar','fsr','dsr']}, 'cue_stim': copy.copy(donut_asterisk_blue) },
             
             'dar': {'cue':'dar', 'Color':'red',  'Texture':'Donut',  'Shape':'Asterisk', 'Task':{'texture':'Face',  'shape':'Scene', 'color':'Face'},  
                     'Stay': {'texture': ['dar','dsr'],            'shape': ['dar','dab'],             'color': ['dar','far']}, 
                     'IDS': {'texture': ['dab','dsb'],             'shape': ['far','fab'],             'color': ['dsr','fsr']}, 
                     'EDS': {'texture': ['fsr','fsb','far','fab'], 'shape': ['fsr','dsr','fsb','dsb'], 'color': ['fab','fsb','dab','dsb']}, 'cue_stim': copy.copy(donut_asterisk_red) }}

##### Load Face and Scene pictures into stim objects, organized into a dict
#Dictionaries and the corresponding file paths
direc = os.path.join(os.getcwd(),'localizer_stim') #_thisDir #'/Users/mpipoly/Desktop/Psychopy/localizer_stim/' #always setup path on the fly in case you switch computers
scene_ext = 'scenes/*.jpg' #file delimiter
faces_ext = 'faces/*.jpg'
faces_list = glob.glob(os.path.join(direc, faces_ext))
scenes_list = glob.glob(os.path.join(direc, scene_ext))
Img_Dict = {}
# randomly select pics from list, only load same number of pics as number of trials to save memory
for i,f in enumerate(np.random.randint(low=0, high=len(faces_list), size=Task_Parameters['n_trials'])):
    if randomchoice([1,2])==1: #'Face'
        Img_Dict[i] = {'cond':'Face', 'img_obj':visual.ImageStim(win=win, image=(Image.open(faces_list[f])).convert('L') ), 'filename':('faces/'+os.path.basename(faces_list[f]))}
    else: #'Scene'
        Img_Dict[i] = {'cond':'Scene', 'img_obj':visual.ImageStim(win=win, image=(Image.open(scenes_list[f])).convert('L') ), 'filename':('scenes/'+os.path.basename(scenes_list[f]))}
#print(Img_Dict)

################################################################################################################################
##################################################     set up for output     ##################################################
# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
thisDir_save=_thisDir
thisDir_save=thisDir_save.split('/')
thisDir_save='/'.join(thisDir_save[:-1])

################################################################################################################################
############################################  Define Task functions & Text Objects  ############################################

# prepare task trials
def prepare_block_trials(i_block, Cue_types, Task_Parameters, delay_cond, retrocues, Img_Dict, pic_order, resp_keys):
    # set up output dict
    cue_types = list(Cue_types.keys())
    out_dict = {'block': np.ones(Task_Parameters['n_trials'])+i_block,
                'trial': np.linspace(1,Task_Parameters['n_trials'],num=Task_Parameters['n_trials'],endpoint=True),
                'delay': np.ones(Task_Parameters['n_trials'])*delay_cond, 'retro_freq': np.ones(Task_Parameters['n_trials'])*Task_Parameters['retro_freq'][0][i_block],
                'cue': list( np.random.permutation( np.tile( cue_types, int(Task_Parameters['n_trials']/len(cue_types)) ) ) ),
                'texture': [],   'shape': [],   'color': [],
                'retrocue': list( np.random.permutation( np.concatenate((np.tile( retrocues[0], round(Task_Parameters['retro_freq'][0][i_block]*Task_Parameters['n_trials']) ), np.tile( retrocues[1], round((1-Task_Parameters['retro_freq'][0][i_block])*Task_Parameters['n_trials']) ))) ) ),
                'stimulus': [],   'image_filename': [],   'task': [],
                'corr_resp': np.zeros(Task_Parameters['n_trials']),   'subj_resp': np.ones(Task_Parameters['n_trials'])*-1,
                'correct': np.zeros(Task_Parameters['n_trials']),   'rt': np.ones(Task_Parameters['n_trials'])*-1}
    
    for i in range(Task_Parameters['n_trials']):
        # record what image type (face|scene) and filename
        out_dict['stimulus'].append( Img_Dict[pic_order[i]]['cond'] )
        out_dict['image_filename'].append( Img_Dict[pic_order[i]]['filename'] )
        # record what correct resp would be 
        # 1st if: what task is associated with this cue 
        out_dict['task'].append(Cue_types[out_dict['cue'][i]]['Task'][out_dict['retrocue'][i]])
        # 2nd if: what stimulus was presented 
        if out_dict['task'][i] == 'Scene':
            if out_dict['stimulus'][i]=='Face':
                out_dict['corr_resp'][i] = resp_keys['no_key'][1]
            elif out_dict['stimulus'][i]=='Scene':
                out_dict['corr_resp'][i] = resp_keys['yes_key'][1]
        elif out_dict['task'][i] == 'Face':
            if out_dict['stimulus'][i]=='Face':
                out_dict['corr_resp'][i] = resp_keys['yes_key'][1]
            elif out_dict['stimulus'][i]=='Scene':
                out_dict['corr_resp'][i] = resp_keys['no_key'][1]
        # add cue textre, shape, and color
        out_dict['texture'].append(Cue_types[out_dict['cue'][i]]['Texture'])
        out_dict['shape'].append(Cue_types[out_dict['cue'][i]]['Shape'])
        out_dict['color'].append(Cue_types[out_dict['cue'][i]]['Color'])
    
    return out_dict

#This will be for CSV initiation
def makeCSV(filename, thistrialDict, trial_num):
    pd.DataFrame(thistrialDict).to_csv(filename, index=False)
#     with open(filename + '.csv', mode='w') as our_data:
#         ExpHead=thistrialDict.keys() # constructing header information from the dictionary keys
#         writer=csv.DictWriter(our_data,fieldnames=ExpHead)
#         writer.writeheader()
#         for n in range(trial_num+1): # for each trial, up to the current trial, write one row 
#            writer.writerow(thistrialDict[n])

def present_obj(obj, fixobj, cue_dur, frame_rate):
    for frame in range(int(cue_dur*frame_rate)):
        obj.draw() # draw the cue obj or retrocue text or stimuli image
        if fixobj != []:
            fixobj.draw()
        if timing_test:
            white_box.draw()
        win.flip()

def present_delay(fixobj, delay_dur, frame_rate):
    for frame in range(int(delay_dur*frame_rate)):
        fixobj.draw()
        win.flip() # draw the cue obj

def present_stim_get_resp(stimobj, fixobj, cue_dur, frame_rate, trigDict, expInfo, resp_keys, corr_resp):
    kb.clearEvents() # restart on each frame to get frame time + button press time
    kb.clock.reset() # restart on each frame to get frame time + button press time
    keys=[]
    cur_resp = -1
    cur_RT = -1 # because no response
    for frame in range(int(cue_dur*frame_rate)):
        stimobj.draw() # draw the cue obj or retrocue text or stimuli image
        fixobj.draw()
        if timing_test:
            white_box.draw()
        win.flip()
        if (keys==[]):
            keys = kb.getKeys(keyList=[resp_keys['yes_key'][0],resp_keys['no_key'][0]], waitRelease=False, clear=True) # keyList=[resp_keys['yes_key'][0],resp_keys['no_key'][0]], 
            if (keys!=[]):
                if (keys!=None) and (keys!=[]):
                    for key in keys:
                        print(key.name)
                        if key.name == resp_keys['yes_key'][0]:
                            cur_resp = resp_keys['yes_key'][1] # [A]
                        elif key.name == resp_keys['no_key'][0]:
                            cur_resp = resp_keys['no_key'][1] # [S]
                        elif key.name == 'escape':
                            win.close()
                            core.quit()
                        cur_RT = key.rt # DOUBLE CHECK THIS
                # throw response flag
                if (expInfo['Method']=='EEG') and (trigDict!={}):
                    if cur_resp != -1:
                        if cur_resp == corr_resp:
                            # if correct
                            port.write(bytes([trigDict['resp']['correct'][cur_resp]]))
                        elif cur_resp != corr_resp:
                            # if incorrect
                            port.write(bytes([trigDict['resp']['incorrect'][cur_resp]]))
                        else:
                            print('Something Wrong with triggers for resp')
                            core.quit()
        # if (keys!=[]):
        #     break # exit for loop that is displaying stimuli
#    if (keys!=None) and (keys!=[]):
#        for key in keys:
#            print(key.name)
#            if key.name == resp_keys['yes_key'][0]:
#                cur_resp = resp_keys['yes_key'][1] # [A]
#            elif key.name == resp_keys['no_key'][0]:
#                cur_resp = resp_keys['no_key'][1] # [S]
#            elif key.name == 'escape':
#                win.close()
#                core.quit()
#            cur_RT = key.rt # DOUBLE CHECK THIS
#    else:
#        cur_resp = -1
#        cur_RT = -1 # because no response
    
    # # Screen 5B - RESP if any
    # if (expInfo['Method']=='EEG') and (trigDict!={}):
    #     if cur_resp != -1:
    #         if cur_resp == corr_resp:
    #             # if correct
    #             port.write(bytes([trigDict['resp']['correct'][cur_resp]]))
    #         elif cur_resp != corr_resp:
    #             # if incorrect
    #             port.write(bytes([trigDict['resp']['incorrect'][cur_resp]]))
    #         else:
    #             print('Something Wrong with triggers for resp')
    #             core.quit()
            
    return cur_RT, cur_resp

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

def copy_cueobjs_change_pos(cur_retrocue, cur_LL_rules, Cue_types, obj_codes):
    # use Cue_Types dictionary to pull out cues related to the current sub-features (want one of each)
    if cur_LL_rules==[]:
        # in this case co1 and co2 will be lists will all cues on each side of the split
        if cur_retrocue=='texture':
            co1 = ['far','fab','fsr','fsb']
            co2 = ['dar','dsr','dab','dsb']
        elif cur_retrocue=='shape':
            co1 = ['far','fab','dar','dab']
            co2 = ['fsr','dsr','fsb','dsb']
        elif cur_retrocue=='color':
            co1 = ['far','dar','fsr','dsr']
            co2 = ['fab','fsb','dab','dsb']

        for ind, cur_cue in enumerate(co1):
            xpos = (-0.15+(-0.24*ind))
            (Cue_types[cur_cue]['cue_stim']).setPos((xpos,-0.4))
        for ind, cur_cue in enumerate(co2):
            xpos = (0.15+(0.24*ind))
            (Cue_types[cur_cue]['cue_stim']).setPos((xpos,-0.4))

    elif cur_LL_rules!=[]:
        rand_3rd_choice = np.random.choice([0,1])
        if cur_LL_rules[0] == 'filled in':
            co1 = [('f' + obj_codes[1][0] + obj_codes[2][rand_3rd_choice]), ('f' + obj_codes[1][0] + obj_codes[2][1-rand_3rd_choice])]
            co2 = [('f' + obj_codes[1][1] + obj_codes[2][rand_3rd_choice]), ('f' + obj_codes[1][1] + obj_codes[2][1-rand_3rd_choice])]
        elif cur_LL_rules[0] == 'only an outline':
            co1 = [('d' + obj_codes[1][rand_3rd_choice] + obj_codes[2][0]), ('d' + obj_codes[1][1-rand_3rd_choice] + obj_codes[2][0])]
            co2 = [('d' + obj_codes[1][rand_3rd_choice] + obj_codes[2][1]), ('d' + obj_codes[1][1-rand_3rd_choice] + obj_codes[2][1])]
        elif cur_LL_rules[0]=='asterisk':
            co1 = [(obj_codes[0][0] + 'a' + obj_codes[2][rand_3rd_choice]), (obj_codes[0][0] + 'a' + obj_codes[2][1-rand_3rd_choice])]
            co2 = [(obj_codes[0][1] + 'a' + obj_codes[2][rand_3rd_choice]), (obj_codes[0][1] + 'a' + obj_codes[2][1-rand_3rd_choice])]
        elif cur_LL_rules[0]=='star':
            co1 = [(obj_codes[0][rand_3rd_choice] + 's' + obj_codes[2][1]), (obj_codes[0][1-rand_3rd_choice] + 's' + obj_codes[2][1])]
            co2 = [(obj_codes[0][rand_3rd_choice] + 's' + obj_codes[2][0]), (obj_codes[0][1-rand_3rd_choice] + 's' + obj_codes[2][0])]
        elif cur_LL_rules[0]=='red':
            co1 = [(obj_codes[0][rand_3rd_choice] + obj_codes[1][0] + 'r'), (obj_codes[0][1-rand_3rd_choice] + obj_codes[1][0] + 'r')]
            co2 = [(obj_codes[0][rand_3rd_choice] + obj_codes[1][1] + 'r'), (obj_codes[0][1-rand_3rd_choice] + obj_codes[1][1] + 'r')]
        elif cur_LL_rules[0]=='blue':
            co1 = [(obj_codes[0][0] + obj_codes[1][rand_3rd_choice] + 'b'), (obj_codes[0][0] + obj_codes[1][1-rand_3rd_choice] + 'b')]
            co2 = [(obj_codes[0][1] + obj_codes[1][rand_3rd_choice] + 'b'), (obj_codes[0][1] + obj_codes[1][1-rand_3rd_choice] + 'b')]

        for ind, cur_cue in enumerate(co1):
            xpos = (-0.2+(-0.24*ind))
            (Cue_types[cur_cue]['cue_stim']).setPos((xpos,-0.4))
        for ind, cur_cue in enumerate(co2):
            xpos = (0.2+(0.24*ind))
            (Cue_types[cur_cue]['cue_stim']).setPos((xpos,-0.4))
        
    return co1, co2


# Instructions screen
Prac = visual.TextStim(win=win, name='Prac', text=u'You are now about to begin the practice block. \n\nGet Ready \n\nPress Any Key to Continue', font=u'Arial', units='norm', pos=(0, 0), height=0.09, color=u'white', colorSpace='rgb');
    
Directions = visual.TextStim(win=win, name='Directions', text=u'You are now about to begin the task. \n\nGet Ready \n\nPress Any Key to Continue', font=u'Arial', units='norm', pos=(0, 0), height=0.09, color=u'white', colorSpace='rgb');
    
Block_Screen = visual.TextStim(win=win, name='BlockScreen', text=u'Press Any Key to Begin the 1st Block', font=u'Arial', units='norm', pos=(0, 0), height=0.09, color=u'white', colorSpace='rgb');

Break_Screen = visual.TextStim(win=win, name='BreakScreen', text=u'Halfway through the block.\nPress any key to continue', font=u'Arial', units='norm', pos=(0, 0), height=0.09, color=u'white', colorSpace='rgb');

End_Exp = visual.TextStim(win=win, name='End_of_Experiment', text=u'You have finished the block\n\n The experiment window will close by itself', font=u'Arial', units='norm', pos=(0, 0), height=0.09, color=u'white', colorSpace='rgb');

Repeat_Screen = visual.TextStim(win=win, name='RepeatScreen', text=u'Wait for the experimentor to press a key', font=u'Arial', units='norm', pos=(0, 0), height=0.09, color=u'white', colorSpace='rgb')
Cur_Acc = visual.TextStim(win=win, name='CurAcc', text=u'Acc', font=u'Arial', units='norm', pos=(0, -0.3), height=0.09, color=u'white', colorSpace='rgb')
# Central fixations
Fix_Cue = visual.TextStim(win=win, name='Fix_Cue', text=u'+', font=u'Arial', units='norm', pos=(0, 0), height=0.1, ori=0, color=u'black', colorSpace='rgb', opacity=1)

PrepForTask_Txt = visual.TextStim(win=win, name='RepeatInstruct', text=u'Please wait while the experimentor goes to the other room to start the EEG recording for the task data', font=u'Arial', units='norm', pos=(0, 0), height=0.09, color=u'white', colorSpace='rgb')



def run_tutorial(version, Prac_Cue_types, tree_list):
    # set up some text variables, dictionaries, and lists up front
    cue_list = ['fsr', 'fsb', 'far', 'fab', 'dsr', 'dsb', 'dab', 'dar']
    Overview = visual.TextStim(win=win, text=("The retro cues you will see in the task are " + version_retrocues[version][0] + " and " + version_retrocues[version][1]), font=u'Arial', units='norm', pos=(0, 0.3), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1);
    textureTxt = visual.TextStim(win=win, text=u'If you see a texture retro cue you will need to first determine if the cue object presented at the beginning of the trial was filled in or just an outline', wrapWidth=1.68, font=u'Arial', units='norm', pos=(0, 0.2), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1);
    shapeTxt = visual.TextStim(win=win, text=u'If you see a shape retro cue you will need to first determine if the cue object presented at the beginning of the trial is an 8-point star or 8-point asterisk', wrapWidth=1.68, font=u'Arial', units='norm', pos=(0, 0.2), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1);
    colorTxt = visual.TextStim(win=win, text=u'If you see a color retro cue you will need to first determine if the cue object presented at the beginning of the trial is red or blue', wrapWidth=1.68, font=u'Arial', units='norm', pos=(0, 0.2), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1);
    RCL_Dict = {'texture':textureTxt, 'shape':shapeTxt, 'color':colorTxt}
    LL_txt = visual.TextStim(win=win, text=u'text', wrapWidth=1.68, font=u'Arial', units='norm', pos=(0,0.2), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1)
    WhatTaskQ = visual.TextStim(win=win, text=u'If you see this cue object, what task should you perform?\nPress any key to see the answer', font=u'Arial', units='norm', pos=(0,0.5), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1)
    Answer = visual.TextStim(win=win, text=u'Face', font=u'Arial', units='norm', pos=(0, 0), height=0.3, ori=0, color=u'white', colorSpace='rgb', opacity=1)
    LL_Dict = {'texture':[['filled in','shape'],['only an outline','color']], 'shape':[['asterisk','texture'],['star','color']], 'color':[['red','shape'],['blue','texture']]}
    LL_to_Task = {'shape':[['asterisk','face'],['star','scene']], 'color':[['red','face'],['blue','scene']], 'texture':[['filled in','face'],['only an outline','scene']]}
    HCC_Dict = {'texture':{'shape':{'asterisk':['far','fab'], 'star':['fsr','fsb']}, 'color':{'red':['dar','dsr'], 'blue':['dsb','dab']}}, 
                'shape':{'texture':{'filled in':['far','fab'], 'only an outline':['dar','dab']}, 'color':{'red':['fsr','dsr'], 'blue':['fsb','dsb']}}, 
                'color':{'shape':{'asterisk':['far','dar'], 'star':['fsr','dsr']}, 'texture':{'filled in':['fab','fsb'], 'only an outline':['dab','dsb']}}}
    obj_codes = {0:['f','d'], 1:['a','s'], 2:['r','b']}
    
    # -------------------- Run Tutorial ------------------
    # display initial text
    Overview.draw()
    tree_list[0].draw()
    tree_list[1].draw()
    win.flip()
    event.waitKeys()

    retrocue_list = version_retrocues[version]
    tree_dict = {retrocue_list[0]:tree_list[0], retrocue_list[1]:tree_list[1]} # make something that indicates what tree is what since we are shuffling
    Shuffle(retrocue_list) # shuffle the order so we don't just show one condition first all the time
    for ind, cur_retrocue in enumerate(retrocue_list):
        cur_tree = visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/'+cur_retrocue+'.png', units='norm', pos=(-0.7,0.7), size=0.55)
        ### Display overview text for first retrocue (what feature dimension we're splitting by)
        print('current retro cue:', cur_retrocue)
        ex_stim1, ex_stim2 = copy_cueobjs_change_pos(cur_retrocue, [], Prac_Cue_types, obj_codes) # use Cue_Types dictionary to pull out cues related to the current sub-features (want one of each)
        for cind in range(len(ex_stim1)): 
            (Prac_Cue_types[ex_stim1[cind]]['cue_stim']).draw()
            (Prac_Cue_types[ex_stim2[cind]]['cue_stim']).draw()
        (RCL_Dict[cur_retrocue]).draw() # print overview for this specific retro cue
        cur_tree.draw()
        win.flip()
        event.waitKeys()
        
        # loop through sub-trees (should be 2)
        LL_lists = LL_Dict[cur_retrocue]
        Shuffle(LL_lists)
        for cur_LL_list in LL_lists:
            ### Display text for the 2nd feature dimension to focus on given which side of the 1st feature we're on
            print('currently in this sub tree: ',cur_LL_list)
            LL_txt.text = (("If the object is " + cur_LL_list[0] + ", the " + cur_LL_list[1] + " will determine what task (Face or Scene) that you should do on this trial."))
            ex_stimA, ex_stimB = copy_cueobjs_change_pos(cur_retrocue, cur_LL_list, Prac_Cue_types, obj_codes)
            for cind in range(len(ex_stimA)): 
                (Prac_Cue_types[ex_stimA[cind]]['cue_stim']).draw()
                (Prac_Cue_types[ex_stimB[cind]]['cue_stim']).draw()
            cur_tree.draw()
            LL_txt.draw()
            win.flip()
            event.waitKeys()
            
            # loop through sub-trees (should be 2)
            task_pair_lists = LL_to_Task[cur_LL_list[1]] # grabs current sub-feat (e.g., if shape, grabs star and asterisk)
            Shuffle(task_pair_lists)
            for cur_pair in task_pair_lists:
                LL_txt.text = ((cur_pair[0] + " means you should do the " + cur_pair[1] + " task."))
                LL_txt.draw()
                cur_tree.draw()
                # use Cue_Types dictionary to pull out cues related to the current sub-features (want both)
                print((cur_pair[0] + " means you should do the " + cur_pair[1] + " task."))
                print(HCC_Dict[cur_retrocue][cur_LL_list[1]][cur_pair[0]][0], '\tretrocue:', cur_retrocue, '\tcur_LL_list:', cur_LL_list[1], '\tcur_pair:', cur_pair[0])
                print(HCC_Dict[cur_retrocue][cur_LL_list[1]][cur_pair[0]][1], '\tretrocue:', cur_retrocue, '\tcur_LL_list:', cur_LL_list[1], '\tcur_pair:', cur_pair[0])
                (Prac_Cue_types[HCC_Dict[cur_retrocue][cur_LL_list[1]][cur_pair[0]][0]]['cue_stim']).setPos((-0.15,-0.4))
                (Prac_Cue_types[HCC_Dict[cur_retrocue][cur_LL_list[1]][cur_pair[0]][1]]['cue_stim']).setPos((0.15,-0.4))
                (Prac_Cue_types[HCC_Dict[cur_retrocue][cur_LL_list[1]][cur_pair[0]][0]]['cue_stim']).draw()
                (Prac_Cue_types[HCC_Dict[cur_retrocue][cur_LL_list[1]][cur_pair[0]][1]]['cue_stim']).draw()
                win.flip()
                event.waitKeys()

            #WhatTaskQ.draw()
            # draw the cue objects one by one for the current sub tree
            # cur_LL_list[0] would tell me the conditon of the feature
            # (Prac_Cue_types[HCC_Dict[cur_retrocue][cur_LL_list[1]][cur_pair[0]][0]]['cue_stim']).draw()
            # (Prac_Cue_types[HCC_Dict[cur_retrocue][cur_LL_list[1]][cur_pair[0]][1]]['cue_stim']).draw()
        
        (tree_dict[cur_retrocue]).draw()
        win.flip()
        event.waitKeys()
        
        # draw the cue objects one by one for the current retro cue tree
        Shuffle(cue_list) # shuffle the order
        for cur_cue in cue_list:
            WhatTaskQ.draw()
            (Prac_Cue_types[cur_cue]['cue_stim']).setPos((0,0))
            (Prac_Cue_types[cur_cue]['cue_stim']).draw()
            win.flip()
            event.waitKeys()
            # now give answer
            Answer.text = Prac_Cue_types[cur_cue]['Task'][cur_retrocue]
            Answer.draw()
            win.flip()
            event.waitKeys()
      
        # practice for just one retro cue
        repeat_prac = 1
        while repeat_prac == 1:
            pic_order = np.random.permutation( np.linspace(0, Prac_Task_Parameters['n_trials'], num=Prac_Task_Parameters['n_trials'], endpoint=False) )
            out_dict = prepare_block_trials(0, Cue_types, Prac_Task_Parameters, Prac_Trl_Durs['delay_2'][ind], [cur_retrocue, cur_retrocue], Img_Dict, pic_order, resp_keys)
            filename = thisDir_save + u'/ThalHi_data/v2_' + expInfo['Method'] + '_data/sub-%s_task-%s_block-00%s_date-%s.csv' % (expInfo['Participant_ID'], (expName+'PracticeWith'+cur_retrocue), (0+1), datetime.today().strftime('%m-%d-%y'))
            
            Prac.draw()
            win.flip()
            event.waitKeys()
            
            block_acc = run_task(0, Prac_Task_Parameters, expInfo, Prac_Trl_Durs, frame_rate, Cue_types, retrocue_textobj, Img_Dict, pic_order, {}, out_dict, resp_keys, filename, 1)
            
            Repeat_Screen.draw()
            Cur_Acc.text = ("Accuracy: ", str(block_acc))
            Cur_Acc.draw()
            win.flip()
            keypress = event.waitKeys(keyList=['r','m'])
            print(keypress[0])
            if keypress[0] == 'm':
                repeat_prac = 0
    
    

####### Set up functions for practice, task, etc.
def run_task(i_block, Task_Parameters, expInfo, Trl_Durs, frame_rate, Cue_types, retrocue_textobj, Img_Dict, pic_order, trigDict, out_dict, resp_keys, filename, giveFeed):
    block_acc = 0
    for i_trial in range(0, Task_Parameters['n_trials']):
        
        # Screen 1 - CUE OBJ
        if (expInfo['Method']=='EEG') and (trigDict!={}):
            cue_trig = trigDict['cue'][out_dict['cue'][i_trial]]
            win.callOnFlip(port.write,bytes([cue_trig]))
        # present_obj(obj, cue_dur, frame_rate)
        present_obj(Cue_types[out_dict['cue'][i_trial]]['cue_stim'], Fix_Cue, Trl_Durs['cue'], frame_rate)

        # Screen 2 - DELAY
        if (expInfo['Method']=='EEG') and (trigDict!={}):
            win.callOnFlip(port.write,bytes([trigDict['delay_1']]))
        # present_delay(fixobj, delay_dur, frame_rate)
        present_delay(Fix_Cue, Trl_Durs['delay_1'], frame_rate)

        # Screen 3 - RETROCUE
        if (expInfo['Method']=='EEG') and (trigDict!={}):
            win.callOnFlip(port.write,bytes([trigDict['retrocue'][out_dict['retrocue'][i_trial]]]))
        # present_obj(obj, cue_dur, frame_rate)
        present_obj(retrocue_textobj[out_dict['retrocue'][i_trial]], [], Trl_Durs['retrocue'], frame_rate)

        # Screen 4 - DELAY
        if (expInfo['Method']=='EEG') and (trigDict!={}):
            win.callOnFlip(port.write,bytes([trigDict['delay_2']]))
        # present_delay(fixobj, delay_dur, frame_rate)
        print('delay length:',Trl_Durs['delay_2'][i_block])
        present_delay(Fix_Cue, Trl_Durs['delay_2'][i_block] , frame_rate)

        # Screen 5 - STIMULI
        if (expInfo['Method']=='EEG') and (trigDict!={}):
            win.callOnFlip(port.write,bytes([trigDict['stim'][out_dict['stimulus'][i_trial]]]))
        #                  present_stim_get_resp(stimobj, cue_dur, frame_rate, trigDict, resp_keys, corr_resp)
        cur_RT, cur_resp = present_stim_get_resp(Img_Dict[pic_order[i_trial]]['img_obj'], Fix_Cue, Trl_Durs['stim'], frame_rate, trigDict, expInfo, resp_keys, out_dict['corr_resp'][i_trial])
        # save out resp info in dict
        out_dict['subj_resp'][i_trial] = cur_resp
        if cur_resp==out_dict['corr_resp'][i_trial]:
            out_dict['correct'][i_trial] = 1
        out_dict['rt'][i_trial] = cur_RT
        
        if giveFeed:
            horz_line.draw()
            if out_dict['correct'][i_trial]==1:
                vert_line.draw()
                block_acc+=1
            else:
                (tree_imgs[out_dict['retrocue'][i_trial]][out_dict['cue'][i_trial]]).draw()
                win.flip()
                event.waitKeys(maxWait=5)
            win.flip()
            core.wait(0.5)
        else: 
            horz_line.draw()
            if out_dict['correct'][i_trial]==1:
                vert_line.draw()
                block_acc+=1
                if (expInfo['Method']=='EEG') and (trigDict!={}):
                    win.callOnFlip(port.write,bytes([trigDict['feedback']['correct']]))
            else:
                if (expInfo['Method']=='EEG') and (trigDict!={}):
                    win.callOnFlip(port.write,bytes([trigDict['feedback']['incorrect']]))
            win.flip()
            core.wait(0.5)
        
        # Screen 6 - ITI
        if (expInfo['Method']=='EEG') and (trigDict!={}):
            win.flip()
            win.callOnFlip(port.write,bytes([trigDict['ITI']]))
        # present_delay(fixobj, delay_dur, frame_rate)
        np.random.shuffle(Trl_Durs['ITI'])
        present_delay(Fix_Cue, Trl_Durs['ITI'][0], frame_rate)
        
        # save out csv after this trial
        makeCSV(filename=filename, thistrialDict=out_dict, trial_num=i_trial) # saving CSV at every trial 
    
    if giveFeed:
        block_acc = round((block_acc/(Task_Parameters['n_trials']))*100,2)
        return block_acc



################################################################################################################################
##################################################         RUN TASK         ##################################################
#### Setting up a global clock to track initiation of experiment to end
kb = keyboard.Keyboard() # keyboard has better timing that other keypress functions

###### load tree pngs
tree1 = visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/'+retrocues[0]+'.png', units='norm', pos=(0.45,-0.3), size=0.85) 
tree2 = visual.ImageStim(win=win, image=os.getcwd()+'/tutorial/'+retrocues[1]+'.png', units='norm', pos=(-0.45,-0.3), size=0.85) 


###### Tutorial
if skip_tutorial==0:
    #run_tutorial(version, Prac_Cue_types, [tree1, tree2])
    repeat_prac = 1
    if expInfo['Method']=='EEG':
        ##### TTL Pulse trigger
        port.open()
else:
    repeat_prac = 0


attempt_num=0
while repeat_prac == 1:
    pic_order = np.random.permutation( np.linspace(0, Prac_Task_Parameters['n_trials'], num=Prac_Task_Parameters['n_trials'], endpoint=False) )
    out_dict = prepare_block_trials(0, Cue_types, Prac_Task_Parameters, Prac_Trl_Durs['delay_2'][2], retrocues, Img_Dict, pic_order, resp_keys)
    filename = thisDir_save + u'/ThalHi_data/v2_' + expInfo['Method'] + '_data/sub-%s_task-%s_block-00%s_date-%s.csv' % ( str(int(expInfo['Participant_ID'])), (expName+'PracticeBothRetrocues'), str((attempt_num+1)), datetime.today().strftime('%Y%m%d%I%M'))
    
    tree1.draw()
    tree2.draw()
    win.flip()
    event.waitKeys()
    
    ###### Practice
    Prac.draw()
    win.flip()
    event.waitKeys()
    
    ##### 3 seconds Intial fixation
    if expInfo['Method']=='EEG':
        Fix_Cue.draw()
        win.flip()
        port.write(trigDict['startSaveflag'])
        core.wait(1) # wait 1 second before throwing block start trigger
        win.callOnFlip(port.write,bytes([trigDict['blockStart']]))
    Fix_Cue.draw()
    win.flip()
    core.wait(3) # wait 3 seconds before beginning practice block

    block_acc = run_task(0, Prac_Task_Parameters, expInfo, Prac_Trl_Durs, frame_rate, Cue_types, retrocue_textobj, Img_Dict, pic_order, trigDict, out_dict, resp_keys, filename, 1)
    
    # throw end of block trigger and stop save if EEG
    if expInfo['Method']=='EEG':
        win.callOnFlip(port.write,bytes([trigDict['blockEnd']]))
        Fix_Cue.draw()
        win.flip()
        core.wait(2) # show fixation for 2 seconds after last trial
        port.write(trigDict['stopSaveflag'])
        core.wait(1) # wait 1 more second after throwing stop save flag
    
    Repeat_Screen.draw()
    Cur_Acc.text = ("Accuracy: ", str(block_acc))
    Cur_Acc.draw()
    win.flip()
    core.wait(1.5)
    keypress = event.waitKeys(keyList=['r','m'])
    print(keypress[0])
    attempt_num+=1
    if keypress[0] == 'm':
        repeat_prac = 0
        if expInfo['Method']=='EEG':
            ##### TTL Pulse trigger
            #win.callonFlip(pport.setData,delay1trig)
            core.wait(1) # wait extra seconds before closing port
            port.close() # close practice file before moving on


#### Reminder to set up the EEG computer for the task
PrepForTask_Txt.draw()
win.flip()
event.waitKeys()

#### Draw instruction screens 
Directions.draw()
win.flip()
event.waitKeys()


##### RUN TASK BLOCKS
if expInfo['Method']=='BEH':
    # shuffle delay order for the 4 blocks
    np.random.shuffle(Trl_Durs['delay_2'])
if expInfo['Method']=='EEG':
    #### TTL Pulse trigger
    port.open()

for i_block in range(int(Task_Parameters['n_blocks'])):
    # set up trials for block while participant takes a break
    pic_order = np.random.permutation( np.linspace(0, Task_Parameters['n_trials'], num=Task_Parameters['n_trials'], endpoint=False) )
    out_dict = prepare_block_trials(i_block, Cue_types, Task_Parameters, Trl_Durs['delay_2'][i_block], retrocues, Img_Dict, pic_order, resp_keys)
    filename = thisDir_save + u'/ThalHi_data/v2_' + expInfo['Method'] + '_data/sub-%s_task-%s_block-00%s_date-%s.csv' % ( str(int(expInfo['Participant_ID'])), expName, (i_block+1), datetime.today().strftime('%m-%d-%y'))
        
    # display block screen
    Block_Screen.draw()
    win.flip()
    if testing_code:
        event.waitKeys(maxWait=5)
    else:
        event.waitKeys()
    
    ##### 3 seconds Intial fixation
    if expInfo['Method']=='EEG':
        Fix_Cue.draw()
        win.flip()
        port.write(trigDict['startSaveflag'])
        core.wait(1) # wait 1 second before throwing block start trigger
        win.callOnFlip(port.write,bytes([trigDict['blockStart']]))
    Fix_Cue.draw()
    win.flip()
    core.wait(3) # wait 3 seconds before beginning block
    
    # run the task block
    if expInfo['Method']=='EEG':
        run_task(i_block, Task_Parameters, expInfo, Trl_Durs, frame_rate, Cue_types, retrocue_textobj, Img_Dict, pic_order, trigDict, out_dict, resp_keys, filename, 0)
    else:
        run_task(i_block, Task_Parameters, expInfo, Trl_Durs, frame_rate, Cue_types, retrocue_textobj, Img_Dict, pic_order, {}, out_dict, resp_keys, filename, 0)
    
    # throw end of block trigger and stop save if EEG
    if expInfo['Method']=='EEG':
        win.callOnFlip(port.write,bytes([trigDict['blockEnd']]))
        Fix_Cue.draw()
        win.flip()
        core.wait(2) # show fixation for 2 seconds after last trial
        port.write(trigDict['stopSaveflag'])
        core.wait(1) # wait 1 more second after throwing stop save flag
    
    # change block screen text
    Block_Screen.text = u'Please Take a Short Break\n\nPress Any Key to Begin Block %s When You Are Ready' % (str(i_block+2))

End_Exp.draw()
win.flip()
core.wait(1.5)

if expInfo['Method']=='EEG':
    ##### TTL Pulse trigger
    #win.callonFlip(pport.setData,delay1trig)
    core.wait(1) # wait extra seconds before closing port
    port.close()

core.wait(1.5)
win.close()