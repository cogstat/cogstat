# -*- coding: utf-8 -*-
"""Test for the outputs.

Run a set of analysis for all available languages and and save them in pdfs.
"""

# TODO more informative message when task fails

print('Running output language test.')

import sys
import os
import glob
from PyQt4 import QtGui
import gettext
import cogstat
from cogstat import cogstat_gui
from cogstat import cogstat_stat


def available_langs(domain=None, localedir=None):
    if domain is None:
        domain = gettext._current_domain
    if localedir is None:
        localedir = gettext._default_localedir
    files = glob.glob(os.path.join(localedir, '*', 'LC_MESSAGES', '%s.mo' % domain))
    langs = [file.split(os.path.sep)[-3] for file in files]
    return langs

langs = ['en']+available_langs(domain='cogstat', localedir='../locale')
print('Available languages:', langs)

app = QtGui.QApplication(sys.argv)

lang_before_test = cogstat_gui.csc.language

for lang in langs:
    print('** Running language: %s **' % lang)
    
    # Set the language
    cogstat_gui.csc.save('language', lang)
    reload(cogstat.csc)
    reload(cogstat_gui.cogstat)
    reload(cogstat_stat)
    reload(cogstat_gui)
    cogstat.output_type = 'gui'

    cs = cogstat_gui.StatMainWindow()
    cs.open_file('data/test_data.csv')

    cs.print_data()
    cs._print_data_brief()

    try:
        print('Explore variables block.')
        cs.explore_variable(['A'])
        cs.explore_variable(['D'])
        cs.explore_variable(['G'])
    except:
        print('Error! Explore variables block failed.')

    try:
        print('Explore variable pairs block.')
        cs.explore_variable_pair(['A', 'B'])
        cs.explore_variable_pair(['D', 'E'])
        cs.explore_variable_pair(['G', 'H'])
    except:
        print('Error! Explore variable pairs block failed.')

    cs.pivot(['A'], row_names=['G'], col_names=['H'], page_names=[], function='N')

    try:
        print('Compare variables block.')
        cs.compare_variables(['A', 'B'])
        cs.compare_variables(['A', 'B', 'C1'])  # 'C' can't be used because of patsy
        cs.compare_variables(['D', 'E'])
        cs.compare_variables(['D', 'E', 'F'])
        cs.compare_variables(['G', 'H'])
        cs.compare_variables(['G', 'H', 'I'])
    except:
        print('Error! Compare variables block failed.')

    try:
        print('Compare groups block.')
        cs.compare_groups(['A'], ['H'])
        cs.compare_groups(['A'], ['G'])
        cs.compare_groups(['D'], ['H'])
        cs.compare_groups(['D'], ['G'])
        cs.compare_groups(['I'], ['H'])
    except:
        print('Error! Compare groups block failed.')

    try:
        cs.print_versions()
    except:
        print('Error! Version print failed.')

    try:
        cs.save_result_as(filename='test_output/test_output__'+lang+'__'+cogstat.__version__+'.pdf')
    except:
        print('Error! Pdf save failed.')

cogstat_gui.csc.save('language', lang_before_test)

print('Output language test finished.')

