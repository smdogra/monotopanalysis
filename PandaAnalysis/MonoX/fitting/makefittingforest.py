#!/usr/bin/env python
from re import sub
from math import *
from array import array
from sys import argv,exit
from os import path,getenv
import os
import ROOT as r
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='make forest')
parser.add_argument('--region',metavar='region',type=str,default=None)
parser.add_argument('--couplings',metavar='couplings',type=str,default=None)
parser.add_argument('--var',metavar='var',type=str,default=None)
parser.add_argument('--input',metavar='input',type=str,default='/eos/cms/store/user/sdogra/monotop/friendtrees/')
#parser.add_argument('--input',metavar='input',type=str,default='/afs/cern.ch/user/s/sdogra/CMSSW_8_0_26_patch1/src/PandaAnalysis/MonoX/fitting/')
#parser.add_argument('--input',metavar='input',type=str,default='/afs/cern.ch/work/s/sdogra/monotop/CMSSW_8_0_25/src/CMGTools/SUSYAnalysis/macros/OUTPUT/')
parser.add_argument('--output',metavar='output',type=str,default='./')
parser.add_argument('--cr',metavar='cr',type=str,default=None)
args = parser.parse_args()
couplings = args.couplings
nddt = args.var
if couplings=='nominal':
    couplings = None
out_region = args.region
region = out_region.split('_')[0]
if region=='test':
    is_test = True 
    region = 'signal'
else:
    is_test = False

argv=[]
import PandaAnalysis.Flat.fitting_forest as forest 
from PandaCore.Tools.Misc import *
import PandaCore.Tools.Functions # kinematics

import PandaAnalysis.MonoX.selection as sel   # Here call the seletionssssssssssssss

#import PandaAnalysis.Monotop.CombinedSelection as sel

toProcess = out_region

#selection of process, the toProcess is defined by the selection key in MonoJetSelection.py
subfolder = ''
if toProcess=='signal':
  subfolder = 'sr/'
elif toProcess=='wmn':
  subfolder = 'cr_w_mu/'
elif toProcess=='wen':
  subfolder = 'cr_w_el/'
elif toProcess=='tme':
  subfolder = 'cr_ttbar_mu/'
elif toProcess=='tem':
  subfolder = 'cr_ttbar_el/'
elif toProcess=='zmm':
  subfolder = 'cr_dimuon/'
elif toProcess=='zee':
  subfolder = 'cr_dielectron/'
elif toProcess=='pho':
  subfolder = 'cr_gamma/'

basedir = args.input #getenv('SKIM_MONOHIGGS_BOOSTED_FLATDIR')+'/%s'%subfolder
if args.cr:
	basedir = args.input+'/%s'%subfolder
lumi = 35900

def f(x):
    return basedir + x + '.root'


#def shift_btags(additional=None,signal=0):
#    shifted_weights = {}
#    #if not any([x in region for x in ['signal','top','w']]):
#    #    return shifted_weights 
#    for shift in ['BUp','BDown','MUp','MDown']:
#        for cent in ['sf_Medbtag']:
#            shiftedlabel = ''
#            if 'sj' in cent:
#                shiftedlabel += 'sj'
#            if 'B' in shift:
#                shiftedlabel += 'btag'
#            else:
#                shiftedlabel += 'mistag'
#            if 'Up' in shift:
#                shiftedlabel += 'Up'
#            else:
#                shiftedlabel += 'Down'
#            weight = sel.weights[out_region+'_'+cent+shift]%lumi
#            if signal==1:
#                weight = weight + signal_eff_SF
#            if signal==2:
#                weight = weight + ttbar_mistag_SF
#            if additional:
#                weight = tTIMES(weight,additional)
#            shifted_weights[shiftedlabel] = weight
 #   return shifted_weights

#def shiftScalesPDF(additional=None,signal=0):
#	shifted_weights = {}
#	for shift in ['ScaleUp','ScaleDown','PDFUp','PDFDown']:
#         	weight = sel.weights[out_region+'_'+shift]%lumi
#                if signal==1:
#       		      weight = weight + signal_eff_SF
#    		if signal==2:
#                      weight = weight + ttbar_mistag_SF
#
#                shiftedlabel = '_'
#                if 'Scale' in shift:
#		      shiftedlabel += 'Scale'
#	        else:
#	  	      shiftedlabel += 'PDF'
#	        if 'Up' in shift:
#		      shiftedlabel += 'Up'
#	        else:
#		      shiftedlabel += 'Down'
#		if additional:
#                      weight = tTIMES(weight,additional)
#                shifted_weights[shiftedlabel] = weight
#        return shifted_weights

#vmap definition
vmap={}
mc_vmap = {'GenMET':'GenMET'}
if region in ['signal','test','bjet2']:
    u,uphi, = ('MT','MT')
#elif 'pho' in region:
#    u,uphi = ('pfUAmag','dphipfUA')
#elif 'wmn'or 'wen' in region:
#    u,uphi = ('pfUWmag','dphipfUW')
#elif 'tem' or 'tme' in region:
#    u,uphi = ('pfUWWmag','dphipfUWW')
#elif 'zee'  or 'zmn' in region:
#    u,uphi = ('pfUZmag','dphipfUZ')

vmap['mTw'] = 'min(%s,9999.9999)'%u
if region in ['signal','test','bjet2']:
    u,uphi, = ('MET','MET')
vmap['MET'] = 'min(%s,9999.9999)'%u
if region in ['signal','test','bjet2']:
    u,uphi, = ('nLep','nLep')
vmap['nLep'] = 'min(%s,9999.9999)'%u
if region in ['signal','test','bjet2']:
    u,uphi, = ('nBJet','nBJet')
vmap['nBJet'] = 'min(%s,9999.9999)'%u


#weights
#print sel.weights[out_region]%lumi

#weights = {'nominal' : sel.weights[out_region]%lumi}
weights = {'nominal' : sel.weights[out_region]}

#if couplings:
#    weights['nominal'] = tTIMES(weights['nominal'],couplings)

#if 'tag' in out_region:
#    weights.update(shift_btags(couplings))

		
region = out_region
factory = forest.RegionFactory(name = region if not(is_test) else 'test',
                               cut = sel.cuts[region],
                               variables = vmap, 
                               mc_variables = mc_vmap, 
                               mc_weights = weights
                               )

#Process and creation of new ntuples process
if is_test:
    factory.add_process(f('Diboson'),'Diboson')

#photon CR
#elif region=='pho':
#    factory.add_process(f('GJets'),'Pho')
#    factory.add_process(f('SinglePhoton'),'Data',is_data=True)
    #factory.add_process(f('SinglePhoton'),'QCD',is_data=True,
    #                    extra_weights='sf_phoPurity',extra_cut=sel.triggers['pho'])
#    factory.add_process(f('QCD'),'QCD')

elif out_region not in ['signal','bjet2']:
    factory.add_process(f('ZtoNuNu'),'Zvv')
    factory.add_process(f('ZJets'),'Zll')
    factory.add_process(f('WJets'),'Wlv')
    factory.add_process(f('SingleTop'),'ST')
    factory.add_process(f('Diboson'),'Diboson')
    factory.add_process(f('QCD'),'QCD')

    if 'zee' in region or 'tem' in region or 'wen' in region:
        factory.add_process(f('SingleElectron'),'Data',is_data=True,extra_cut=sel.eleTrigger)
    	#factory.add_process(f('TTbar'),'ttbar') #need ttbar_weight

    if 'zmm' in region or 'tme' in region or 'wmn' in region:
        factory.add_process(f('MET'),'Data',is_data=True,extra_cut=sel.metTrigger)
    	#factory.add_process(f('TTbar'),'ttbar') #without ttbar weight
	
elif out_region in ['signal','bjet2']:
    print "Start adding the process ..."
    factory.add_process(f('evVarFriend_WJetsToLNu_HT100to200_ext'),'WJets100')
    factory.add_process(f('evVarFriend_WJetsToLNu_HT200to400_ext'),'WJets200')
    factory.add_process(f('evVarFriend_WJetsToLNu_HT400to600'),'WJets400')
    factory.add_process(f('evVarFriend_WJetsToLNu_HT600to800'),'WJets600')
    factory.add_process(f('evVarFriend_WJetsToLNu_HT800to1200_ext'),'WJets800')
    factory.add_process(f('evVarFriend_WJetsToLNu_HT1200to2500'),'WJets1200')
    factory.add_process(f('evVarFriend_WJetsToLNu_HT2500toInf'),'WJets2500')
    factory.add_process(f('evVarFriend_TTJets_DiLepton'),'TTbar_DiLep')
    
    factory.add_process(f('evVarFriend_DYJetsToLL_M50_HT100to200'),  'DYJetsToLL_M50_HT100to200')
    factory.add_process(f('evVarFriend_DYJetsToLL_M50_HT1200to2500'),  'DYJetsToLL_M50_HT1200to2500')
    factory.add_process(f('evVarFriend_DYJetsToLL_M50_HT200to400'),  'DYJetsToLL_M50_HT200to400')
    factory.add_process(f('evVarFriend_DYJetsToLL_M50_HT2500toInf'),  'DYJetsToLL_M50_HT2500toInf')
    factory.add_process(f('evVarFriend_DYJetsToLL_M50_HT400to600'),  'DYJetsToLL_M50_HT400to600')
    factory.add_process(f('evVarFriend_DYJetsToLL_M50_HT600to800'),  'DYJetsToLL_M50_HT600to800')
    factory.add_process(f('evVarFriend_DYJetsToLL_M50_HT70to100'),  'DYJetsToLL_M50_HT70to100')
    factory.add_process(f('evVarFriend_DYJetsToLL_M50_HT800to1200'),  'DYJetsToLL_M50_HT800to1200')
    
    factory.add_process(f('evVarFriend_QCD_HT1000to1500'),  'QCD_HT1000to1500')
    factory.add_process(f('evVarFriend_QCD_HT100to200'),  'QCD_HT100to200')
    factory.add_process(f('evVarFriend_QCD_HT1500to2000'),  'QCD_HT1500to2000')
    factory.add_process(f('evVarFriend_QCD_HT2000toInf'),  'QCD_HT2000toInf')
    factory.add_process(f('evVarFriend_QCD_HT200to300'),  'QCD_HT200to300')
    factory.add_process(f('evVarFriend_QCD_HT300to500'),  'QCD_HT300to500')
    factory.add_process(f('evVarFriend_QCD_HT500to700'),  'QCD_HT500to700')
    factory.add_process(f('evVarFriend_QCD_HT700to1000'),  'QCD_HT700to1000')
    
    factory.add_process(f('evVarFriend_TBar_tWch_ext1'),  'TBar_tWch')
    factory.add_process(f('evVarFriend_TBar_tch_powheg'),  'TBar_tch')
    
    factory.add_process(f('evVarFriend_TTJets_SingleLeptonFromT'),  'TTJets_SingleLeptonFromT')
    factory.add_process(f('evVarFriend_TTJets_SingleLeptonFromTbar'),  'TTJets_SingleLeptonFromTbar')
    factory.add_process(f('evVarFriend_TTWToLNu_ext'),  'TTWToLNu')
    factory.add_process(f('evVarFriend_TTZToLLNuNu'),  'TTZToLLNuNu')
    factory.add_process(f('evVarFriend_TToLeptons_sch'),  'TToLeptons_sch')
    factory.add_process(f('evVarFriend_T_tWch_ext1'),  'T_tWch')
    factory.add_process(f('evVarFriend_T_tch_powheg'),  'T_tch')
    
    factory.add_process(f('evVarFriend_WWTo2L2Nu'),  'WWTo2L2Nu')
    factory.add_process(f('evVarFriend_WWToLNuQQ'),  'WWToLNuQQ')
    factory.add_process(f('evVarFriend_WZTo1L3Nu'),  'WZTo1L3Nu')
    factory.add_process(f('evVarFriend_ZZTo2L2Nu'),  'ZZTo2L2Nu')
    factory.add_process(f('evVarFriend_ZZTo2L2Q'),  'ZZTo2L2Q')
    factory.add_process(f('evVarFriend_WJetsToLNu_LO_ext2'),  'WJetsToLNu_LO')
    
    factory.add_process(f('evVarFriend_SingleElectron_Run2016B_23Sep2016'),'ELec2016B',  is_data=True,extra_cut=sel.Data_Trig)
    factory.add_process(f('evVarFriend_SingleElectron_Run2016C_23Sep2016_v1'),'ELec2016C',is_data=True,extra_cut=sel.Data_Trig)
    factory.add_process(f('evVarFriend_SingleElectron_Run2016D_23Sep2016_v1'),'Elec2016D',is_data=True,extra_cut=sel.Data_Trig)
    factory.add_process(f('evVarFriend_SingleElectron_Run2016E_23Sep2016_v1'),'Elec2016E',is_data=True,extra_cut=sel.Data_Trig)
    factory.add_process(f('evVarFriend_SingleElectron_Run2016F_23Sep2016_v1'),'Elec2016F',is_data=True,extra_cut=sel.Data_Trig)
    factory.add_process(f('evVarFriend_SingleElectron_Run2016G_23Sep2016_v1'),'Elec2016G',is_data=True,extra_cut=sel.Data_Trig)
    
    factory.add_process(f('evVarFriend_SingleMuon_Run2016B_23Sep2016_v1'),'Muon2016B',is_data=True,extra_cut=sel.Data_Trig)
    factory.add_process(f('evVarFriend_SingleMuon_Run2016C_23Sep2016_v1'),'Muon2016C',is_data=True,extra_cut=sel.Data_Trig)
    factory.add_process(f('evVarFriend_SingleMuon_Run2016D_23Sep2016_v1'),'Muon2016D',is_data=True,extra_cut=sel.Data_Trig)
    factory.add_process(f('evVarFriend_SingleMuon_Run2016E_23Sep2016_v1'),'Muon2016E',is_data=True,extra_cut=sel.Data_Trig)
    factory.add_process(f('evVarFriend_SingleMuon_Run2016F_23Sep2016_v1'),'Muon2016F',is_data=True,extra_cut=sel.Data_Trig)
    factory.add_process(f('evVarFriend_SingleMuon_Run2016G_23Sep2016_v1'),'Muon2016G',is_data=True,extra_cut=sel.Data_Trig)
    


    print "Done..."
    
  
'''
elif out_region=='signal_vector':
    signal_files = glob(basedir+'/Vector*root')
    if couplings:
        out_region += '_'+couplings
    for f in signal_files:
        fname = f.split('/')[-1].replace('.root','')
        signame = fname
        replacements = {
            'Vector_MonoTop_NLO_Mphi-':'',
            '_gSM-0p25_gDM-1p0_13TeV-madgraph':'',
            '_Mchi-':'_',
        }
        for k,v in replacements.iteritems():
            signame = signame.replace(k,v)
        factory.add_process(f,signame)
elif out_region=='signal_scalar':
    signal_files = glob(basedir+'/Scalar*root')
    if couplings:
        out_region += '_'+couplings
    for f in signal_files:
        fname = f.split('/')[-1].replace('.root','')
        signame = fname
        replacements = {
            'Scalar_MonoTop_LO_Mphi-':'',
            '_13TeV-madgraph':'',
            '_Mchi-':'_',
        }
        for k,v in replacements.iteritems():
            signame = signame.replace(k,v)
        factory.add_process(f,'scalar_'+signame)
elif out_region=='signal_thq':
    factory.add_process(f('thq'),'thq')
elif out_region=='signal_stdm':
    for m in [300,500,1000]:
        factory.add_process(f('ST_tch_DM-scalar_LO-%i_1-13_TeV'%m),'stdm_%i'%m)
'''
forestDir = args.output #'/data/t3home000/mcremone/lpc/jorgem/skim/monohiggs_boosted/'
os.system('mkdir -p %s/%s'%(forestDir,'fittingForest'))
factory.run(forestDir+'/fittingForest/fittingForest_%s.root'%out_region)

