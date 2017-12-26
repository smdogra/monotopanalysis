#!/usr/bin/env python

from os import system,mkdir,getenv
from sys import argv,exit
import argparse

basedir = getenv('PANDA_FLATDIR')
figsdir = basedir+'/figs'

parser = argparse.ArgumentParser(description='plot stuff')
parser.add_argument('--indir',metavar='indir',type=str,default=basedir)
parser.add_argument('--outdir',metavar='outdir',type=str,default=figsdir)
parser.add_argument('--cut',metavar='cut',type=str,default=None)
parser.add_argument('--sel',metavar='sel',type=str,default='tag')
parser.add_argument('--pt',metavar='pt',type=str,default='inc')
args = parser.parse_args()

figsdir = args.outdir
basedir = args.indir
argv=[]

import ROOT as root
from PandaCore.Tools.Load import *
from PandaCore.Tools.Misc import *
tcut = root.TCut
Load('Drawers','PlotUtility')

### SET GLOBAL VARIABLES ###
lumi = 12918.
logy=False
nlo = 'sf_ewkV*sf_qcdV'
if args.sel=='mistag':
  cut = 'nFatjet==1 && fj1Pt>250 && fj1MaxCSV<0.46 && nLooseLep==1 && nTightMuon==1 && nLooseElectron==0 && nLoosePhoton==0 && nTau==0 && UWmag>250 && isojetNBtags==0'
  weight = '%f*normalizedWeight*sf_pu*sf_lep*%s*sf_sjbtag0*sf_btag0*sf_tt*sf_metTrig'%(lumi,nlo)
  label = 'mistag_'
elif args.sel=='photon':
  cut = 'nFatjet==1 && fj1Pt>250 && nLooseLep==0 && nLoosePhoton==1 && loosePho1IsTight==1 && nTau==0 && UAmag>250'
  weight = '%f*normalizedWeight*sf_pu*sf_lep*%s*sf_tt*sf_phoTrig*sf_pho'%(lumi,nlo)
  label = 'photon_'
else:
  cut = 'nFatjet==1 && fj1Pt>250 && fj1MaxCSV>0.46 && nLooseLep==1 && nTightMuon==1 && nLooseElectron==0 && nLoosePhoton==0 && nTau==0 && UWmag>250 && isojetNBtags==1'
  weight = '%f*normalizedWeight*sf_pu*sf_lep*%s*sf_sjbtag1*sf_btag1*sf_tt*sf_metTrig'%(lumi,nlo)
  label = 'tag_'

if not args.cut:
  label += 'noCut_'
  plotlabel = '40 GeV < m_{SD}'
  cut = tAND(cut,'fj1MSD>40')
elif args.cut=='mass':
  cut = tAND(cut,'fj1MSD>110 && fj1MSD<210')
  label += 'massCut_'
  plotlabel = '110 < m_{SD} < 210 GeV'
elif args.cut=='massW':
  cut = tAND(cut,'fj1MSD>50 && fj1MSD<100')
  label += 'massWCut_'
  plotlabel = '50 < m_{SD} < 100 GeV'

if args.pt=='inc':
  figsdir += '/inc/'
elif args.pt=='lo':
  figsdir += '/lo/'
  cut = tAND(cut,'250 < fj1Pt && fj1Pt<450')
  plotlabel = '#splitline{%s}{250 < p_{T} < 450 GeV}'%plotlabel
elif args.pt=='hi':
  figsdir += '/hi/'
  cut = tAND(cut,'450 < fj1Pt && fj1Pt<1000')
  plotlabel = '#splitline{%s}{450 < p_{T} < 1000 GeV}'%plotlabel

### LOAD PLOTTING UTILITY ###
plot = root.PlotUtility()
plot.Stack(True)
plot.InitLegend()
plot.SetCut(tcut(cut))
plot.Ratio(1) 
plot.FixRatio(.25)
plot.SetLumi(lumi/1000)
plot.DrawMCErrors(True)
plot.SetTDRStyle()
plot.SetNormFactor(False)
plot.AddCMSLabel()
plot.AddLumiLabel()
if plotlabel:
  plot.AddPlotLabel(plotlabel,.18,.77,False,42,.04)
plot.SetMCWeight(weight)


### DEFINE PROCESSES ###
wjetsq     = root.Process('W+q',root.kWjets); wjetsq.additionalCut = root.TCut('abs(fj1HighestPtGen)!=21')
wjetsg     = root.Process('W+g',root.kExtra2); wjetsg.additionalCut = root.TCut('abs(fj1HighestPtGen)==21')
diboson   = root.Process('Diboson',root.kDiboson)
ttbar     = root.Process('Top [matched]',root.kTTbar); ttbar.additionalCut = root.TCut('(fj1IsMatched==1&&fj1GenSize<1.44)')
ttbarunmatched     = root.Process('Top [unmatched]',root.kExtra1); ttbarunmatched.additionalCut = root.TCut('(fj1IsMatched==0||fj1GenSize>1.44)')
#singletop = root.Process('Single t',root.kST)
qcd       = root.Process("QCD",root.kQCD)
data      = root.Process("Data",root.kData)
gjetsq    = root.Process('#gamma+q',root.kGjets); gjetsq.additionalCut = root.TCut('abs(fj1HighestPtGen)!=21')
gjetsg    = root.Process('#gamma+g',root.kExtra3); gjetsg.additionalCut = root.TCut('abs(fj1HighestPtGen)==21')
if args.sel=='photon':
  data.additionalCut = root.TCut('(trigger&4)!=0')
  qcd.useCommonWeight=False
  qcd.additionalWeight = root.TCut('sf_phoPurity')
  qcd.additionalCut = root.TCut('(trigger&4)!=0')
  processes = [qcd,gjetsg,gjetsq]
else:
  data.additionalCut = root.TCut('(trigger&1)!=0')
  processes = [diboson,wjetsg,wjetsq,ttbarunmatched,ttbar]

### ASSIGN FILES TO PROCESSES ###
if args.sel=='photon':
  gjetsq.AddFile(basedir+'GJets.root')
  gjetsg.AddFile(basedir+'GJets.root')
  data.AddFile(basedir+'SinglePhoton.root') 
  qcd.AddFile(basedir+'SinglePhoton.root') 
else:
  data.AddFile(basedir+'MET.root') 
  qcd.AddFile(basedir+'QCD.root')
  wjetsq.AddFile(basedir+'WJets.root')
  wjetsg.AddFile(basedir+'WJets.root')
  diboson.AddFile(basedir+'Diboson.root')
  ttbar.AddFile(basedir+'TTbar.root')
  ttbar.AddFile(basedir+'SingleTop.root')
  ttbarunmatched.AddFile(basedir+'TTbar.root')
  ttbarunmatched.AddFile(basedir+'SingleTop.root')
#  singletop.AddFile(basedir+'SingleTop.root')
processes.append(data)

for p in processes:
  plot.AddProcess(p)


'''
plot.AddDistribution(root.Distribution('fj1MSDL2L3',40,450,20,'L3L3-corr fatjet m_{SD} [GeV]','Events/12.5 GeV'))

plot.AddDistribution(root.Distribution('fj1MSD',40,450,20,'fatjet m_{SD} [GeV]','Events/12.5 GeV'))

plot.AddDistribution(root.Distribution('UWmag',250,500,20,'W recoil [GeV]','Events'))

plot.AddDistribution(root.Distribution('UAmag',250,500,20,'#gamma recoil [GeV]','Events'))
'''

#plot.AddDistribution(root.Distribution('npv',0,50,25,'npv','Events'))

#plot.AddDistribution(root.Distribution('top_allv2_bdt',-1.3,1,23,'Top 50ECF BDT','Events'))

#plot.AddDistribution(root.Distribution('top_ecfv14_bdt',-1,1.,20,'Top ECF+#tau_{32}^{SD}+f_{rec} BDT v2','Events'))
#plot.AddDistribution(root.Distribution('top_ecfv13_bdt',-1.3,1.,23,'Top ECF+#tau_{32}^{SD} BDT v2','Events'))
#plot.AddDistribution(root.Distribution('top_ecfv12_bdt',-1,1.,20,'Top ECF BDT v2','Events'))

plot.AddDistribution(root.Distribution('top_ecf_bdt',-1,1.,20,'Top ECF+#tau_{32}^{SD}+f_{rec} BDT','Events'))
#plot.AddDistribution(root.Distribution('top_ecfv8_bdt',-1,1.,20,'Top ECF+#tau_{32}^{SD}+f_{rec} BDT','Events'))
#plot.AddDistribution(root.Distribution('top_ecfv7_bdt',-1.3,1.,23,'Top ECF+#tau_{32}^{SD} BDT','Events'))
#plot.AddDistribution(root.Distribution('top_ecfv6_bdt',-1,1.,20,'Top ECF BDT','Events'))

#plot.AddDistribution(root.Distribution('top_ecf_bdt',-0.5,.5,20,'Top ECF BDT','Events'))
'''
plot.AddDistribution(root.Distribution('fj1Tau32SD',0,1,20,'Groomed #tau_{32}','Events',999,-999,'tau32SD'))

plot.AddDistribution(root.Distribution('fj1Tau32',0,1,20,'#tau_{32}','Events',999,-999,'tau32'))

plot.AddDistribution(root.Distribution('jet1Pt',15,500,20,'leading jet p_{T} [GeV]','Events'))

plot.AddDistribution(root.Distribution('nJet',-0.5,8.5,9,'N_{jet}','Events'))

plot.AddDistribution(root.Distribution('puppimet',0,750,20,'MET [GeV]','Events/37.5 GeV'))

plot.AddDistribution(root.Distribution('fj1Pt',250,1000,20,'fatjet p_{T} [GeV]','Events/37.5 GeV'))

plot.AddDistribution(root.Distribution('fj1HTTMass',40,450,20,'fatjet m_{HTT} [GeV]','Events/12.5 GeV'))

plot.AddDistribution(root.Distribution('fj1HTTFRec',0,1,20,'HTT f_{rec}','Events'))

plot.AddDistribution(root.Distribution('fj1ECFN_1_2_20/pow(fj1ECFN_1_2_10,2.00)',2,10,20,'e(1,2,2)/e(1,2,1)^{2}','Events',999,-999,'input0'))
plot.AddDistribution(root.Distribution('fj1ECFN_1_3_40/fj1ECFN_2_3_20',0,1,20,'e(1,3,4)/e(2,3,2)','Events',999,-999,'input1'))
plot.AddDistribution(root.Distribution('fj1ECFN_3_3_10/pow(fj1ECFN_1_3_40,.75)',.5,4,20,'e(3,3,1)/e(1,3,4)^{3/4}','Events',999,-999,'input2'))
plot.AddDistribution(root.Distribution('fj1ECFN_3_3_10/pow(fj1ECFN_2_3_20,.75)',0.4,1.4,20,'e(3,3,1)/e(2,3,2)^{3/4}','Events',999,-999,'input3'))
plot.AddDistribution(root.Distribution('fj1ECFN_3_3_20/pow(fj1ECFN_3_3_40,.5)',0,.25,20,'e(3,3,2)/e(3,3,4)^{1/2}','Events',999,-999,'input4'))
plot.AddDistribution(root.Distribution('fj1ECFN_1_4_20/pow(fj1ECFN_1_3_10,2)',0,2,20,'e(1,4,2)/e(1,3,1)^{2}','Events',999,-999,'input5'))
plot.AddDistribution(root.Distribution('fj1ECFN_1_4_40/pow(fj1ECFN_1_3_20,2)',0,2.5,20,'e(1,4,4)/e(1,3,2)^{2}','Events',999,-999,'input6'))
plot.AddDistribution(root.Distribution('fj1ECFN_2_4_05/pow(fj1ECFN_1_3_05,2)',1.25,2.5,20,'e(2,4,0.5)/e(1,3,0.5)^{2}','Events',999,-999,'input7'))
plot.AddDistribution(root.Distribution('fj1ECFN_2_4_10/pow(fj1ECFN_1_3_10,2)',1,4,20,'e(2,4,1)/e(1,3,1)^{2}','Events',999,-999,'input8'))
plot.AddDistribution(root.Distribution('fj1ECFN_2_4_10/pow(fj1ECFN_2_3_05,2)',0,1.5,20,'e(2,4,1)/e(2,3,0.5)^{2}','Events',999,-999,'input9'))
plot.AddDistribution(root.Distribution('fj1ECFN_2_4_20/pow(fj1ECFN_1_3_20,2)',0,5,20,'e(2,4,2)/e(1,3,2)^{2}','Events',999,-999,'input10'))

#plot.AddDistribution(root.Distribution("1",0,2,1,"dummy","dummy"))
'''

'''
plot.AddDistribution(root.Distribution('fj1ECFN_1_4_10/pow(fj1ECFN_1_3_05,2)',0.6,1.5,20,'_{1}e_{4}^{1.}/(_{1}e_{3}^{0.5})^{2}','Events',999,-999,'input0'))

plot.AddDistribution(root.Distribution('fj1ECFN_2_4_20/pow(fj1ECFN_1_3_20,2)',0.25,3.,20,'_{2}e_{4}^{2.}/(_{1}e_{3}^{02.})^{2}','Events',999,-999,'input1'))

plot.AddDistribution(root.Distribution('fj1ECFN_2_4_10/pow(fj1ECFN_1_3_10,2)',1.,3.,20,'_{2}e_{4}^{1.}/(_{1}e_{3}^{01.})^{2}','Events',999,-999,'input2'))

plot.AddDistribution(root.Distribution('fj1ECFN_3_3_10/pow(fj1ECFN_1_2_20,1.5)',0.,.4,20,'_{3}e_{3}^{1.}/(_{1}e_{2}^{2.})^{3/2}','Events',999,-999,'input3'))

plot.AddDistribution(root.Distribution('fj1ECFN_2_4_10/fj1ECFN_1_2_20',0.,0.015,20,'_{2}e_{4}^{1.}/_{1}e_{2}^{2.}','Events',999,-999,'input4'))

plot.AddDistribution(root.Distribution('fj1ECFN_1_4_20/pow(fj1ECFN_1_3_10,2)',0.25,1.5,20,'_{1}e_{4}^{2.}/(_{1}e_{3}^{1.})^{2}','Events',999,-999,'input5'))

plot.AddDistribution(root.Distribution('fj1ECFN_2_4_05/pow(fj1ECFN_1_3_05,2)',1.25,2.5,20,'_{2}e_{4}^{0.5}/(_{1}e_{3}^{0.5})^{2}','Events',999,-999,'input6'))

plot.AddDistribution(root.Distribution('fj1ECFN_1_3_10/fj1ECFN_2_3_05',0.25,.9,20,'_{1}e_{3}^{1.}/_{2}e_{3}^{0.5}','Events',999,-999,'input7'))

plot.AddDistribution(root.Distribution('fj1ECFN_3_3_10/pow(fj1ECFN_3_3_20,.5)',0.,.35,20,'_{3}e_{3}^{1.}/(_{3}e_{3}^{2.})^{1/2}','Events',999,-999,'input8'))

plot.AddDistribution(root.Distribution('fj1ECFN_3_3_05/pow(fj1ECFN_1_2_05,3.)',1.,3.,20,'_{3}e_{3}^{0.5}/(_{1}e_{2}^{0.5})^{3}','Events',999,-999,'input9'))
'''
'''
plot.AddDistribution(root.Distribution('fj1ECFN_2_4_20/pow(fj1ECFN_1_3_20,2)',0.25,2.5,20,'N_{3}(#beta=2.0)','Events',999,-999,'N3_20'))

plot.AddDistribution(root.Distribution('fj1ECFN_2_4_10/pow(fj1ECFN_1_3_10,2)',0.75,2.5,20,'N_{3}(#beta=1.0)','Events',999,-999,'N3_10'))

plot.AddDistribution(root.Distribution('fj1ECFN_2_4_40/pow(fj1ECFN_1_3_40,2)',0,2,20,'N_{3}(#beta=4.0)','Events',999,-999,'N3_40'))
'''

# plot.AddDistribution(root.Distribution('fj1Tau32SD',0,1,20,'Groomed #tau_{32}','Events',999,-999,'tau32SD'))

# plot.AddDistribution(root.Distribution('fj1Tau32',0,1,20,'#tau_{32}','Events',999,-999,'tau32'))

### DRAW AND CATALOGUE ###
plot.DrawAll(figsdir+'/'+label)
