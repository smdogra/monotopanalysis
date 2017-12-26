#!/usr/bin/env python

from os import system,getenv
from sys import argv
import argparse
from collections import namedtuple

### SET GLOBAL VARIABLES ###
basedir = getenv('PANDA_FLATDIR')+'/' 
parser = argparse.ArgumentParser(description='plot stuff')
parser.add_argument('--basedir',metavar='basedir',type=str,default=None)
parser.add_argument('--outdir',metavar='outdir',type=str,default=None)
parser.add_argument('--cat',metavar='cat',type=str)
parser.add_argument('--wp',metavar='wp',type=str)
parser.add_argument('--region',metavar='region',type=str,default=None)
parser.add_argument('--syst',metavar='syst',type=str,default=None)
parser.add_argument('--var',metavar='var',type=str,default='')
args = parser.parse_args()

ylabel = 'Events/bin'

lumi = 35800.
region = args.region
sname = argv[0]
if args.basedir:
    basedir = args.basedir

argv=[]
import ROOT as root
root.gROOT.SetBatch()
from PandaCore.Tools.Misc import *
import PandaCore.Tools.Functions
from PandaCore.Drawers.plot_utility import *
import PandaAnalysis.Tagging.Selection as sel

### DEFINE REGIONS ###
prefix = region+'_'+args.cat+'_'

op = '>' if args.cat=='pass' else '<'
wp = 0.45 if args.wp=='tight' else 0.1
cut = 'top_ecf_bdt%s%.2f'%(op,wp)
cut = tAND(sel.cuts[args.region],cut)
weight = sel.weights[region]%lumi
label = 'BDT %s %.2f'%(op,wp)

pt_name = 'fj1Pt'
msd_name = 'fj1MSD'
merge_rad = 1.44
if args.syst:
    prefix += args.syst + args.var + '_'
    if args.syst == 'smeared':
        pt_name = 'fj1PtSmeared'
        msd_name = 'fj1MSDSmeared'
        label = '#splitline{%s}{Smeared p_{T} and m_{SD}}'%(label)
    elif args.syst == 'scale':
        pt_name += 'Scale' + args.var
        msd_name += 'Scale' + args.var
        label = '#splitline{%s}{JES %s}'%(label,args.var)
    elif args.syst == 'smearedSJ':
        pt_name = 'fj1PtSmeared_sj'
        msd_name = 'fj1MSDSmeared_sj'
        label = '#splitline{%s}{Smeared subjet p_{T} and m_{SD}}'%(label)
    elif args.syst == 'scaleSJ':
        pt_name += 'Scale' + args.var + '_sj'
        msd_name += 'Scale' + args.var + '_sj'  
        label = '#splitline{%s}{Subjet JES %s}'%(label,args.var)
    elif args.syst == 'merge':
        label = '#splitline{%s}{Merging Radius %s}'%(label,args.var)
        if args.var=='Up':
            merge_rad = 2.25
        else:
            merge_rad = 1
    elif args.syst == 'topPt':
        weight = weight.replace('sf_tt','1')
        label = '#splitline{%s}{No top p_{T} weight}'%(label)
    elif args.syst == 'btag':
        weight = sub('sf_([sj]*)btag([01])','sf_\\1btag\\2B%s'%(args.var),weight)
        label = '#splitline{%s}{b-tag %s}'%(label,args.var)
    cut = cut.replace('fj1Pt',pt_name)
    cut = cut.replace('fj1MSD',msd_name)

### LOAD PLOTTING UTILITY ###
plot = PlotUtility()
plot.Stack(True)
plot.Ratio(True)
plot.FixRatio(0.4)
plot.SetTDRStyle()
plot.InitLegend()
plot.DrawMCErrors(True)
plot.AddCMSLabel()
plot.cut = cut
plot.SetLumi(lumi/1000)
plot.AddLumiLabel(True)
plot.do_overflow = False
plot.do_underflow = False
plot.AddPlotLabel(label,.18,.77,False,42,.04)

plot.mc_weight = weight

### DEFINE PROCESSES ###
wjetsq         = Process('W+q',root.kWjets); wjetsq.additional_cut = 'abs(fj1HighestPtGen)!=21'
wjetsg         = Process('W+g',root.kExtra2); wjetsg.additional_cut = 'abs(fj1HighestPtGen)==21'

zjetsq         = Process('Z+q',root.kZjets); zjetsq.additional_cut = 'abs(fj1HighestPtGen)!=21'
zjetsg         = Process('Z+g',root.kExtra5); zjetsg.additional_cut = 'abs(fj1HighestPtGen)==21'

gjetsq         = Process('#gamma+q',root.kGjets); gjetsq.additional_cut = 'abs(fj1HighestPtGen)!=21'
gjetsg         = Process('#gamma+g',root.kExtra3); gjetsg.additional_cut = 'abs(fj1HighestPtGen)==21'

diboson        = Process('Diboson',root.kDiboson)
qcd             = Process("QCD",root.kQCD)

top            = Process('Top [t-matched]',root.kTTbar)
top.additional_cut   = 'fj1IsMatched==1 && fj1GenSize<{0}'.format(merge_rad)

wtop           = Process('Top [W-matched]',root.kExtra1)
wtop.additional_cut  = '(fj1IsMatched==0 || fj1GenSize>{0}) && fj1IsWMatched==1 && fj1GenWSize<{0}'.format(merge_rad)

untop          = Process('Top [unmatched]',root.kExtra4)
untop.additional_cut = tNOT(tOR(top.additional_cut, wtop.additional_cut))
#'!((fj1IsMatched==1 && fj1GenSize>{0} && fj1IsWMatched && fj1GenWSize<{0})||(fj1IsMatched==1&&fj1GenSize<{0}))'.format(merge_rad)

data            = Process("Data",root.kData)
if region=='photon':
    processes = [qcd,gjetsg,gjetsq,data]
elif region=='mistag':
    processes = [qcd,diboson,zjetsg,zjetsq,untop,wtop,top,wjetsg,wjetsq,data]
elif region=='tag':
    processes = [qcd,diboson,zjetsg,zjetsq,wjetsg,wjetsq,untop,wtop,top,data]
else:
    processes = [qcd,diboson,wjetsg,wjetsq,untop,wtop,top,zjetsg,zjetsq,data]

### ASSIGN FILES TO PROCESSES ###
for p in [wjetsq,wjetsg]:
    p.add_file(basedir+'WJets.root')

for p in [zjetsq,zjetsg]:
    p.add_file(basedir+'ZJets.root')

for p in [gjetsq,gjetsg]:
    p.add_file(basedir+'GJets.root')

for p in [top,wtop,untop]:
    p.add_file(basedir+'TTbar.root')
    p.add_file(basedir+'SingleTop.root')

diboson.add_file(basedir+'Diboson.root')

if 'pho' in region:
    qcd.add_file(basedir+'SinglePhoton.root')
    qcd.additional_cut = sel.triggers['pho']
    qcd.use_common_weight = False
    qcd.additional_weight = 'sf_phoPurity'
    data.add_file(basedir+'SinglePhoton.root')
    data.additional_cut = sel.triggers['pho']
else:
    qcd.add_file(basedir+'QCD.root')
    data.add_file(basedir+'MET.root')
    data.additional_cut = sel.triggers['met']

for p in processes:
    plot.add_process(p)

plot.add_distribution(FDistribution(msd_name,50,350,20,'fatjet m_{SD} [GeV]',ylabel))

### DRAW AND CATALOGUE ###
plot.draw_all(args.outdir+'/'+prefix)
