#!/usr/bin/env python

from re import sub
from sys import argv,exit
from os import system,getenv
import json

debug_level = 2
torun = argv[1]
output = 'testskim.root'
if len(argv)>2:
    debug_level = int(argv[2])
    if len(argv)>3:
        output = argv[3]

argv = []

import ROOT as root
from PandaCore.Tools.Load import *

Load('PandaAnalyzer')

skimmer = root.PandaAnalyzer(debug_level)


# skimmer.firstEvent=0
# skimmer.lastEvent=50
skimmer.isData=False
skimmer.SetFlag('puppi',False)
skimmer.SetFlag('fatjet',False)
skimmer.SetFlag('vbf',False)
skimmer.SetFlag('firstGen',False)
skimmer.SetFlag('applyEGCorr',False)
skimmer.SetFlag('applyJSON',False)
skimmer.SetFlag('pfCands',True)
skimmer.SetFlag('monohiggs',True)
#skimmer.SetFlag('monohiggs',True)
if skimmer.isData and False:
    with open(getenv('CMSSW_BASE')+'/src/PandaAnalysis/data/certs/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt') as jsonFile:
        payload = json.load(jsonFile)
        for run,lumis in payload.iteritems():
            for l in lumis:
                skimmer.AddGoodLumiRange(int(run),l[0],l[1])
skimmer.processType = root.PandaAnalyzer.kTT                                                                                                                                                 
#skimmer.SetPreselectionBit(root.PandaAnalyzer.kMonojet)
#system("pxrdcp %s input.root '!pfCandidates'"%(torun))
#fin = root.TFile.Open('input.root')
fin = root.TFile.Open(torun)

tree = fin.FindObjectAny("events")
hweights = fin.FindObjectAny("hSumW")
weights = fin.FindObjectAny('weights')
if not weights:
    weights = None
print tree, hweights, weights

skimmer.SetDataDir(getenv('CMSSW_BASE')+'/src/PandaAnalysis/data/')
skimmer.Init(tree,hweights,weights)
skimmer.SetOutputFile(output)

skimmer.Run()
skimmer.Terminate()
