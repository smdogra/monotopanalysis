#include "../interface/Event.h"

panda::Event::Event() :
  EventBase()
{
  std::vector<Object*> myObjects{{&genReweight, &pfCandidates, &vertices, &tracks, &superClusters, &electrons, &muons, &taus, &photons, &chsAK4Jets, &puppiAK4Jets, &chsAK8Jets, &chsAK8Subjets, &chsCA15Jets, &chsCA15Subjets, &puppiAK8Jets, &puppiAK8Subjets, &puppiCA15Jets, &puppiCA15Subjets, &ak4GenJets, &ak8GenJets, &ca15GenJets, &genParticles, &genVertex, &partons, &pfMet, &puppiMet, &rawMet, &caloMet, &noMuMet, &noHFMet, &trkMet, &neutralMet, &photonMet, &hfMet, &genMet, &metMuOnlyFix, &metNoFix, &metFilters, &recoil}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&pfCandidates, &vertices, &tracks, &superClusters, &electrons, &muons, &taus, &photons, &chsAK4Jets, &puppiAK4Jets, &chsAK8Jets, &chsAK8Subjets, &chsCA15Jets, &chsCA15Subjets, &puppiAK8Jets, &puppiAK8Subjets, &puppiCA15Jets, &puppiCA15Subjets, &ak4GenJets, &ak8GenJets, &ca15GenJets, &genParticles, &partons}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());

  pfCandidates.data.vertexContainer_ = &vertices;
  pfCandidates.data.trackContainer_ = &tracks;
  electrons.data.superClusterContainer_ = &superClusters;
  electrons.data.matchedPFContainer_ = &pfCandidates;
  electrons.data.matchedGenContainer_ = &genParticles;
  electrons.data.vertexContainer_ = &vertices;
  muons.data.matchedPFContainer_ = &pfCandidates;
  muons.data.matchedGenContainer_ = &genParticles;
  muons.data.vertexContainer_ = &vertices;
  taus.data.matchedGenContainer_ = &genParticles;
  taus.data.vertexContainer_ = &vertices;
  photons.data.superClusterContainer_ = &superClusters;
  photons.data.matchedPFContainer_ = &pfCandidates;
  photons.data.matchedGenContainer_ = &genParticles;
  genParticles.data.parentContainer_ = &genParticles;
  chsAK4Jets.data.constituentsContainer_ = &pfCandidates;
  chsAK4Jets.data.matchedGenJetContainer_ = &ak4GenJets;
  puppiAK4Jets.data.constituentsContainer_ = &pfCandidates;
  puppiAK4Jets.data.matchedGenJetContainer_ = &ak4GenJets;
  chsAK8Jets.data.constituentsContainer_ = &pfCandidates;
  chsAK8Jets.data.subjetsContainer_ = &chsAK8Subjets;
  chsAK8Jets.data.matchedGenJetContainer_ = &ak8GenJets;
  chsCA15Jets.data.constituentsContainer_ = &pfCandidates;
  chsCA15Jets.data.subjetsContainer_ = &chsCA15Subjets;
  chsCA15Jets.data.matchedGenJetContainer_ = &ca15GenJets;
  puppiAK8Jets.data.constituentsContainer_ = &pfCandidates;
  puppiAK8Jets.data.subjetsContainer_ = &puppiAK8Subjets;
  puppiAK8Jets.data.matchedGenJetContainer_ = &ak8GenJets;
  puppiCA15Jets.data.constituentsContainer_ = &pfCandidates;
  puppiCA15Jets.data.subjetsContainer_ = &puppiCA15Subjets;
  puppiCA15Jets.data.matchedGenJetContainer_ = &ca15GenJets;
}

panda::Event::Event(Event const& _src) :
  EventBase(_src),
  genReweight(_src.genReweight),
  pfCandidates(_src.pfCandidates),
  vertices(_src.vertices),
  tracks(_src.tracks),
  superClusters(_src.superClusters),
  electrons(_src.electrons),
  muons(_src.muons),
  taus(_src.taus),
  photons(_src.photons),
  chsAK4Jets(_src.chsAK4Jets),
  puppiAK4Jets(_src.puppiAK4Jets),
  chsAK8Jets(_src.chsAK8Jets),
  chsAK8Subjets(_src.chsAK8Subjets),
  chsCA15Jets(_src.chsCA15Jets),
  chsCA15Subjets(_src.chsCA15Subjets),
  puppiAK8Jets(_src.puppiAK8Jets),
  puppiAK8Subjets(_src.puppiAK8Subjets),
  puppiCA15Jets(_src.puppiCA15Jets),
  puppiCA15Subjets(_src.puppiCA15Subjets),
  ak4GenJets(_src.ak4GenJets),
  ak8GenJets(_src.ak8GenJets),
  ca15GenJets(_src.ca15GenJets),
  genParticles(_src.genParticles),
  genVertex(_src.genVertex),
  partons(_src.partons),
  pfMet(_src.pfMet),
  puppiMet(_src.puppiMet),
  rawMet(_src.rawMet),
  caloMet(_src.caloMet),
  noMuMet(_src.noMuMet),
  noHFMet(_src.noHFMet),
  trkMet(_src.trkMet),
  neutralMet(_src.neutralMet),
  photonMet(_src.photonMet),
  hfMet(_src.hfMet),
  genMet(_src.genMet),
  metMuOnlyFix(_src.metMuOnlyFix),
  metNoFix(_src.metNoFix),
  metFilters(_src.metFilters),
  recoil(_src.recoil),
  npv(_src.npv),
  npvTrue(_src.npvTrue),
  rho(_src.rho),
  rhoCentralCalo(_src.rhoCentralCalo)
{
  std::vector<Object*> myObjects{{&genReweight, &pfCandidates, &vertices, &tracks, &superClusters, &electrons, &muons, &taus, &photons, &chsAK4Jets, &puppiAK4Jets, &chsAK8Jets, &chsAK8Subjets, &chsCA15Jets, &chsCA15Subjets, &puppiAK8Jets, &puppiAK8Subjets, &puppiCA15Jets, &puppiCA15Subjets, &ak4GenJets, &ak8GenJets, &ca15GenJets, &genParticles, &genVertex, &partons, &pfMet, &puppiMet, &rawMet, &caloMet, &noMuMet, &noHFMet, &trkMet, &neutralMet, &photonMet, &hfMet, &genMet, &metMuOnlyFix, &metNoFix, &metFilters, &recoil}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&pfCandidates, &vertices, &tracks, &superClusters, &electrons, &muons, &taus, &photons, &chsAK4Jets, &puppiAK4Jets, &chsAK8Jets, &chsAK8Subjets, &chsCA15Jets, &chsCA15Subjets, &puppiAK8Jets, &puppiAK8Subjets, &puppiCA15Jets, &puppiCA15Subjets, &ak4GenJets, &ak8GenJets, &ca15GenJets, &genParticles, &partons}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());


  pfCandidates.data.vertexContainer_ = &vertices;
  pfCandidates.data.trackContainer_ = &tracks;
  electrons.data.superClusterContainer_ = &superClusters;
  electrons.data.matchedPFContainer_ = &pfCandidates;
  electrons.data.matchedGenContainer_ = &genParticles;
  electrons.data.vertexContainer_ = &vertices;
  muons.data.matchedPFContainer_ = &pfCandidates;
  muons.data.matchedGenContainer_ = &genParticles;
  muons.data.vertexContainer_ = &vertices;
  taus.data.matchedGenContainer_ = &genParticles;
  taus.data.vertexContainer_ = &vertices;
  photons.data.superClusterContainer_ = &superClusters;
  photons.data.matchedPFContainer_ = &pfCandidates;
  photons.data.matchedGenContainer_ = &genParticles;
  genParticles.data.parentContainer_ = &genParticles;
  chsAK4Jets.data.constituentsContainer_ = &pfCandidates;
  chsAK4Jets.data.matchedGenJetContainer_ = &ak4GenJets;
  puppiAK4Jets.data.constituentsContainer_ = &pfCandidates;
  puppiAK4Jets.data.matchedGenJetContainer_ = &ak4GenJets;
  chsAK8Jets.data.constituentsContainer_ = &pfCandidates;
  chsAK8Jets.data.subjetsContainer_ = &chsAK8Subjets;
  chsAK8Jets.data.matchedGenJetContainer_ = &ak8GenJets;
  chsCA15Jets.data.constituentsContainer_ = &pfCandidates;
  chsCA15Jets.data.subjetsContainer_ = &chsCA15Subjets;
  chsCA15Jets.data.matchedGenJetContainer_ = &ca15GenJets;
  puppiAK8Jets.data.constituentsContainer_ = &pfCandidates;
  puppiAK8Jets.data.subjetsContainer_ = &puppiAK8Subjets;
  puppiAK8Jets.data.matchedGenJetContainer_ = &ak8GenJets;
  puppiCA15Jets.data.constituentsContainer_ = &pfCandidates;
  puppiCA15Jets.data.subjetsContainer_ = &puppiCA15Subjets;
  puppiCA15Jets.data.matchedGenJetContainer_ = &ca15GenJets;
  /* BEGIN CUSTOM Event.cc.copy_ctor */
  /* END CUSTOM */
}

panda::Event::~Event()
{
  /* BEGIN CUSTOM Event.cc.dtor */
  /* END CUSTOM */
}

panda::Event&
panda::Event::operator=(Event const& _src)
{
  EventBase::operator=(_src);

  /* BEGIN CUSTOM Event.cc.operator= */
  /* END CUSTOM */

  npv = _src.npv;
  npvTrue = _src.npvTrue;
  rho = _src.rho;
  rhoCentralCalo = _src.rhoCentralCalo;

  genReweight = _src.genReweight;
  pfCandidates = _src.pfCandidates;
  vertices = _src.vertices;
  tracks = _src.tracks;
  superClusters = _src.superClusters;
  electrons = _src.electrons;
  muons = _src.muons;
  taus = _src.taus;
  photons = _src.photons;
  chsAK4Jets = _src.chsAK4Jets;
  puppiAK4Jets = _src.puppiAK4Jets;
  chsAK8Jets = _src.chsAK8Jets;
  chsAK8Subjets = _src.chsAK8Subjets;
  chsCA15Jets = _src.chsCA15Jets;
  chsCA15Subjets = _src.chsCA15Subjets;
  puppiAK8Jets = _src.puppiAK8Jets;
  puppiAK8Subjets = _src.puppiAK8Subjets;
  puppiCA15Jets = _src.puppiCA15Jets;
  puppiCA15Subjets = _src.puppiCA15Subjets;
  ak4GenJets = _src.ak4GenJets;
  ak8GenJets = _src.ak8GenJets;
  ca15GenJets = _src.ca15GenJets;
  genParticles = _src.genParticles;
  genVertex = _src.genVertex;
  partons = _src.partons;
  pfMet = _src.pfMet;
  puppiMet = _src.puppiMet;
  rawMet = _src.rawMet;
  caloMet = _src.caloMet;
  noMuMet = _src.noMuMet;
  noHFMet = _src.noHFMet;
  trkMet = _src.trkMet;
  neutralMet = _src.neutralMet;
  photonMet = _src.photonMet;
  hfMet = _src.hfMet;
  genMet = _src.genMet;
  metMuOnlyFix = _src.metMuOnlyFix;
  metNoFix = _src.metNoFix;
  metFilters = _src.metFilters;
  recoil = _src.recoil;

  pfCandidates.data.vertexContainer_ = &vertices;
  pfCandidates.data.trackContainer_ = &tracks;
  electrons.data.superClusterContainer_ = &superClusters;
  electrons.data.matchedPFContainer_ = &pfCandidates;
  electrons.data.matchedGenContainer_ = &genParticles;
  electrons.data.vertexContainer_ = &vertices;
  muons.data.matchedPFContainer_ = &pfCandidates;
  muons.data.matchedGenContainer_ = &genParticles;
  muons.data.vertexContainer_ = &vertices;
  taus.data.matchedGenContainer_ = &genParticles;
  taus.data.vertexContainer_ = &vertices;
  photons.data.superClusterContainer_ = &superClusters;
  photons.data.matchedPFContainer_ = &pfCandidates;
  photons.data.matchedGenContainer_ = &genParticles;
  genParticles.data.parentContainer_ = &genParticles;
  chsAK4Jets.data.constituentsContainer_ = &pfCandidates;
  chsAK4Jets.data.matchedGenJetContainer_ = &ak4GenJets;
  puppiAK4Jets.data.constituentsContainer_ = &pfCandidates;
  puppiAK4Jets.data.matchedGenJetContainer_ = &ak4GenJets;
  chsAK8Jets.data.constituentsContainer_ = &pfCandidates;
  chsAK8Jets.data.subjetsContainer_ = &chsAK8Subjets;
  chsAK8Jets.data.matchedGenJetContainer_ = &ak8GenJets;
  chsCA15Jets.data.constituentsContainer_ = &pfCandidates;
  chsCA15Jets.data.subjetsContainer_ = &chsCA15Subjets;
  chsCA15Jets.data.matchedGenJetContainer_ = &ca15GenJets;
  puppiAK8Jets.data.constituentsContainer_ = &pfCandidates;
  puppiAK8Jets.data.subjetsContainer_ = &puppiAK8Subjets;
  puppiAK8Jets.data.matchedGenJetContainer_ = &ak8GenJets;
  puppiCA15Jets.data.constituentsContainer_ = &pfCandidates;
  puppiCA15Jets.data.subjetsContainer_ = &puppiCA15Subjets;
  puppiCA15Jets.data.matchedGenJetContainer_ = &ca15GenJets;

  return *this;
}

void
panda::Event::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM Event.cc.print */
  if (_level >= 3) {
    Event::dump(_out);
  }
  else if (_level == 2) {
    // debug level
    EventBase::print(_out, _level);
    
    _out << "npv = " << npv << std::endl;
    _out << "npvTrue = " << npvTrue << std::endl;
    _out << "rho = " << rho << std::endl;
    _out << "rhoCentralCalo = " << rhoCentralCalo << std::endl;
  }
  else 
    return;
  /* END CUSTOM */
}

void
panda::Event::dump(std::ostream& _out/* = std::cout*/) const
{
  EventBase::dump(_out);

  _out << "npv = " << npv << std::endl;
  _out << "npvTrue = " << npvTrue << std::endl;
  _out << "rho = " << rho << std::endl;
  _out << "rhoCentralCalo = " << rhoCentralCalo << std::endl;

  genReweight.dump(_out);
  pfCandidates.dump(_out);
  vertices.dump(_out);
  tracks.dump(_out);
  superClusters.dump(_out);
  electrons.dump(_out);
  muons.dump(_out);
  taus.dump(_out);
  photons.dump(_out);
  chsAK4Jets.dump(_out);
  puppiAK4Jets.dump(_out);
  chsAK8Jets.dump(_out);
  chsAK8Subjets.dump(_out);
  chsCA15Jets.dump(_out);
  chsCA15Subjets.dump(_out);
  puppiAK8Jets.dump(_out);
  puppiAK8Subjets.dump(_out);
  puppiCA15Jets.dump(_out);
  puppiCA15Subjets.dump(_out);
  ak4GenJets.dump(_out);
  ak8GenJets.dump(_out);
  ca15GenJets.dump(_out);
  genParticles.dump(_out);
  genVertex.dump(_out);
  partons.dump(_out);
  pfMet.dump(_out);
  puppiMet.dump(_out);
  rawMet.dump(_out);
  caloMet.dump(_out);
  noMuMet.dump(_out);
  noHFMet.dump(_out);
  trkMet.dump(_out);
  neutralMet.dump(_out);
  photonMet.dump(_out);
  hfMet.dump(_out);
  genMet.dump(_out);
  metMuOnlyFix.dump(_out);
  metNoFix.dump(_out);
  metFilters.dump(_out);
  recoil.dump(_out);

}
/*static*/
panda::utils::BranchList
panda::Event::getListOfBranches(Bool_t _direct/* = kFALSE*/)
{
  utils::BranchList blist;
  blist += EventBase::getListOfBranches(_direct);

  blist += {"npv", "npvTrue", "rho", "rhoCentralCalo"};
  if (!_direct) {
    blist += GenReweight::getListOfBranches().fullNames("genReweight");
    blist += PFCand::getListOfBranches().fullNames("pfCandidates");
    blist += RecoVertex::getListOfBranches().fullNames("vertices");
    blist += PackedTrack::getListOfBranches().fullNames("tracks");
    blist += SuperCluster::getListOfBranches().fullNames("superClusters");
    blist += Electron::getListOfBranches().fullNames("electrons");
    blist += Muon::getListOfBranches().fullNames("muons");
    blist += Tau::getListOfBranches().fullNames("taus");
    blist += Photon::getListOfBranches().fullNames("photons");
    blist += Jet::getListOfBranches().fullNames("chsAK4Jets");
    blist += Jet::getListOfBranches().fullNames("puppiAK4Jets");
    blist += FatJet::getListOfBranches().fullNames("chsAK8Jets");
    blist += MicroJet::getListOfBranches().fullNames("chsAK8Subjets");
    blist += FatJet::getListOfBranches().fullNames("chsCA15Jets");
    blist += MicroJet::getListOfBranches().fullNames("chsCA15Subjets");
    blist += FatJet::getListOfBranches().fullNames("puppiAK8Jets");
    blist += MicroJet::getListOfBranches().fullNames("puppiAK8Subjets");
    blist += FatJet::getListOfBranches().fullNames("puppiCA15Jets");
    blist += MicroJet::getListOfBranches().fullNames("puppiCA15Subjets");
    blist += GenJet::getListOfBranches().fullNames("ak4GenJets");
    blist += GenJet::getListOfBranches().fullNames("ak8GenJets");
    blist += GenJet::getListOfBranches().fullNames("ca15GenJets");
    blist += GenParticle::getListOfBranches().fullNames("genParticles");
    blist += Vertex::getListOfBranches().fullNames("genVertex");
    blist += Parton::getListOfBranches().fullNames("partons");
    blist += RecoMet::getListOfBranches().fullNames("pfMet");
    blist += RecoMet::getListOfBranches().fullNames("puppiMet");
    blist += Met::getListOfBranches().fullNames("rawMet");
    blist += Met::getListOfBranches().fullNames("caloMet");
    blist += Met::getListOfBranches().fullNames("noMuMet");
    blist += Met::getListOfBranches().fullNames("noHFMet");
    blist += Met::getListOfBranches().fullNames("trkMet");
    blist += Met::getListOfBranches().fullNames("neutralMet");
    blist += Met::getListOfBranches().fullNames("photonMet");
    blist += Met::getListOfBranches().fullNames("hfMet");
    blist += Met::getListOfBranches().fullNames("genMet");
    blist += RecoMet::getListOfBranches().fullNames("metMuOnlyFix");
    blist += RecoMet::getListOfBranches().fullNames("metNoFix");
    blist += MetFilters::getListOfBranches().fullNames("metFilters");
    blist += Recoil::getListOfBranches().fullNames("recoil");
  }
  return blist;
}

/*protected*/
void
panda::Event::doSetStatus_(TTree& _tree, utils::BranchList const& _branches)
{
  EventBase::doSetStatus_(_tree, _branches);
  utils::setStatus(_tree, "", "npv", _branches);
  utils::setStatus(_tree, "", "npvTrue", _branches);
  utils::setStatus(_tree, "", "rho", _branches);
  utils::setStatus(_tree, "", "rhoCentralCalo", _branches);
}

/*protected*/
panda::utils::BranchList
panda::Event::doGetStatus_(TTree& _tree) const
{
  utils::BranchList blist;
  blist += EventBase::doGetStatus_(_tree);

  blist.push_back(utils::getStatus(_tree, "", "npv"));
  blist.push_back(utils::getStatus(_tree, "", "npvTrue"));
  blist.push_back(utils::getStatus(_tree, "", "rho"));
  blist.push_back(utils::getStatus(_tree, "", "rhoCentralCalo"));
  return blist;
}

/*protected*/
panda::utils::BranchList
panda::Event::doGetBranchNames_() const
{
  return getListOfBranches(true);
}

/*protected*/
void
panda::Event::doSetAddress_(TTree& _tree, utils::BranchList const& _branches, Bool_t _setStatus)
{
  EventBase::doSetAddress_(_tree, _branches, _setStatus);

  utils::setAddress(_tree, "", "npv", &npv, _branches, _setStatus);
  utils::setAddress(_tree, "", "npvTrue", &npvTrue, _branches, _setStatus);
  utils::setAddress(_tree, "", "rho", &rho, _branches, _setStatus);
  utils::setAddress(_tree, "", "rhoCentralCalo", &rhoCentralCalo, _branches, _setStatus);
}

/*protected*/
void
panda::Event::doBook_(TTree& _tree, utils::BranchList const& _branches)
{
  EventBase::doBook_(_tree, _branches);

  utils::book(_tree, "", "npv", "", 's', &npv, _branches);
  utils::book(_tree, "", "npvTrue", "", 's', &npvTrue, _branches);
  utils::book(_tree, "", "rho", "", 'F', &rho, _branches);
  utils::book(_tree, "", "rhoCentralCalo", "", 'F', &rhoCentralCalo, _branches);
}

/*protected*/
void
panda::Event::doGetEntry_(TTree& _tree)
{
  EventBase::doGetEntry_(_tree);

  /* BEGIN CUSTOM Event.cc.doGetEntry_ */
  bool hasVertices(!vertices.empty());
  bool hasTracks(!tracks.empty());

  if (!pfCandidates.empty() && (hasVertices || hasTracks)) {
    unsigned iVtx(0);
    unsigned iTrk(0);

    for (unsigned iC(0); iC != pfCandidates.size(); ++iC) {
      auto& cand(pfCandidates[iC]);

      if (hasVertices) {
        if (iC == vertices[iVtx].pfRangeMax) {
          ++iVtx;
          if (iVtx == vertices.size())
            break;
        }

        cand.vertex.idx() = iVtx;
      }

      if (hasTracks) {
        switch (cand.ptype) {
        case PFCand::hp:
        case PFCand::hm:
        case PFCand::ep:
        case PFCand::em:
        case PFCand::mup:
        case PFCand::mum:
          cand.track.idx() = iTrk++;
          break;
        default:
          break;
        }
      }
    }
  }
  /* END CUSTOM */
}

void
panda::Event::doInit_()
{
  EventBase::doInit_();

  npv = 0;
  npvTrue = 0;
  rho = 0.;
  rhoCentralCalo = 0.;
  /* BEGIN CUSTOM Event.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::Event::doUnlink_(TTree& _tree)
{
  EventBase::doUnlink_(_tree);

  /* BEGIN CUSTOM Event.cc.doUnlink_ */
  /* END CUSTOM */
}


/* BEGIN CUSTOM Event.cc.global */
/* END CUSTOM */
