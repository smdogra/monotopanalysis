#include "../interface/EventMonophoton.h"

panda::EventMonophoton::EventMonophoton() :
  EventBase()
{
  std::vector<Object*> myObjects{{&genReweight, &pfCandidates, &vertices, &superClusters, &electrons, &muons, &taus, &photons, &jets, &genJets, &genParticles, &genVertex, &partons, &t1Met, &rawMet, &caloMet, &metMuOnlyFix, &metNoFix, &metFilters}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&pfCandidates, &vertices, &superClusters, &electrons, &muons, &taus, &photons, &jets, &genJets, &genParticles, &partons}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());

  pfCandidates.data.vertexContainer_ = &vertices;
  electrons.data.superClusterContainer_ = &superClusters;
  photons.data.superClusterContainer_ = &superClusters;
  jets.data.constituentsContainer_ = &pfCandidates;
  genParticles.data.parentContainer_ = &genParticles;
  jets.data.matchedGenJetContainer_ = &genJets;
}

panda::EventMonophoton::EventMonophoton(EventMonophoton const& _src) :
  EventBase(_src),
  genReweight(_src.genReweight),
  pfCandidates(_src.pfCandidates),
  vertices(_src.vertices),
  superClusters(_src.superClusters),
  electrons(_src.electrons),
  muons(_src.muons),
  taus(_src.taus),
  photons(_src.photons),
  jets(_src.jets),
  genJets(_src.genJets),
  genParticles(_src.genParticles),
  genVertex(_src.genVertex),
  partons(_src.partons),
  t1Met(_src.t1Met),
  rawMet(_src.rawMet),
  caloMet(_src.caloMet),
  metMuOnlyFix(_src.metMuOnlyFix),
  metNoFix(_src.metNoFix),
  metFilters(_src.metFilters),
  npv(_src.npv),
  npvTrue(_src.npvTrue),
  rho(_src.rho),
  rhoCentralCalo(_src.rhoCentralCalo)
{
  std::vector<Object*> myObjects{{&genReweight, &pfCandidates, &vertices, &superClusters, &electrons, &muons, &taus, &photons, &jets, &genJets, &genParticles, &genVertex, &partons, &t1Met, &rawMet, &caloMet, &metMuOnlyFix, &metNoFix, &metFilters}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&pfCandidates, &vertices, &superClusters, &electrons, &muons, &taus, &photons, &jets, &genJets, &genParticles, &partons}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());


  pfCandidates.data.vertexContainer_ = &vertices;
  electrons.data.superClusterContainer_ = &superClusters;
  photons.data.superClusterContainer_ = &superClusters;
  jets.data.constituentsContainer_ = &pfCandidates;
  genParticles.data.parentContainer_ = &genParticles;
  jets.data.matchedGenJetContainer_ = &genJets;
  /* BEGIN CUSTOM EventMonophoton.cc.copy_ctor */
  /* END CUSTOM */
}

panda::EventMonophoton::~EventMonophoton()
{
  /* BEGIN CUSTOM EventMonophoton.cc.dtor */
  /* END CUSTOM */
}

panda::EventMonophoton&
panda::EventMonophoton::operator=(EventMonophoton const& _src)
{
  EventBase::operator=(_src);

  /* BEGIN CUSTOM EventMonophoton.cc.operator= */
  /* END CUSTOM */

  npv = _src.npv;
  npvTrue = _src.npvTrue;
  rho = _src.rho;
  rhoCentralCalo = _src.rhoCentralCalo;

  genReweight = _src.genReweight;
  pfCandidates = _src.pfCandidates;
  vertices = _src.vertices;
  superClusters = _src.superClusters;
  electrons = _src.electrons;
  muons = _src.muons;
  taus = _src.taus;
  photons = _src.photons;
  jets = _src.jets;
  genJets = _src.genJets;
  genParticles = _src.genParticles;
  genVertex = _src.genVertex;
  partons = _src.partons;
  t1Met = _src.t1Met;
  rawMet = _src.rawMet;
  caloMet = _src.caloMet;
  metMuOnlyFix = _src.metMuOnlyFix;
  metNoFix = _src.metNoFix;
  metFilters = _src.metFilters;

  pfCandidates.data.vertexContainer_ = &vertices;
  electrons.data.superClusterContainer_ = &superClusters;
  photons.data.superClusterContainer_ = &superClusters;
  jets.data.constituentsContainer_ = &pfCandidates;
  genParticles.data.parentContainer_ = &genParticles;
  jets.data.matchedGenJetContainer_ = &genJets;

  return *this;
}

void
panda::EventMonophoton::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM EventMonophoton.cc.print */
  if (_level >= 3) {
    EventMonophoton::dump(_out);
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
panda::EventMonophoton::dump(std::ostream& _out/* = std::cout*/) const
{
  EventBase::dump(_out);

  _out << "npv = " << npv << std::endl;
  _out << "npvTrue = " << npvTrue << std::endl;
  _out << "rho = " << rho << std::endl;
  _out << "rhoCentralCalo = " << rhoCentralCalo << std::endl;

  genReweight.dump(_out);
  pfCandidates.dump(_out);
  vertices.dump(_out);
  superClusters.dump(_out);
  electrons.dump(_out);
  muons.dump(_out);
  taus.dump(_out);
  photons.dump(_out);
  jets.dump(_out);
  genJets.dump(_out);
  genParticles.dump(_out);
  genVertex.dump(_out);
  partons.dump(_out);
  t1Met.dump(_out);
  rawMet.dump(_out);
  caloMet.dump(_out);
  metMuOnlyFix.dump(_out);
  metNoFix.dump(_out);
  metFilters.dump(_out);

}
/*static*/
panda::utils::BranchList
panda::EventMonophoton::getListOfBranches(Bool_t _direct/* = kFALSE*/)
{
  utils::BranchList blist;
  blist += EventBase::getListOfBranches(_direct);

  blist += {"npv", "npvTrue", "rho", "rhoCentralCalo"};
  if (!_direct) {
    blist += GenReweight::getListOfBranches().fullNames("genReweight");
    blist += PFCand::getListOfBranches().fullNames("pfCandidates");
    blist += RecoVertex::getListOfBranches().fullNames("vertices");
    blist += SuperCluster::getListOfBranches().fullNames("superClusters");
    blist += Electron::getListOfBranches().fullNames("electrons");
    blist += Muon::getListOfBranches().fullNames("muons");
    blist += Tau::getListOfBranches().fullNames("taus");
    blist += XPhoton::getListOfBranches().fullNames("photons");
    blist += Jet::getListOfBranches().fullNames("jets");
    blist += GenJet::getListOfBranches().fullNames("genJets");
    blist += UnpackedGenParticle::getListOfBranches().fullNames("genParticles");
    blist += Vertex::getListOfBranches().fullNames("genVertex");
    blist += Parton::getListOfBranches().fullNames("partons");
    blist += RecoMet::getListOfBranches().fullNames("t1Met");
    blist += Met::getListOfBranches().fullNames("rawMet");
    blist += Met::getListOfBranches().fullNames("caloMet");
    blist += RecoMet::getListOfBranches().fullNames("metMuOnlyFix");
    blist += RecoMet::getListOfBranches().fullNames("metNoFix");
    blist += MetFilters::getListOfBranches().fullNames("metFilters");
  }
  return blist;
}

/*protected*/
void
panda::EventMonophoton::doSetStatus_(TTree& _tree, utils::BranchList const& _branches)
{
  EventBase::doSetStatus_(_tree, _branches);
  utils::setStatus(_tree, "", "npv", _branches);
  utils::setStatus(_tree, "", "npvTrue", _branches);
  utils::setStatus(_tree, "", "rho", _branches);
  utils::setStatus(_tree, "", "rhoCentralCalo", _branches);
}

/*protected*/
panda::utils::BranchList
panda::EventMonophoton::doGetStatus_(TTree& _tree) const
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
panda::EventMonophoton::doGetBranchNames_() const
{
  return getListOfBranches(true);
}

/*protected*/
void
panda::EventMonophoton::doSetAddress_(TTree& _tree, utils::BranchList const& _branches, Bool_t _setStatus)
{
  EventBase::doSetAddress_(_tree, _branches, _setStatus);

  utils::setAddress(_tree, "", "npv", &npv, _branches, _setStatus);
  utils::setAddress(_tree, "", "npvTrue", &npvTrue, _branches, _setStatus);
  utils::setAddress(_tree, "", "rho", &rho, _branches, _setStatus);
  utils::setAddress(_tree, "", "rhoCentralCalo", &rhoCentralCalo, _branches, _setStatus);
}

/*protected*/
void
panda::EventMonophoton::doBook_(TTree& _tree, utils::BranchList const& _branches)
{
  EventBase::doBook_(_tree, _branches);

  utils::book(_tree, "", "npv", "", 's', &npv, _branches);
  utils::book(_tree, "", "npvTrue", "", 's', &npvTrue, _branches);
  utils::book(_tree, "", "rho", "", 'F', &rho, _branches);
  utils::book(_tree, "", "rhoCentralCalo", "", 'F', &rhoCentralCalo, _branches);
}

/*protected*/
void
panda::EventMonophoton::doGetEntry_(TTree& _tree)
{
  EventBase::doGetEntry_(_tree);

  /* BEGIN CUSTOM EventMonophoton.cc.doGetEntry_ */
  if (!pfCandidates.empty() && !vertices.empty()) {
    unsigned iVtx(0);

    for (unsigned iC(0); iC != pfCandidates.size(); ++iC) {
      auto& cand(pfCandidates[iC]);

      if (iC == vertices[iVtx].pfRangeMax) {
        ++iVtx;
        if (iVtx == vertices.size())
          break;
      }

      cand.vertex.idx() = iVtx;
    }
  }
  /* END CUSTOM */
}

void
panda::EventMonophoton::doInit_()
{
  EventBase::doInit_();

  npv = 0;
  npvTrue = 0;
  rho = 0.;
  rhoCentralCalo = 0.;
  /* BEGIN CUSTOM EventMonophoton.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::EventMonophoton::doUnlink_(TTree& _tree)
{
  EventBase::doUnlink_(_tree);

  /* BEGIN CUSTOM EventMonophoton.cc.doUnlink_ */
  /* END CUSTOM */
}


/* BEGIN CUSTOM EventMonophoton.cc.global */
#include <map>

panda::EventMonophoton&
panda::EventMonophoton::copy(Event const& _src)
{
  runNumber = _src.runNumber;
  lumiNumber = _src.lumiNumber;
  eventNumber = _src.eventNumber;
  isData = _src.isData;
  weight = _src.weight;

  triggers = _src.triggers;

  npv = _src.npv;
  npvTrue = _src.npvTrue;
  rho = _src.rho;
  rhoCentralCalo = _src.rhoCentralCalo;

  genReweight = _src.genReweight;
  pfCandidates = _src.pfCandidates;
  vertices = _src.vertices;
  superClusters = _src.superClusters;
  electrons = _src.electrons;
  muons = _src.muons;
  taus = _src.taus;
  photons.resize(_src.photons.size());
  for (unsigned iP(0); iP != photons.size(); ++iP)
    static_cast<Photon&>(photons[iP]) = _src.photons[iP];
  jets = _src.chsAK4Jets;
  genJets = _src.ak4GenJets;
  partons = _src.partons;
  t1Met = _src.pfMet;
  rawMet = _src.rawMet;
  caloMet = _src.caloMet;
  metMuOnlyFix = _src.metMuOnlyFix;
  metNoFix = _src.metNoFix;
  metFilters = _src.metFilters;
  genVertex = _src.genVertex;
  copyGenParticles(_src.genParticles);

  electrons.data.superClusterContainer_ = &superClusters;
  photons.data.superClusterContainer_ = &superClusters;
  genParticles.data.parentContainer_ = &genParticles;
  jets.data.matchedGenJetContainer_ = &genJets;
  jets.data.constituentsContainer_ = 0;

  return *this;
}

void
panda::EventMonophoton::copyGenParticles(GenParticleCollection const& _src)
{
  genParticles.clear();

  // only save leptons, prompt photons, and their ancestors
  std::map<short, unsigned> parentMap;

  int iGen(-1);
  for (auto& part : _src) {
    ++iGen;
    if (!part.finalState)
      continue;

    switch (std::abs(part.pdgid)) {
    case 22:
      if (part.testFlag(GenParticle::kIsPrompt) ||
          (part.parent.isValid() && part.parent->pdgid == 22 && part.parent->testFlag(GenParticle::kIsPrompt)))
        break;
      continue;
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
      break;
    default:
      continue;
    }

    auto& out(genParticles.create_back());
    out = part;

    // save the first mother
    short parentIdx(part.parent.idx());
    while (parentIdx != -1) {
      auto& parent(_src[parentIdx]);

      if (parent.pdgid == part.pdgid) {
        // skip if it's the same particle
        parentIdx = parent.parent.idx();
        continue;
      }
      
      auto itr(parentMap.find(parentIdx));
      if (itr == parentMap.end()) {
        // parent particle not yet in output genParticles
        parentMap[parentIdx] = genParticles.size();

        auto& srcParent(_src[parentIdx]);
        auto& outParent(genParticles.create_back());
        outParent = srcParent;
        out.parent.setRef(&outParent);
      }
      else {
        // already recorded particle; just set the parent ref
        out.parent.setRef(&genParticles[itr->second]);
      }

      break;
    }
  }
}

/* END CUSTOM */
