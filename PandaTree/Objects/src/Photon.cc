#include "../interface/Photon.h"

TString panda::Photon::TriggerObjectName[] = {
  "fSEG34IorSEG40",
  "fSEG40IorSJet200",
  "fSEG34IorSEG40IorSJet200",
  "fSEG24",
  "fSEG30",
  "fSEG40",
  "fPh120",
  "fPh135",
  "fPh165HE10",
  "fPh175",
  "fPh22EBR9Iso",
  "fPh36EBR9Iso",
  "fPh50EBR9Iso",
  "fPh75EBR9Iso",
  "fPh90EBR9Iso",
  "fPh120EBR9Iso"
};

/*static*/
panda::utils::BranchList
panda::Photon::getListOfBranches()
{
  utils::BranchList blist;
  blist += ParticleP::getListOfBranches();
  blist += {"pfPt", "chIso", "chIsoMax", "nhIso", "phIso", "sieie", "sipip", "hOverE", "genIso", "mipEnergy", "emax", "e2nd", "eleft", "eright", "etop", "ebottom", "r9", "etaWidth", "phiWidth", "time", "timeSpan", "regPt", "smearedPt", "originalPt", "loose", "medium", "tight", "highpt", "pixelVeto", "csafeVeto", "triggerMatch", "superCluster_", "matchedPF_", "matchedGen_"};
  return blist;
}

void
panda::Photon::datastore::allocate(UInt_t _nmax)
{
  ParticleP::datastore::allocate(_nmax);

  pfPt = new Float_t[nmax_];
  chIso = new Float_t[nmax_];
  chIsoMax = new Float_t[nmax_];
  nhIso = new Float_t[nmax_];
  phIso = new Float_t[nmax_];
  sieie = new Float_t[nmax_];
  sipip = new Float_t[nmax_];
  hOverE = new Float_t[nmax_];
  genIso = new Float_t[nmax_];
  mipEnergy = new Float_t[nmax_];
  emax = new Float_t[nmax_];
  e2nd = new Float_t[nmax_];
  eleft = new Float_t[nmax_];
  eright = new Float_t[nmax_];
  etop = new Float_t[nmax_];
  ebottom = new Float_t[nmax_];
  r9 = new Float_t[nmax_];
  etaWidth = new Float_t[nmax_];
  phiWidth = new Float_t[nmax_];
  time = new Float_t[nmax_];
  timeSpan = new Float_t[nmax_];
  regPt = new Float_t[nmax_];
  smearedPt = new Float_t[nmax_];
  originalPt = new Float_t[nmax_];
  loose = new Bool_t[nmax_];
  medium = new Bool_t[nmax_];
  tight = new Bool_t[nmax_];
  highpt = new Bool_t[nmax_];
  pixelVeto = new Bool_t[nmax_];
  csafeVeto = new Bool_t[nmax_];
  triggerMatch = new Bool_t[nmax_][nTriggerObjects];
  superCluster_ = new Short_t[nmax_];
  matchedPF_ = new Short_t[nmax_];
  matchedGen_ = new Short_t[nmax_];
}

void
panda::Photon::datastore::deallocate()
{
  ParticleP::datastore::deallocate();

  delete [] pfPt;
  pfPt = 0;
  delete [] chIso;
  chIso = 0;
  delete [] chIsoMax;
  chIsoMax = 0;
  delete [] nhIso;
  nhIso = 0;
  delete [] phIso;
  phIso = 0;
  delete [] sieie;
  sieie = 0;
  delete [] sipip;
  sipip = 0;
  delete [] hOverE;
  hOverE = 0;
  delete [] genIso;
  genIso = 0;
  delete [] mipEnergy;
  mipEnergy = 0;
  delete [] emax;
  emax = 0;
  delete [] e2nd;
  e2nd = 0;
  delete [] eleft;
  eleft = 0;
  delete [] eright;
  eright = 0;
  delete [] etop;
  etop = 0;
  delete [] ebottom;
  ebottom = 0;
  delete [] r9;
  r9 = 0;
  delete [] etaWidth;
  etaWidth = 0;
  delete [] phiWidth;
  phiWidth = 0;
  delete [] time;
  time = 0;
  delete [] timeSpan;
  timeSpan = 0;
  delete [] regPt;
  regPt = 0;
  delete [] smearedPt;
  smearedPt = 0;
  delete [] originalPt;
  originalPt = 0;
  delete [] loose;
  loose = 0;
  delete [] medium;
  medium = 0;
  delete [] tight;
  tight = 0;
  delete [] highpt;
  highpt = 0;
  delete [] pixelVeto;
  pixelVeto = 0;
  delete [] csafeVeto;
  csafeVeto = 0;
  delete [] triggerMatch;
  triggerMatch = 0;
  delete [] superCluster_;
  superCluster_ = 0;
  delete [] matchedPF_;
  matchedPF_ = 0;
  delete [] matchedGen_;
  matchedGen_ = 0;
}

void
panda::Photon::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  ParticleP::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "pfPt", _branches);
  utils::setStatus(_tree, _name, "chIso", _branches);
  utils::setStatus(_tree, _name, "chIsoMax", _branches);
  utils::setStatus(_tree, _name, "nhIso", _branches);
  utils::setStatus(_tree, _name, "phIso", _branches);
  utils::setStatus(_tree, _name, "sieie", _branches);
  utils::setStatus(_tree, _name, "sipip", _branches);
  utils::setStatus(_tree, _name, "hOverE", _branches);
  utils::setStatus(_tree, _name, "genIso", _branches);
  utils::setStatus(_tree, _name, "mipEnergy", _branches);
  utils::setStatus(_tree, _name, "emax", _branches);
  utils::setStatus(_tree, _name, "e2nd", _branches);
  utils::setStatus(_tree, _name, "eleft", _branches);
  utils::setStatus(_tree, _name, "eright", _branches);
  utils::setStatus(_tree, _name, "etop", _branches);
  utils::setStatus(_tree, _name, "ebottom", _branches);
  utils::setStatus(_tree, _name, "r9", _branches);
  utils::setStatus(_tree, _name, "etaWidth", _branches);
  utils::setStatus(_tree, _name, "phiWidth", _branches);
  utils::setStatus(_tree, _name, "time", _branches);
  utils::setStatus(_tree, _name, "timeSpan", _branches);
  utils::setStatus(_tree, _name, "regPt", _branches);
  utils::setStatus(_tree, _name, "smearedPt", _branches);
  utils::setStatus(_tree, _name, "originalPt", _branches);
  utils::setStatus(_tree, _name, "loose", _branches);
  utils::setStatus(_tree, _name, "medium", _branches);
  utils::setStatus(_tree, _name, "tight", _branches);
  utils::setStatus(_tree, _name, "highpt", _branches);
  utils::setStatus(_tree, _name, "pixelVeto", _branches);
  utils::setStatus(_tree, _name, "csafeVeto", _branches);
  utils::setStatus(_tree, _name, "triggerMatch", _branches);
  utils::setStatus(_tree, _name, "superCluster_", _branches);
  utils::setStatus(_tree, _name, "matchedPF_", _branches);
  utils::setStatus(_tree, _name, "matchedGen_", _branches);
}

panda::utils::BranchList
panda::Photon::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(ParticleP::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "pfPt"));
  blist.push_back(utils::getStatus(_tree, _name, "chIso"));
  blist.push_back(utils::getStatus(_tree, _name, "chIsoMax"));
  blist.push_back(utils::getStatus(_tree, _name, "nhIso"));
  blist.push_back(utils::getStatus(_tree, _name, "phIso"));
  blist.push_back(utils::getStatus(_tree, _name, "sieie"));
  blist.push_back(utils::getStatus(_tree, _name, "sipip"));
  blist.push_back(utils::getStatus(_tree, _name, "hOverE"));
  blist.push_back(utils::getStatus(_tree, _name, "genIso"));
  blist.push_back(utils::getStatus(_tree, _name, "mipEnergy"));
  blist.push_back(utils::getStatus(_tree, _name, "emax"));
  blist.push_back(utils::getStatus(_tree, _name, "e2nd"));
  blist.push_back(utils::getStatus(_tree, _name, "eleft"));
  blist.push_back(utils::getStatus(_tree, _name, "eright"));
  blist.push_back(utils::getStatus(_tree, _name, "etop"));
  blist.push_back(utils::getStatus(_tree, _name, "ebottom"));
  blist.push_back(utils::getStatus(_tree, _name, "r9"));
  blist.push_back(utils::getStatus(_tree, _name, "etaWidth"));
  blist.push_back(utils::getStatus(_tree, _name, "phiWidth"));
  blist.push_back(utils::getStatus(_tree, _name, "time"));
  blist.push_back(utils::getStatus(_tree, _name, "timeSpan"));
  blist.push_back(utils::getStatus(_tree, _name, "regPt"));
  blist.push_back(utils::getStatus(_tree, _name, "smearedPt"));
  blist.push_back(utils::getStatus(_tree, _name, "originalPt"));
  blist.push_back(utils::getStatus(_tree, _name, "loose"));
  blist.push_back(utils::getStatus(_tree, _name, "medium"));
  blist.push_back(utils::getStatus(_tree, _name, "tight"));
  blist.push_back(utils::getStatus(_tree, _name, "highpt"));
  blist.push_back(utils::getStatus(_tree, _name, "pixelVeto"));
  blist.push_back(utils::getStatus(_tree, _name, "csafeVeto"));
  blist.push_back(utils::getStatus(_tree, _name, "triggerMatch"));
  blist.push_back(utils::getStatus(_tree, _name, "superCluster_"));
  blist.push_back(utils::getStatus(_tree, _name, "matchedPF_"));
  blist.push_back(utils::getStatus(_tree, _name, "matchedGen_"));

  return blist;
}

void
panda::Photon::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  ParticleP::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "pfPt", pfPt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "chIso", chIso, _branches, _setStatus);
  utils::setAddress(_tree, _name, "chIsoMax", chIsoMax, _branches, _setStatus);
  utils::setAddress(_tree, _name, "nhIso", nhIso, _branches, _setStatus);
  utils::setAddress(_tree, _name, "phIso", phIso, _branches, _setStatus);
  utils::setAddress(_tree, _name, "sieie", sieie, _branches, _setStatus);
  utils::setAddress(_tree, _name, "sipip", sipip, _branches, _setStatus);
  utils::setAddress(_tree, _name, "hOverE", hOverE, _branches, _setStatus);
  utils::setAddress(_tree, _name, "genIso", genIso, _branches, _setStatus);
  utils::setAddress(_tree, _name, "mipEnergy", mipEnergy, _branches, _setStatus);
  utils::setAddress(_tree, _name, "emax", emax, _branches, _setStatus);
  utils::setAddress(_tree, _name, "e2nd", e2nd, _branches, _setStatus);
  utils::setAddress(_tree, _name, "eleft", eleft, _branches, _setStatus);
  utils::setAddress(_tree, _name, "eright", eright, _branches, _setStatus);
  utils::setAddress(_tree, _name, "etop", etop, _branches, _setStatus);
  utils::setAddress(_tree, _name, "ebottom", ebottom, _branches, _setStatus);
  utils::setAddress(_tree, _name, "r9", r9, _branches, _setStatus);
  utils::setAddress(_tree, _name, "etaWidth", etaWidth, _branches, _setStatus);
  utils::setAddress(_tree, _name, "phiWidth", phiWidth, _branches, _setStatus);
  utils::setAddress(_tree, _name, "time", time, _branches, _setStatus);
  utils::setAddress(_tree, _name, "timeSpan", timeSpan, _branches, _setStatus);
  utils::setAddress(_tree, _name, "regPt", regPt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "smearedPt", smearedPt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "originalPt", originalPt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "loose", loose, _branches, _setStatus);
  utils::setAddress(_tree, _name, "medium", medium, _branches, _setStatus);
  utils::setAddress(_tree, _name, "tight", tight, _branches, _setStatus);
  utils::setAddress(_tree, _name, "highpt", highpt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "pixelVeto", pixelVeto, _branches, _setStatus);
  utils::setAddress(_tree, _name, "csafeVeto", csafeVeto, _branches, _setStatus);
  utils::setAddress(_tree, _name, "triggerMatch", triggerMatch, _branches, _setStatus);
  utils::setAddress(_tree, _name, "superCluster_", superCluster_, _branches, _setStatus);
  utils::setAddress(_tree, _name, "matchedPF_", matchedPF_, _branches, _setStatus);
  utils::setAddress(_tree, _name, "matchedGen_", matchedGen_, _branches, _setStatus);
}

void
panda::Photon::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  ParticleP::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "pfPt", size, 'F', pfPt, _branches);
  utils::book(_tree, _name, "chIso", size, 'F', chIso, _branches);
  utils::book(_tree, _name, "chIsoMax", size, 'F', chIsoMax, _branches);
  utils::book(_tree, _name, "nhIso", size, 'F', nhIso, _branches);
  utils::book(_tree, _name, "phIso", size, 'F', phIso, _branches);
  utils::book(_tree, _name, "sieie", size, 'F', sieie, _branches);
  utils::book(_tree, _name, "sipip", size, 'F', sipip, _branches);
  utils::book(_tree, _name, "hOverE", size, 'F', hOverE, _branches);
  utils::book(_tree, _name, "genIso", size, 'F', genIso, _branches);
  utils::book(_tree, _name, "mipEnergy", size, 'F', mipEnergy, _branches);
  utils::book(_tree, _name, "emax", size, 'F', emax, _branches);
  utils::book(_tree, _name, "e2nd", size, 'F', e2nd, _branches);
  utils::book(_tree, _name, "eleft", size, 'F', eleft, _branches);
  utils::book(_tree, _name, "eright", size, 'F', eright, _branches);
  utils::book(_tree, _name, "etop", size, 'F', etop, _branches);
  utils::book(_tree, _name, "ebottom", size, 'F', ebottom, _branches);
  utils::book(_tree, _name, "r9", size, 'F', r9, _branches);
  utils::book(_tree, _name, "etaWidth", size, 'F', etaWidth, _branches);
  utils::book(_tree, _name, "phiWidth", size, 'F', phiWidth, _branches);
  utils::book(_tree, _name, "time", size, 'F', time, _branches);
  utils::book(_tree, _name, "timeSpan", size, 'F', timeSpan, _branches);
  utils::book(_tree, _name, "regPt", size, 'F', regPt, _branches);
  utils::book(_tree, _name, "smearedPt", size, 'F', smearedPt, _branches);
  utils::book(_tree, _name, "originalPt", size, 'F', originalPt, _branches);
  utils::book(_tree, _name, "loose", size, 'O', loose, _branches);
  utils::book(_tree, _name, "medium", size, 'O', medium, _branches);
  utils::book(_tree, _name, "tight", size, 'O', tight, _branches);
  utils::book(_tree, _name, "highpt", size, 'O', highpt, _branches);
  utils::book(_tree, _name, "pixelVeto", size, 'O', pixelVeto, _branches);
  utils::book(_tree, _name, "csafeVeto", size, 'O', csafeVeto, _branches);
  utils::book(_tree, _name, "triggerMatch", size + TString::Format("[%d]", nTriggerObjects), 'O', triggerMatch, _branches);
  utils::book(_tree, _name, "superCluster_", size, 'S', superCluster_, _branches);
  utils::book(_tree, _name, "matchedPF_", size, 'S', matchedPF_, _branches);
  utils::book(_tree, _name, "matchedGen_", size, 'S', matchedGen_, _branches);
}

void
panda::Photon::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  ParticleP::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "pfPt");
  utils::resetAddress(_tree, _name, "chIso");
  utils::resetAddress(_tree, _name, "chIsoMax");
  utils::resetAddress(_tree, _name, "nhIso");
  utils::resetAddress(_tree, _name, "phIso");
  utils::resetAddress(_tree, _name, "sieie");
  utils::resetAddress(_tree, _name, "sipip");
  utils::resetAddress(_tree, _name, "hOverE");
  utils::resetAddress(_tree, _name, "genIso");
  utils::resetAddress(_tree, _name, "mipEnergy");
  utils::resetAddress(_tree, _name, "emax");
  utils::resetAddress(_tree, _name, "e2nd");
  utils::resetAddress(_tree, _name, "eleft");
  utils::resetAddress(_tree, _name, "eright");
  utils::resetAddress(_tree, _name, "etop");
  utils::resetAddress(_tree, _name, "ebottom");
  utils::resetAddress(_tree, _name, "r9");
  utils::resetAddress(_tree, _name, "etaWidth");
  utils::resetAddress(_tree, _name, "phiWidth");
  utils::resetAddress(_tree, _name, "time");
  utils::resetAddress(_tree, _name, "timeSpan");
  utils::resetAddress(_tree, _name, "regPt");
  utils::resetAddress(_tree, _name, "smearedPt");
  utils::resetAddress(_tree, _name, "originalPt");
  utils::resetAddress(_tree, _name, "loose");
  utils::resetAddress(_tree, _name, "medium");
  utils::resetAddress(_tree, _name, "tight");
  utils::resetAddress(_tree, _name, "highpt");
  utils::resetAddress(_tree, _name, "pixelVeto");
  utils::resetAddress(_tree, _name, "csafeVeto");
  utils::resetAddress(_tree, _name, "triggerMatch");
  utils::resetAddress(_tree, _name, "superCluster_");
  utils::resetAddress(_tree, _name, "matchedPF_");
  utils::resetAddress(_tree, _name, "matchedGen_");
}

void
panda::Photon::datastore::resizeVectors_(UInt_t _size)
{
  ParticleP::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::Photon::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return Photon::getListOfBranches().fullNames(_name);
}

panda::Photon::Photon(char const* _name/* = ""*/) :
  ParticleP(new PhotonArray(1, _name)),
  pfPt(gStore.getData(this).pfPt[0]),
  chIso(gStore.getData(this).chIso[0]),
  chIsoMax(gStore.getData(this).chIsoMax[0]),
  nhIso(gStore.getData(this).nhIso[0]),
  phIso(gStore.getData(this).phIso[0]),
  sieie(gStore.getData(this).sieie[0]),
  sipip(gStore.getData(this).sipip[0]),
  hOverE(gStore.getData(this).hOverE[0]),
  genIso(gStore.getData(this).genIso[0]),
  mipEnergy(gStore.getData(this).mipEnergy[0]),
  emax(gStore.getData(this).emax[0]),
  e2nd(gStore.getData(this).e2nd[0]),
  eleft(gStore.getData(this).eleft[0]),
  eright(gStore.getData(this).eright[0]),
  etop(gStore.getData(this).etop[0]),
  ebottom(gStore.getData(this).ebottom[0]),
  r9(gStore.getData(this).r9[0]),
  etaWidth(gStore.getData(this).etaWidth[0]),
  phiWidth(gStore.getData(this).phiWidth[0]),
  time(gStore.getData(this).time[0]),
  timeSpan(gStore.getData(this).timeSpan[0]),
  regPt(gStore.getData(this).regPt[0]),
  smearedPt(gStore.getData(this).smearedPt[0]),
  originalPt(gStore.getData(this).originalPt[0]),
  loose(gStore.getData(this).loose[0]),
  medium(gStore.getData(this).medium[0]),
  tight(gStore.getData(this).tight[0]),
  highpt(gStore.getData(this).highpt[0]),
  pixelVeto(gStore.getData(this).pixelVeto[0]),
  csafeVeto(gStore.getData(this).csafeVeto[0]),
  triggerMatch(gStore.getData(this).triggerMatch[0]),
  superCluster(gStore.getData(this).superClusterContainer_, gStore.getData(this).superCluster_[0]),
  matchedPF(gStore.getData(this).matchedPFContainer_, gStore.getData(this).matchedPF_[0]),
  matchedGen(gStore.getData(this).matchedGenContainer_, gStore.getData(this).matchedGen_[0])
{
}

panda::Photon::Photon(Photon const& _src) :
  ParticleP(new PhotonArray(1, _src.getName())),
  pfPt(gStore.getData(this).pfPt[0]),
  chIso(gStore.getData(this).chIso[0]),
  chIsoMax(gStore.getData(this).chIsoMax[0]),
  nhIso(gStore.getData(this).nhIso[0]),
  phIso(gStore.getData(this).phIso[0]),
  sieie(gStore.getData(this).sieie[0]),
  sipip(gStore.getData(this).sipip[0]),
  hOverE(gStore.getData(this).hOverE[0]),
  genIso(gStore.getData(this).genIso[0]),
  mipEnergy(gStore.getData(this).mipEnergy[0]),
  emax(gStore.getData(this).emax[0]),
  e2nd(gStore.getData(this).e2nd[0]),
  eleft(gStore.getData(this).eleft[0]),
  eright(gStore.getData(this).eright[0]),
  etop(gStore.getData(this).etop[0]),
  ebottom(gStore.getData(this).ebottom[0]),
  r9(gStore.getData(this).r9[0]),
  etaWidth(gStore.getData(this).etaWidth[0]),
  phiWidth(gStore.getData(this).phiWidth[0]),
  time(gStore.getData(this).time[0]),
  timeSpan(gStore.getData(this).timeSpan[0]),
  regPt(gStore.getData(this).regPt[0]),
  smearedPt(gStore.getData(this).smearedPt[0]),
  originalPt(gStore.getData(this).originalPt[0]),
  loose(gStore.getData(this).loose[0]),
  medium(gStore.getData(this).medium[0]),
  tight(gStore.getData(this).tight[0]),
  highpt(gStore.getData(this).highpt[0]),
  pixelVeto(gStore.getData(this).pixelVeto[0]),
  csafeVeto(gStore.getData(this).csafeVeto[0]),
  triggerMatch(gStore.getData(this).triggerMatch[0]),
  superCluster(gStore.getData(this).superClusterContainer_, gStore.getData(this).superCluster_[0]),
  matchedPF(gStore.getData(this).matchedPFContainer_, gStore.getData(this).matchedPF_[0]),
  matchedGen(gStore.getData(this).matchedGenContainer_, gStore.getData(this).matchedGen_[0])
{
  ParticleP::operator=(_src);

  pfPt = _src.pfPt;
  chIso = _src.chIso;
  chIsoMax = _src.chIsoMax;
  nhIso = _src.nhIso;
  phIso = _src.phIso;
  sieie = _src.sieie;
  sipip = _src.sipip;
  hOverE = _src.hOverE;
  genIso = _src.genIso;
  mipEnergy = _src.mipEnergy;
  emax = _src.emax;
  e2nd = _src.e2nd;
  eleft = _src.eleft;
  eright = _src.eright;
  etop = _src.etop;
  ebottom = _src.ebottom;
  r9 = _src.r9;
  etaWidth = _src.etaWidth;
  phiWidth = _src.phiWidth;
  time = _src.time;
  timeSpan = _src.timeSpan;
  regPt = _src.regPt;
  smearedPt = _src.smearedPt;
  originalPt = _src.originalPt;
  loose = _src.loose;
  medium = _src.medium;
  tight = _src.tight;
  highpt = _src.highpt;
  pixelVeto = _src.pixelVeto;
  csafeVeto = _src.csafeVeto;
  std::memcpy(triggerMatch, _src.triggerMatch, sizeof(Bool_t) * nTriggerObjects);
  superCluster = _src.superCluster;
  matchedPF = _src.matchedPF;
  matchedGen = _src.matchedGen;
}

panda::Photon::Photon(datastore& _data, UInt_t _idx) :
  ParticleP(_data, _idx),
  pfPt(_data.pfPt[_idx]),
  chIso(_data.chIso[_idx]),
  chIsoMax(_data.chIsoMax[_idx]),
  nhIso(_data.nhIso[_idx]),
  phIso(_data.phIso[_idx]),
  sieie(_data.sieie[_idx]),
  sipip(_data.sipip[_idx]),
  hOverE(_data.hOverE[_idx]),
  genIso(_data.genIso[_idx]),
  mipEnergy(_data.mipEnergy[_idx]),
  emax(_data.emax[_idx]),
  e2nd(_data.e2nd[_idx]),
  eleft(_data.eleft[_idx]),
  eright(_data.eright[_idx]),
  etop(_data.etop[_idx]),
  ebottom(_data.ebottom[_idx]),
  r9(_data.r9[_idx]),
  etaWidth(_data.etaWidth[_idx]),
  phiWidth(_data.phiWidth[_idx]),
  time(_data.time[_idx]),
  timeSpan(_data.timeSpan[_idx]),
  regPt(_data.regPt[_idx]),
  smearedPt(_data.smearedPt[_idx]),
  originalPt(_data.originalPt[_idx]),
  loose(_data.loose[_idx]),
  medium(_data.medium[_idx]),
  tight(_data.tight[_idx]),
  highpt(_data.highpt[_idx]),
  pixelVeto(_data.pixelVeto[_idx]),
  csafeVeto(_data.csafeVeto[_idx]),
  triggerMatch(_data.triggerMatch[_idx]),
  superCluster(_data.superClusterContainer_, _data.superCluster_[_idx]),
  matchedPF(_data.matchedPFContainer_, _data.matchedPF_[_idx]),
  matchedGen(_data.matchedGenContainer_, _data.matchedGen_[_idx])
{
}

panda::Photon::Photon(ArrayBase* _array) :
  ParticleP(_array),
  pfPt(gStore.getData(this).pfPt[0]),
  chIso(gStore.getData(this).chIso[0]),
  chIsoMax(gStore.getData(this).chIsoMax[0]),
  nhIso(gStore.getData(this).nhIso[0]),
  phIso(gStore.getData(this).phIso[0]),
  sieie(gStore.getData(this).sieie[0]),
  sipip(gStore.getData(this).sipip[0]),
  hOverE(gStore.getData(this).hOverE[0]),
  genIso(gStore.getData(this).genIso[0]),
  mipEnergy(gStore.getData(this).mipEnergy[0]),
  emax(gStore.getData(this).emax[0]),
  e2nd(gStore.getData(this).e2nd[0]),
  eleft(gStore.getData(this).eleft[0]),
  eright(gStore.getData(this).eright[0]),
  etop(gStore.getData(this).etop[0]),
  ebottom(gStore.getData(this).ebottom[0]),
  r9(gStore.getData(this).r9[0]),
  etaWidth(gStore.getData(this).etaWidth[0]),
  phiWidth(gStore.getData(this).phiWidth[0]),
  time(gStore.getData(this).time[0]),
  timeSpan(gStore.getData(this).timeSpan[0]),
  regPt(gStore.getData(this).regPt[0]),
  smearedPt(gStore.getData(this).smearedPt[0]),
  originalPt(gStore.getData(this).originalPt[0]),
  loose(gStore.getData(this).loose[0]),
  medium(gStore.getData(this).medium[0]),
  tight(gStore.getData(this).tight[0]),
  highpt(gStore.getData(this).highpt[0]),
  pixelVeto(gStore.getData(this).pixelVeto[0]),
  csafeVeto(gStore.getData(this).csafeVeto[0]),
  triggerMatch(gStore.getData(this).triggerMatch[0]),
  superCluster(gStore.getData(this).superClusterContainer_, gStore.getData(this).superCluster_[0]),
  matchedPF(gStore.getData(this).matchedPFContainer_, gStore.getData(this).matchedPF_[0]),
  matchedGen(gStore.getData(this).matchedGenContainer_, gStore.getData(this).matchedGen_[0])
{
}

panda::Photon::~Photon()
{
  destructor();
}

void
panda::Photon::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM Photon.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    ParticleP::destructor(kTRUE);
}

panda::Photon&
panda::Photon::operator=(Photon const& _src)
{
  ParticleP::operator=(_src);

  pfPt = _src.pfPt;
  chIso = _src.chIso;
  chIsoMax = _src.chIsoMax;
  nhIso = _src.nhIso;
  phIso = _src.phIso;
  sieie = _src.sieie;
  sipip = _src.sipip;
  hOverE = _src.hOverE;
  genIso = _src.genIso;
  mipEnergy = _src.mipEnergy;
  emax = _src.emax;
  e2nd = _src.e2nd;
  eleft = _src.eleft;
  eright = _src.eright;
  etop = _src.etop;
  ebottom = _src.ebottom;
  r9 = _src.r9;
  etaWidth = _src.etaWidth;
  phiWidth = _src.phiWidth;
  time = _src.time;
  timeSpan = _src.timeSpan;
  regPt = _src.regPt;
  smearedPt = _src.smearedPt;
  originalPt = _src.originalPt;
  loose = _src.loose;
  medium = _src.medium;
  tight = _src.tight;
  highpt = _src.highpt;
  pixelVeto = _src.pixelVeto;
  csafeVeto = _src.csafeVeto;
  std::memcpy(triggerMatch, _src.triggerMatch, sizeof(Bool_t) * nTriggerObjects);
  superCluster = _src.superCluster;
  matchedPF = _src.matchedPF;
  matchedGen = _src.matchedGen;

  /* BEGIN CUSTOM Photon.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::Photon::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  ParticleP::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "pfPt", "", 'F', &pfPt, _branches);
  utils::book(_tree, _name, "chIso", "", 'F', &chIso, _branches);
  utils::book(_tree, _name, "chIsoMax", "", 'F', &chIsoMax, _branches);
  utils::book(_tree, _name, "nhIso", "", 'F', &nhIso, _branches);
  utils::book(_tree, _name, "phIso", "", 'F', &phIso, _branches);
  utils::book(_tree, _name, "sieie", "", 'F', &sieie, _branches);
  utils::book(_tree, _name, "sipip", "", 'F', &sipip, _branches);
  utils::book(_tree, _name, "hOverE", "", 'F', &hOverE, _branches);
  utils::book(_tree, _name, "genIso", "", 'F', &genIso, _branches);
  utils::book(_tree, _name, "mipEnergy", "", 'F', &mipEnergy, _branches);
  utils::book(_tree, _name, "emax", "", 'F', &emax, _branches);
  utils::book(_tree, _name, "e2nd", "", 'F', &e2nd, _branches);
  utils::book(_tree, _name, "eleft", "", 'F', &eleft, _branches);
  utils::book(_tree, _name, "eright", "", 'F', &eright, _branches);
  utils::book(_tree, _name, "etop", "", 'F', &etop, _branches);
  utils::book(_tree, _name, "ebottom", "", 'F', &ebottom, _branches);
  utils::book(_tree, _name, "r9", "", 'F', &r9, _branches);
  utils::book(_tree, _name, "etaWidth", "", 'F', &etaWidth, _branches);
  utils::book(_tree, _name, "phiWidth", "", 'F', &phiWidth, _branches);
  utils::book(_tree, _name, "time", "", 'F', &time, _branches);
  utils::book(_tree, _name, "timeSpan", "", 'F', &timeSpan, _branches);
  utils::book(_tree, _name, "regPt", "", 'F', &regPt, _branches);
  utils::book(_tree, _name, "smearedPt", "", 'F', &smearedPt, _branches);
  utils::book(_tree, _name, "originalPt", "", 'F', &originalPt, _branches);
  utils::book(_tree, _name, "loose", "", 'O', &loose, _branches);
  utils::book(_tree, _name, "medium", "", 'O', &medium, _branches);
  utils::book(_tree, _name, "tight", "", 'O', &tight, _branches);
  utils::book(_tree, _name, "highpt", "", 'O', &highpt, _branches);
  utils::book(_tree, _name, "pixelVeto", "", 'O', &pixelVeto, _branches);
  utils::book(_tree, _name, "csafeVeto", "", 'O', &csafeVeto, _branches);
  utils::book(_tree, _name, "triggerMatch", TString::Format("[%d]", nTriggerObjects), 'O', triggerMatch, _branches);
  utils::book(_tree, _name, "superCluster_", "", 'S', gStore.getData(this).superCluster_, _branches);
  utils::book(_tree, _name, "matchedPF_", "", 'S', gStore.getData(this).matchedPF_, _branches);
  utils::book(_tree, _name, "matchedGen_", "", 'S', gStore.getData(this).matchedGen_, _branches);
}

void
panda::Photon::doInit_()
{
  ParticleP::doInit_();

  pfPt = -1.;
  chIso = 0.;
  chIsoMax = 0.;
  nhIso = 0.;
  phIso = 0.;
  sieie = 0.;
  sipip = 0.;
  hOverE = 0.;
  genIso = 0.;
  mipEnergy = 0.;
  emax = 0.;
  e2nd = 0.;
  eleft = 0.;
  eright = 0.;
  etop = 0.;
  ebottom = 0.;
  r9 = 0.;
  etaWidth = 0.;
  phiWidth = 0.;
  time = 0.;
  timeSpan = 0.;
  regPt = 0.;
  smearedPt = 0.;
  originalPt = -1.;
  loose = false;
  medium = false;
  tight = false;
  highpt = false;
  pixelVeto = false;
  csafeVeto = false;
  for (auto& p0 : triggerMatch) p0 = false;
  superCluster.init();
  matchedPF.init();
  matchedGen.init();

  /* BEGIN CUSTOM Photon.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::Photon::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM Photon.cc.print */
  if (_level >= 3) {
    Photon::dump(_out);
  }
  else if (_level == 2) {
    ParticleP::print(_out, _level);
    
    _out << "pfPt = " << pfPt << std::endl;
    _out << "smearedPt = " << smearedPt << std::endl;
    _out << "regPt = " << regPt << std::endl;
    _out << "hOverE = " << hOverE << std::endl;
    _out << "chIso = " << chIso << std::endl;
    _out << "chIsoMax = " << chIsoMax << std::endl;
    _out << "nhIso = " << nhIso << std::endl;
    _out << "phIso = " << phIso << std::endl;
    _out << "sieie = " << sieie << std::endl;
    _out << "sipip = " << sipip << std::endl;
    _out << "mipEnergy = " << mipEnergy << std::endl;
    _out << "time = " << time << std::endl;
    _out << "pixelVeto = " << pixelVeto << std::endl;
  }
  else
    return;
  /* END CUSTOM */
}

void
panda::Photon::dump(std::ostream& _out/* = std::cout*/) const
{
  ParticleP::dump(_out);

  _out << "pfPt = " << pfPt << std::endl;
  _out << "chIso = " << chIso << std::endl;
  _out << "chIsoMax = " << chIsoMax << std::endl;
  _out << "nhIso = " << nhIso << std::endl;
  _out << "phIso = " << phIso << std::endl;
  _out << "sieie = " << sieie << std::endl;
  _out << "sipip = " << sipip << std::endl;
  _out << "hOverE = " << hOverE << std::endl;
  _out << "genIso = " << genIso << std::endl;
  _out << "mipEnergy = " << mipEnergy << std::endl;
  _out << "emax = " << emax << std::endl;
  _out << "e2nd = " << e2nd << std::endl;
  _out << "eleft = " << eleft << std::endl;
  _out << "eright = " << eright << std::endl;
  _out << "etop = " << etop << std::endl;
  _out << "ebottom = " << ebottom << std::endl;
  _out << "r9 = " << r9 << std::endl;
  _out << "etaWidth = " << etaWidth << std::endl;
  _out << "phiWidth = " << phiWidth << std::endl;
  _out << "time = " << time << std::endl;
  _out << "timeSpan = " << timeSpan << std::endl;
  _out << "regPt = " << regPt << std::endl;
  _out << "smearedPt = " << smearedPt << std::endl;
  _out << "originalPt = " << originalPt << std::endl;
  _out << "loose = " << loose << std::endl;
  _out << "medium = " << medium << std::endl;
  _out << "tight = " << tight << std::endl;
  _out << "highpt = " << highpt << std::endl;
  _out << "pixelVeto = " << pixelVeto << std::endl;
  _out << "csafeVeto = " << csafeVeto << std::endl;
  _out << "triggerMatch = " << triggerMatch << std::endl;
  _out << "superCluster = " << superCluster << std::endl;
  _out << "matchedPF = " << matchedPF << std::endl;
  _out << "matchedGen = " << matchedGen << std::endl;
}

/* BEGIN CUSTOM Photon.cc.global */
/* END CUSTOM */
