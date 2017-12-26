#include "../interface/Electron.h"

TString panda::Electron::TriggerObjectName[] = {
  "fEl23El12FirstLeg",
  "fEl23El12SecondLeg",
  "fEl25Tight",
  "fEl27Loose",
  "fEl27Tight",
  "fEl120Ph",
  "fEl135Ph",
  "fEl165HE10Ph",
  "fEl175Ph",
  "fEl22EBR9IsoPh",
  "fEl36EBR9IsoPh",
  "fEl50EBR9IsoPh",
  "fEl75EBR9IsoPh",
  "fEl90EBR9IsoPh",
  "fEl120EBR9IsoPh"
};

/*static*/
panda::utils::BranchList
panda::Electron::getListOfBranches()
{
  utils::BranchList blist;
  blist += Lepton::getListOfBranches();
  blist += {"hltsafe", "chIsoPh", "nhIsoPh", "phIsoPh", "ecalIso", "hcalIso", "isoPUOffset", "sieie", "sipip", "eseed", "hOverE", "regPt", "smearedPt", "originalPt", "dxy", "dz", "veto", "triggerMatch", "superCluster_"};
  return blist;
}

void
panda::Electron::datastore::allocate(UInt_t _nmax)
{
  Lepton::datastore::allocate(_nmax);

  hltsafe = new Bool_t[nmax_];
  chIsoPh = new Float_t[nmax_];
  nhIsoPh = new Float_t[nmax_];
  phIsoPh = new Float_t[nmax_];
  ecalIso = new Float_t[nmax_];
  hcalIso = new Float_t[nmax_];
  isoPUOffset = new Float_t[nmax_];
  sieie = new Float_t[nmax_];
  sipip = new Float_t[nmax_];
  eseed = new Float_t[nmax_];
  hOverE = new Float_t[nmax_];
  regPt = new Float_t[nmax_];
  smearedPt = new Float_t[nmax_];
  originalPt = new Float_t[nmax_];
  dxy = new Float_t[nmax_];
  dz = new Float_t[nmax_];
  veto = new Bool_t[nmax_];
  triggerMatch = new Bool_t[nmax_][nTriggerObjects];
  superCluster_ = new Short_t[nmax_];
}

void
panda::Electron::datastore::deallocate()
{
  Lepton::datastore::deallocate();

  delete [] hltsafe;
  hltsafe = 0;
  delete [] chIsoPh;
  chIsoPh = 0;
  delete [] nhIsoPh;
  nhIsoPh = 0;
  delete [] phIsoPh;
  phIsoPh = 0;
  delete [] ecalIso;
  ecalIso = 0;
  delete [] hcalIso;
  hcalIso = 0;
  delete [] isoPUOffset;
  isoPUOffset = 0;
  delete [] sieie;
  sieie = 0;
  delete [] sipip;
  sipip = 0;
  delete [] eseed;
  eseed = 0;
  delete [] hOverE;
  hOverE = 0;
  delete [] regPt;
  regPt = 0;
  delete [] smearedPt;
  smearedPt = 0;
  delete [] originalPt;
  originalPt = 0;
  delete [] dxy;
  dxy = 0;
  delete [] dz;
  dz = 0;
  delete [] veto;
  veto = 0;
  delete [] triggerMatch;
  triggerMatch = 0;
  delete [] superCluster_;
  superCluster_ = 0;
}

void
panda::Electron::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  Lepton::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "hltsafe", _branches);
  utils::setStatus(_tree, _name, "chIsoPh", _branches);
  utils::setStatus(_tree, _name, "nhIsoPh", _branches);
  utils::setStatus(_tree, _name, "phIsoPh", _branches);
  utils::setStatus(_tree, _name, "ecalIso", _branches);
  utils::setStatus(_tree, _name, "hcalIso", _branches);
  utils::setStatus(_tree, _name, "isoPUOffset", _branches);
  utils::setStatus(_tree, _name, "sieie", _branches);
  utils::setStatus(_tree, _name, "sipip", _branches);
  utils::setStatus(_tree, _name, "eseed", _branches);
  utils::setStatus(_tree, _name, "hOverE", _branches);
  utils::setStatus(_tree, _name, "regPt", _branches);
  utils::setStatus(_tree, _name, "smearedPt", _branches);
  utils::setStatus(_tree, _name, "originalPt", _branches);
  utils::setStatus(_tree, _name, "dxy", _branches);
  utils::setStatus(_tree, _name, "dz", _branches);
  utils::setStatus(_tree, _name, "veto", _branches);
  utils::setStatus(_tree, _name, "triggerMatch", _branches);
  utils::setStatus(_tree, _name, "superCluster_", _branches);
}

panda::utils::BranchList
panda::Electron::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(Lepton::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "hltsafe"));
  blist.push_back(utils::getStatus(_tree, _name, "chIsoPh"));
  blist.push_back(utils::getStatus(_tree, _name, "nhIsoPh"));
  blist.push_back(utils::getStatus(_tree, _name, "phIsoPh"));
  blist.push_back(utils::getStatus(_tree, _name, "ecalIso"));
  blist.push_back(utils::getStatus(_tree, _name, "hcalIso"));
  blist.push_back(utils::getStatus(_tree, _name, "isoPUOffset"));
  blist.push_back(utils::getStatus(_tree, _name, "sieie"));
  blist.push_back(utils::getStatus(_tree, _name, "sipip"));
  blist.push_back(utils::getStatus(_tree, _name, "eseed"));
  blist.push_back(utils::getStatus(_tree, _name, "hOverE"));
  blist.push_back(utils::getStatus(_tree, _name, "regPt"));
  blist.push_back(utils::getStatus(_tree, _name, "smearedPt"));
  blist.push_back(utils::getStatus(_tree, _name, "originalPt"));
  blist.push_back(utils::getStatus(_tree, _name, "dxy"));
  blist.push_back(utils::getStatus(_tree, _name, "dz"));
  blist.push_back(utils::getStatus(_tree, _name, "veto"));
  blist.push_back(utils::getStatus(_tree, _name, "triggerMatch"));
  blist.push_back(utils::getStatus(_tree, _name, "superCluster_"));

  return blist;
}

void
panda::Electron::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  Lepton::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "hltsafe", hltsafe, _branches, _setStatus);
  utils::setAddress(_tree, _name, "chIsoPh", chIsoPh, _branches, _setStatus);
  utils::setAddress(_tree, _name, "nhIsoPh", nhIsoPh, _branches, _setStatus);
  utils::setAddress(_tree, _name, "phIsoPh", phIsoPh, _branches, _setStatus);
  utils::setAddress(_tree, _name, "ecalIso", ecalIso, _branches, _setStatus);
  utils::setAddress(_tree, _name, "hcalIso", hcalIso, _branches, _setStatus);
  utils::setAddress(_tree, _name, "isoPUOffset", isoPUOffset, _branches, _setStatus);
  utils::setAddress(_tree, _name, "sieie", sieie, _branches, _setStatus);
  utils::setAddress(_tree, _name, "sipip", sipip, _branches, _setStatus);
  utils::setAddress(_tree, _name, "eseed", eseed, _branches, _setStatus);
  utils::setAddress(_tree, _name, "hOverE", hOverE, _branches, _setStatus);
  utils::setAddress(_tree, _name, "regPt", regPt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "smearedPt", smearedPt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "originalPt", originalPt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "dxy", dxy, _branches, _setStatus);
  utils::setAddress(_tree, _name, "dz", dz, _branches, _setStatus);
  utils::setAddress(_tree, _name, "veto", veto, _branches, _setStatus);
  utils::setAddress(_tree, _name, "triggerMatch", triggerMatch, _branches, _setStatus);
  utils::setAddress(_tree, _name, "superCluster_", superCluster_, _branches, _setStatus);
}

void
panda::Electron::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  Lepton::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "hltsafe", size, 'O', hltsafe, _branches);
  utils::book(_tree, _name, "chIsoPh", size, 'F', chIsoPh, _branches);
  utils::book(_tree, _name, "nhIsoPh", size, 'F', nhIsoPh, _branches);
  utils::book(_tree, _name, "phIsoPh", size, 'F', phIsoPh, _branches);
  utils::book(_tree, _name, "ecalIso", size, 'F', ecalIso, _branches);
  utils::book(_tree, _name, "hcalIso", size, 'F', hcalIso, _branches);
  utils::book(_tree, _name, "isoPUOffset", size, 'F', isoPUOffset, _branches);
  utils::book(_tree, _name, "sieie", size, 'F', sieie, _branches);
  utils::book(_tree, _name, "sipip", size, 'F', sipip, _branches);
  utils::book(_tree, _name, "eseed", size, 'F', eseed, _branches);
  utils::book(_tree, _name, "hOverE", size, 'F', hOverE, _branches);
  utils::book(_tree, _name, "regPt", size, 'F', regPt, _branches);
  utils::book(_tree, _name, "smearedPt", size, 'F', smearedPt, _branches);
  utils::book(_tree, _name, "originalPt", size, 'F', originalPt, _branches);
  utils::book(_tree, _name, "dxy", size, 'F', dxy, _branches);
  utils::book(_tree, _name, "dz", size, 'F', dz, _branches);
  utils::book(_tree, _name, "veto", size, 'O', veto, _branches);
  utils::book(_tree, _name, "triggerMatch", size + TString::Format("[%d]", nTriggerObjects), 'O', triggerMatch, _branches);
  utils::book(_tree, _name, "superCluster_", size, 'S', superCluster_, _branches);
}

void
panda::Electron::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  Lepton::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "hltsafe");
  utils::resetAddress(_tree, _name, "chIsoPh");
  utils::resetAddress(_tree, _name, "nhIsoPh");
  utils::resetAddress(_tree, _name, "phIsoPh");
  utils::resetAddress(_tree, _name, "ecalIso");
  utils::resetAddress(_tree, _name, "hcalIso");
  utils::resetAddress(_tree, _name, "isoPUOffset");
  utils::resetAddress(_tree, _name, "sieie");
  utils::resetAddress(_tree, _name, "sipip");
  utils::resetAddress(_tree, _name, "eseed");
  utils::resetAddress(_tree, _name, "hOverE");
  utils::resetAddress(_tree, _name, "regPt");
  utils::resetAddress(_tree, _name, "smearedPt");
  utils::resetAddress(_tree, _name, "originalPt");
  utils::resetAddress(_tree, _name, "dxy");
  utils::resetAddress(_tree, _name, "dz");
  utils::resetAddress(_tree, _name, "veto");
  utils::resetAddress(_tree, _name, "triggerMatch");
  utils::resetAddress(_tree, _name, "superCluster_");
}

void
panda::Electron::datastore::resizeVectors_(UInt_t _size)
{
  Lepton::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::Electron::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return Electron::getListOfBranches().fullNames(_name);
}

panda::Electron::Electron(char const* _name/* = ""*/) :
  Lepton(new ElectronArray(1, _name)),
  hltsafe(gStore.getData(this).hltsafe[0]),
  chIsoPh(gStore.getData(this).chIsoPh[0]),
  nhIsoPh(gStore.getData(this).nhIsoPh[0]),
  phIsoPh(gStore.getData(this).phIsoPh[0]),
  ecalIso(gStore.getData(this).ecalIso[0]),
  hcalIso(gStore.getData(this).hcalIso[0]),
  isoPUOffset(gStore.getData(this).isoPUOffset[0]),
  sieie(gStore.getData(this).sieie[0]),
  sipip(gStore.getData(this).sipip[0]),
  eseed(gStore.getData(this).eseed[0]),
  hOverE(gStore.getData(this).hOverE[0]),
  regPt(gStore.getData(this).regPt[0]),
  smearedPt(gStore.getData(this).smearedPt[0]),
  originalPt(gStore.getData(this).originalPt[0]),
  dxy(gStore.getData(this).dxy[0]),
  dz(gStore.getData(this).dz[0]),
  veto(gStore.getData(this).veto[0]),
  triggerMatch(gStore.getData(this).triggerMatch[0]),
  superCluster(gStore.getData(this).superClusterContainer_, gStore.getData(this).superCluster_[0])
{
}

panda::Electron::Electron(Electron const& _src) :
  Lepton(new ElectronArray(1, _src.getName())),
  hltsafe(gStore.getData(this).hltsafe[0]),
  chIsoPh(gStore.getData(this).chIsoPh[0]),
  nhIsoPh(gStore.getData(this).nhIsoPh[0]),
  phIsoPh(gStore.getData(this).phIsoPh[0]),
  ecalIso(gStore.getData(this).ecalIso[0]),
  hcalIso(gStore.getData(this).hcalIso[0]),
  isoPUOffset(gStore.getData(this).isoPUOffset[0]),
  sieie(gStore.getData(this).sieie[0]),
  sipip(gStore.getData(this).sipip[0]),
  eseed(gStore.getData(this).eseed[0]),
  hOverE(gStore.getData(this).hOverE[0]),
  regPt(gStore.getData(this).regPt[0]),
  smearedPt(gStore.getData(this).smearedPt[0]),
  originalPt(gStore.getData(this).originalPt[0]),
  dxy(gStore.getData(this).dxy[0]),
  dz(gStore.getData(this).dz[0]),
  veto(gStore.getData(this).veto[0]),
  triggerMatch(gStore.getData(this).triggerMatch[0]),
  superCluster(gStore.getData(this).superClusterContainer_, gStore.getData(this).superCluster_[0])
{
  Lepton::operator=(_src);

  hltsafe = _src.hltsafe;
  chIsoPh = _src.chIsoPh;
  nhIsoPh = _src.nhIsoPh;
  phIsoPh = _src.phIsoPh;
  ecalIso = _src.ecalIso;
  hcalIso = _src.hcalIso;
  isoPUOffset = _src.isoPUOffset;
  sieie = _src.sieie;
  sipip = _src.sipip;
  eseed = _src.eseed;
  hOverE = _src.hOverE;
  regPt = _src.regPt;
  smearedPt = _src.smearedPt;
  originalPt = _src.originalPt;
  dxy = _src.dxy;
  dz = _src.dz;
  veto = _src.veto;
  std::memcpy(triggerMatch, _src.triggerMatch, sizeof(Bool_t) * nTriggerObjects);
  superCluster = _src.superCluster;
}

panda::Electron::Electron(datastore& _data, UInt_t _idx) :
  Lepton(_data, _idx),
  hltsafe(_data.hltsafe[_idx]),
  chIsoPh(_data.chIsoPh[_idx]),
  nhIsoPh(_data.nhIsoPh[_idx]),
  phIsoPh(_data.phIsoPh[_idx]),
  ecalIso(_data.ecalIso[_idx]),
  hcalIso(_data.hcalIso[_idx]),
  isoPUOffset(_data.isoPUOffset[_idx]),
  sieie(_data.sieie[_idx]),
  sipip(_data.sipip[_idx]),
  eseed(_data.eseed[_idx]),
  hOverE(_data.hOverE[_idx]),
  regPt(_data.regPt[_idx]),
  smearedPt(_data.smearedPt[_idx]),
  originalPt(_data.originalPt[_idx]),
  dxy(_data.dxy[_idx]),
  dz(_data.dz[_idx]),
  veto(_data.veto[_idx]),
  triggerMatch(_data.triggerMatch[_idx]),
  superCluster(_data.superClusterContainer_, _data.superCluster_[_idx])
{
}

panda::Electron::Electron(ArrayBase* _array) :
  Lepton(_array),
  hltsafe(gStore.getData(this).hltsafe[0]),
  chIsoPh(gStore.getData(this).chIsoPh[0]),
  nhIsoPh(gStore.getData(this).nhIsoPh[0]),
  phIsoPh(gStore.getData(this).phIsoPh[0]),
  ecalIso(gStore.getData(this).ecalIso[0]),
  hcalIso(gStore.getData(this).hcalIso[0]),
  isoPUOffset(gStore.getData(this).isoPUOffset[0]),
  sieie(gStore.getData(this).sieie[0]),
  sipip(gStore.getData(this).sipip[0]),
  eseed(gStore.getData(this).eseed[0]),
  hOverE(gStore.getData(this).hOverE[0]),
  regPt(gStore.getData(this).regPt[0]),
  smearedPt(gStore.getData(this).smearedPt[0]),
  originalPt(gStore.getData(this).originalPt[0]),
  dxy(gStore.getData(this).dxy[0]),
  dz(gStore.getData(this).dz[0]),
  veto(gStore.getData(this).veto[0]),
  triggerMatch(gStore.getData(this).triggerMatch[0]),
  superCluster(gStore.getData(this).superClusterContainer_, gStore.getData(this).superCluster_[0])
{
}

panda::Electron::~Electron()
{
  destructor();
}

void
panda::Electron::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM Electron.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    Lepton::destructor(kTRUE);
}

panda::Electron&
panda::Electron::operator=(Electron const& _src)
{
  Lepton::operator=(_src);

  hltsafe = _src.hltsafe;
  chIsoPh = _src.chIsoPh;
  nhIsoPh = _src.nhIsoPh;
  phIsoPh = _src.phIsoPh;
  ecalIso = _src.ecalIso;
  hcalIso = _src.hcalIso;
  isoPUOffset = _src.isoPUOffset;
  sieie = _src.sieie;
  sipip = _src.sipip;
  eseed = _src.eseed;
  hOverE = _src.hOverE;
  regPt = _src.regPt;
  smearedPt = _src.smearedPt;
  originalPt = _src.originalPt;
  dxy = _src.dxy;
  dz = _src.dz;
  veto = _src.veto;
  std::memcpy(triggerMatch, _src.triggerMatch, sizeof(Bool_t) * nTriggerObjects);
  superCluster = _src.superCluster;

  /* BEGIN CUSTOM Electron.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::Electron::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  Lepton::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "hltsafe", "", 'O', &hltsafe, _branches);
  utils::book(_tree, _name, "chIsoPh", "", 'F', &chIsoPh, _branches);
  utils::book(_tree, _name, "nhIsoPh", "", 'F', &nhIsoPh, _branches);
  utils::book(_tree, _name, "phIsoPh", "", 'F', &phIsoPh, _branches);
  utils::book(_tree, _name, "ecalIso", "", 'F', &ecalIso, _branches);
  utils::book(_tree, _name, "hcalIso", "", 'F', &hcalIso, _branches);
  utils::book(_tree, _name, "isoPUOffset", "", 'F', &isoPUOffset, _branches);
  utils::book(_tree, _name, "sieie", "", 'F', &sieie, _branches);
  utils::book(_tree, _name, "sipip", "", 'F', &sipip, _branches);
  utils::book(_tree, _name, "eseed", "", 'F', &eseed, _branches);
  utils::book(_tree, _name, "hOverE", "", 'F', &hOverE, _branches);
  utils::book(_tree, _name, "regPt", "", 'F', &regPt, _branches);
  utils::book(_tree, _name, "smearedPt", "", 'F', &smearedPt, _branches);
  utils::book(_tree, _name, "originalPt", "", 'F', &originalPt, _branches);
  utils::book(_tree, _name, "dxy", "", 'F', &dxy, _branches);
  utils::book(_tree, _name, "dz", "", 'F', &dz, _branches);
  utils::book(_tree, _name, "veto", "", 'O', &veto, _branches);
  utils::book(_tree, _name, "triggerMatch", TString::Format("[%d]", nTriggerObjects), 'O', triggerMatch, _branches);
  utils::book(_tree, _name, "superCluster_", "", 'S', gStore.getData(this).superCluster_, _branches);
}

void
panda::Electron::doInit_()
{
  Lepton::doInit_();

  hltsafe = false;
  chIsoPh = 0.;
  nhIsoPh = 0.;
  phIsoPh = 0.;
  ecalIso = 0.;
  hcalIso = 0.;
  isoPUOffset = 0.;
  sieie = 0.;
  sipip = 0.;
  eseed = 0.;
  hOverE = 0.;
  regPt = 0.;
  smearedPt = 0.;
  originalPt = -1.;
  dxy = 0.;
  dz = 0.;
  veto = false;
  for (auto& p0 : triggerMatch) p0 = false;
  superCluster.init();

  /* BEGIN CUSTOM Electron.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::Electron::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM Electron.cc.print */
  if (_level >= 3) {
    Electron::dump(_out);
    
    _out << "combIso = " << combIso() << std::endl;
    _out << "combRelIso = " << combIso() / pt() << std::endl;
  }
  else if (_level == 2) {
    Lepton::print(_out, _level);
    
    _out << "smearedPt = " << smearedPt << std::endl;
    _out << "regPt = " << regPt << std::endl;
    _out << "hltsafe = " << hltsafe << std::endl;
    _out << "veto = " << veto << std::endl;

    _out << "combIso = " << combIso() << std::endl;
    _out << "combRelIso = " << combIso() / pt() << std::endl;
  }
  else
    return;
  /* END CUSTOM */
}

void
panda::Electron::dump(std::ostream& _out/* = std::cout*/) const
{
  Lepton::dump(_out);

  _out << "hltsafe = " << hltsafe << std::endl;
  _out << "chIsoPh = " << chIsoPh << std::endl;
  _out << "nhIsoPh = " << nhIsoPh << std::endl;
  _out << "phIsoPh = " << phIsoPh << std::endl;
  _out << "ecalIso = " << ecalIso << std::endl;
  _out << "hcalIso = " << hcalIso << std::endl;
  _out << "isoPUOffset = " << isoPUOffset << std::endl;
  _out << "sieie = " << sieie << std::endl;
  _out << "sipip = " << sipip << std::endl;
  _out << "eseed = " << eseed << std::endl;
  _out << "hOverE = " << hOverE << std::endl;
  _out << "regPt = " << regPt << std::endl;
  _out << "smearedPt = " << smearedPt << std::endl;
  _out << "originalPt = " << originalPt << std::endl;
  _out << "dxy = " << dxy << std::endl;
  _out << "dz = " << dz << std::endl;
  _out << "veto = " << veto << std::endl;
  _out << "triggerMatch = " << triggerMatch << std::endl;
  _out << "superCluster = " << superCluster << std::endl;
}


/* BEGIN CUSTOM Electron.cc.global */
/* END CUSTOM */
