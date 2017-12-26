#include "../interface/Muon.h"

TString panda::Muon::TriggerObjectName[] = {
  "fMu17Mu8FirstLeg",
  "fMu17Mu8SecondLeg",
  "fIsoMu22er",
  "fIsoTkMu22er",
  "fIsoMu24",
  "fIsoTkMu24"
};

/*static*/
panda::utils::BranchList
panda::Muon::getListOfBranches()
{
  utils::BranchList blist;
  blist += Lepton::getListOfBranches();
  blist += {"mediumBtoF", "triggerMatch"};
  return blist;
}

void
panda::Muon::datastore::allocate(UInt_t _nmax)
{
  Lepton::datastore::allocate(_nmax);

  mediumBtoF = new Bool_t[nmax_];
  triggerMatch = new Bool_t[nmax_][nTriggerObjects];
}

void
panda::Muon::datastore::deallocate()
{
  Lepton::datastore::deallocate();

  delete [] mediumBtoF;
  mediumBtoF = 0;
  delete [] triggerMatch;
  triggerMatch = 0;
}

void
panda::Muon::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  Lepton::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "mediumBtoF", _branches);
  utils::setStatus(_tree, _name, "triggerMatch", _branches);
}

panda::utils::BranchList
panda::Muon::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(Lepton::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "mediumBtoF"));
  blist.push_back(utils::getStatus(_tree, _name, "triggerMatch"));

  return blist;
}

void
panda::Muon::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  Lepton::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "mediumBtoF", mediumBtoF, _branches, _setStatus);
  utils::setAddress(_tree, _name, "triggerMatch", triggerMatch, _branches, _setStatus);
}

void
panda::Muon::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  Lepton::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "mediumBtoF", size, 'O', mediumBtoF, _branches);
  utils::book(_tree, _name, "triggerMatch", size + TString::Format("[%d]", nTriggerObjects), 'O', triggerMatch, _branches);
}

void
panda::Muon::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  Lepton::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "mediumBtoF");
  utils::resetAddress(_tree, _name, "triggerMatch");
}

void
panda::Muon::datastore::resizeVectors_(UInt_t _size)
{
  Lepton::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::Muon::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return Muon::getListOfBranches().fullNames(_name);
}

panda::Muon::Muon(char const* _name/* = ""*/) :
  Lepton(new MuonArray(1, _name)),
  mediumBtoF(gStore.getData(this).mediumBtoF[0]),
  triggerMatch(gStore.getData(this).triggerMatch[0])
{
}

panda::Muon::Muon(Muon const& _src) :
  Lepton(new MuonArray(1, _src.getName())),
  mediumBtoF(gStore.getData(this).mediumBtoF[0]),
  triggerMatch(gStore.getData(this).triggerMatch[0])
{
  Lepton::operator=(_src);

  mediumBtoF = _src.mediumBtoF;
  std::memcpy(triggerMatch, _src.triggerMatch, sizeof(Bool_t) * nTriggerObjects);
}

panda::Muon::Muon(datastore& _data, UInt_t _idx) :
  Lepton(_data, _idx),
  mediumBtoF(_data.mediumBtoF[_idx]),
  triggerMatch(_data.triggerMatch[_idx])
{
}

panda::Muon::Muon(ArrayBase* _array) :
  Lepton(_array),
  mediumBtoF(gStore.getData(this).mediumBtoF[0]),
  triggerMatch(gStore.getData(this).triggerMatch[0])
{
}

panda::Muon::~Muon()
{
  destructor();
}

void
panda::Muon::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM Muon.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    Lepton::destructor(kTRUE);
}

panda::Muon&
panda::Muon::operator=(Muon const& _src)
{
  Lepton::operator=(_src);

  mediumBtoF = _src.mediumBtoF;
  std::memcpy(triggerMatch, _src.triggerMatch, sizeof(Bool_t) * nTriggerObjects);

  /* BEGIN CUSTOM Muon.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::Muon::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  Lepton::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "mediumBtoF", "", 'O', &mediumBtoF, _branches);
  utils::book(_tree, _name, "triggerMatch", TString::Format("[%d]", nTriggerObjects), 'O', triggerMatch, _branches);
}

void
panda::Muon::doInit_()
{
  Lepton::doInit_();

  mediumBtoF = false;
  for (auto& p0 : triggerMatch) p0 = false;

  /* BEGIN CUSTOM Muon.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::Muon::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM Muon.cc.print */
  if (_level >= 3) {
    Muon::dump(_out);
    
    _out << "combIso = " << combIso() << std::endl;
    _out << "combRelIso = " << combIso() / pt() << std::endl;
  }
  else if (_level == 2) {
    Lepton::print(_out, _level);
    
    _out << "combIso = " << combIso() << std::endl;
    _out << "combRelIso = " << combIso() / pt() << std::endl;
  }
  else
    return;
  /* END CUSTOM */
}

void
panda::Muon::dump(std::ostream& _out/* = std::cout*/) const
{
  Lepton::dump(_out);

  _out << "mediumBtoF = " << mediumBtoF << std::endl;
  _out << "triggerMatch = " << triggerMatch << std::endl;
}


/* BEGIN CUSTOM Muon.cc.global */
/* END CUSTOM */
