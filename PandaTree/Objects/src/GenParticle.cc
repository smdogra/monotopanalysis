#include "../interface/GenParticle.h"

TString panda::GenParticle::StatusFlagName[] = {
  "kIsPrompt",
  "kIsDecayedLeptonHadron",
  "kIsTauDecayProduct",
  "kIsPromptTauDecayProduct",
  "kIsDirectTauDecayProduct",
  "kIsDirectPromptTauDecayProduct",
  "kIsDirectHadronDecayProduct",
  "kIsHardProcess",
  "kFromHardProcess",
  "kIsHardProcessTauDecayProduct",
  "kIsDirectHardProcessTauDecayProduct",
  "kFromHardProcessBeforeFSR",
  "kIsFirstCopy",
  "kIsLastCopy",
  "kIsLastCopyBeforeFSR"
};

/*static*/
panda::utils::BranchList
panda::GenParticle::getListOfBranches()
{
  utils::BranchList blist;
  blist += PackedParticle::getListOfBranches();
  blist += {"pdgid", "finalState", "statusFlags", "parent_"};
  return blist;
}

void
panda::GenParticle::datastore::allocate(UInt_t _nmax)
{
  PackedParticle::datastore::allocate(_nmax);

  pdgid = new Int_t[nmax_];
  finalState = new Bool_t[nmax_];
  statusFlags = new UShort_t[nmax_];
  parent_ = new Short_t[nmax_];
}

void
panda::GenParticle::datastore::deallocate()
{
  PackedParticle::datastore::deallocate();

  delete [] pdgid;
  pdgid = 0;
  delete [] finalState;
  finalState = 0;
  delete [] statusFlags;
  statusFlags = 0;
  delete [] parent_;
  parent_ = 0;
}

void
panda::GenParticle::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  PackedParticle::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "pdgid", _branches);
  utils::setStatus(_tree, _name, "finalState", _branches);
  utils::setStatus(_tree, _name, "statusFlags", _branches);
  utils::setStatus(_tree, _name, "parent_", _branches);
}

panda::utils::BranchList
panda::GenParticle::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(PackedParticle::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "pdgid"));
  blist.push_back(utils::getStatus(_tree, _name, "finalState"));
  blist.push_back(utils::getStatus(_tree, _name, "statusFlags"));
  blist.push_back(utils::getStatus(_tree, _name, "parent_"));

  return blist;
}

void
panda::GenParticle::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  PackedParticle::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "pdgid", pdgid, _branches, _setStatus);
  utils::setAddress(_tree, _name, "finalState", finalState, _branches, _setStatus);
  utils::setAddress(_tree, _name, "statusFlags", statusFlags, _branches, _setStatus);
  utils::setAddress(_tree, _name, "parent_", parent_, _branches, _setStatus);
}

void
panda::GenParticle::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  PackedParticle::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "pdgid", size, 'I', pdgid, _branches);
  utils::book(_tree, _name, "finalState", size, 'O', finalState, _branches);
  utils::book(_tree, _name, "statusFlags", size, 's', statusFlags, _branches);
  utils::book(_tree, _name, "parent_", size, 'S', parent_, _branches);
}

void
panda::GenParticle::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  PackedParticle::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "pdgid");
  utils::resetAddress(_tree, _name, "finalState");
  utils::resetAddress(_tree, _name, "statusFlags");
  utils::resetAddress(_tree, _name, "parent_");
}

void
panda::GenParticle::datastore::resizeVectors_(UInt_t _size)
{
  PackedParticle::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::GenParticle::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return GenParticle::getListOfBranches().fullNames(_name);
}

panda::GenParticle::GenParticle(char const* _name/* = ""*/) :
  PackedParticle(new GenParticleArray(1, _name)),
  pdgid(gStore.getData(this).pdgid[0]),
  finalState(gStore.getData(this).finalState[0]),
  statusFlags(gStore.getData(this).statusFlags[0]),
  parent(gStore.getData(this).parentContainer_, gStore.getData(this).parent_[0])
{
}

panda::GenParticle::GenParticle(GenParticle const& _src) :
  PackedParticle(new GenParticleArray(1, _src.getName())),
  pdgid(gStore.getData(this).pdgid[0]),
  finalState(gStore.getData(this).finalState[0]),
  statusFlags(gStore.getData(this).statusFlags[0]),
  parent(gStore.getData(this).parentContainer_, gStore.getData(this).parent_[0])
{
  PackedParticle::operator=(_src);

  pdgid = _src.pdgid;
  finalState = _src.finalState;
  statusFlags = _src.statusFlags;
  parent = _src.parent;
}

panda::GenParticle::GenParticle(datastore& _data, UInt_t _idx) :
  PackedParticle(_data, _idx),
  pdgid(_data.pdgid[_idx]),
  finalState(_data.finalState[_idx]),
  statusFlags(_data.statusFlags[_idx]),
  parent(_data.parentContainer_, _data.parent_[_idx])
{
}

panda::GenParticle::GenParticle(ArrayBase* _array) :
  PackedParticle(_array),
  pdgid(gStore.getData(this).pdgid[0]),
  finalState(gStore.getData(this).finalState[0]),
  statusFlags(gStore.getData(this).statusFlags[0]),
  parent(gStore.getData(this).parentContainer_, gStore.getData(this).parent_[0])
{
}

panda::GenParticle::~GenParticle()
{
  destructor();
}

void
panda::GenParticle::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM GenParticle.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    PackedParticle::destructor(kTRUE);
}

panda::GenParticle&
panda::GenParticle::operator=(GenParticle const& _src)
{
  PackedParticle::operator=(_src);

  pdgid = _src.pdgid;
  finalState = _src.finalState;
  statusFlags = _src.statusFlags;
  parent = _src.parent;

  /* BEGIN CUSTOM GenParticle.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::GenParticle::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  PackedParticle::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "pdgid", "", 'I', &pdgid, _branches);
  utils::book(_tree, _name, "finalState", "", 'O', &finalState, _branches);
  utils::book(_tree, _name, "statusFlags", "", 's', &statusFlags, _branches);
  utils::book(_tree, _name, "parent_", "", 'S', gStore.getData(this).parent_, _branches);
}

void
panda::GenParticle::doInit_()
{
  PackedParticle::doInit_();

  pdgid = 0;
  finalState = false;
  statusFlags = 0;
  parent.init();

  /* BEGIN CUSTOM GenParticle.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::GenParticle::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM GenParticle.cc.print */
  _out << "GenParticle " << pdgid << " (" << pt() << ", " << eta() << ", " << phi() << ") "
       << "flag=";
  for (unsigned i(0); i != 16; ++i)
    _out << ((statusFlags >> i) & 1);
  _out << std::endl;
  /* END CUSTOM */
}

void
panda::GenParticle::dump(std::ostream& _out/* = std::cout*/) const
{
  PackedParticle::dump(_out);

  _out << "pdgid = " << pdgid << std::endl;
  _out << "finalState = " << finalState << std::endl;
  _out << "statusFlags = " << statusFlags << std::endl;
  _out << "parent = " << parent << std::endl;
}


/* BEGIN CUSTOM GenParticle.cc.global */
/* END CUSTOM */
