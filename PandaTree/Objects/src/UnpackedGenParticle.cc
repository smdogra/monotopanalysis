#include "../interface/UnpackedGenParticle.h"

/*static*/
panda::utils::BranchList
panda::UnpackedGenParticle::getListOfBranches()
{
  utils::BranchList blist;
  blist += ParticleM::getListOfBranches();
  blist += {"pdgid", "finalState", "statusFlags", "parent_"};
  return blist;
}

void
panda::UnpackedGenParticle::datastore::allocate(UInt_t _nmax)
{
  ParticleM::datastore::allocate(_nmax);

  pdgid = new Int_t[nmax_];
  finalState = new Bool_t[nmax_];
  statusFlags = new UShort_t[nmax_];
  parent_ = new Short_t[nmax_];
}

void
panda::UnpackedGenParticle::datastore::deallocate()
{
  ParticleM::datastore::deallocate();

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
panda::UnpackedGenParticle::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  ParticleM::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "pdgid", _branches);
  utils::setStatus(_tree, _name, "finalState", _branches);
  utils::setStatus(_tree, _name, "statusFlags", _branches);
  utils::setStatus(_tree, _name, "parent_", _branches);
}

panda::utils::BranchList
panda::UnpackedGenParticle::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(ParticleM::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "pdgid"));
  blist.push_back(utils::getStatus(_tree, _name, "finalState"));
  blist.push_back(utils::getStatus(_tree, _name, "statusFlags"));
  blist.push_back(utils::getStatus(_tree, _name, "parent_"));

  return blist;
}

void
panda::UnpackedGenParticle::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  ParticleM::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "pdgid", pdgid, _branches, _setStatus);
  utils::setAddress(_tree, _name, "finalState", finalState, _branches, _setStatus);
  utils::setAddress(_tree, _name, "statusFlags", statusFlags, _branches, _setStatus);
  utils::setAddress(_tree, _name, "parent_", parent_, _branches, _setStatus);
}

void
panda::UnpackedGenParticle::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  ParticleM::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "pdgid", size, 'I', pdgid, _branches);
  utils::book(_tree, _name, "finalState", size, 'O', finalState, _branches);
  utils::book(_tree, _name, "statusFlags", size, 's', statusFlags, _branches);
  utils::book(_tree, _name, "parent_", size, 'S', parent_, _branches);
}

void
panda::UnpackedGenParticle::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  ParticleM::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "pdgid");
  utils::resetAddress(_tree, _name, "finalState");
  utils::resetAddress(_tree, _name, "statusFlags");
  utils::resetAddress(_tree, _name, "parent_");
}

void
panda::UnpackedGenParticle::datastore::resizeVectors_(UInt_t _size)
{
  ParticleM::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::UnpackedGenParticle::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return UnpackedGenParticle::getListOfBranches().fullNames(_name);
}

panda::UnpackedGenParticle::UnpackedGenParticle(char const* _name/* = ""*/) :
  ParticleM(new UnpackedGenParticleArray(1, _name)),
  pdgid(gStore.getData(this).pdgid[0]),
  finalState(gStore.getData(this).finalState[0]),
  statusFlags(gStore.getData(this).statusFlags[0]),
  parent(gStore.getData(this).parentContainer_, gStore.getData(this).parent_[0])
{
}

panda::UnpackedGenParticle::UnpackedGenParticle(UnpackedGenParticle const& _src) :
  ParticleM(new UnpackedGenParticleArray(1, _src.getName())),
  pdgid(gStore.getData(this).pdgid[0]),
  finalState(gStore.getData(this).finalState[0]),
  statusFlags(gStore.getData(this).statusFlags[0]),
  parent(gStore.getData(this).parentContainer_, gStore.getData(this).parent_[0])
{
  ParticleM::operator=(_src);

  pdgid = _src.pdgid;
  finalState = _src.finalState;
  statusFlags = _src.statusFlags;
  parent = _src.parent;
}

panda::UnpackedGenParticle::UnpackedGenParticle(datastore& _data, UInt_t _idx) :
  ParticleM(_data, _idx),
  pdgid(_data.pdgid[_idx]),
  finalState(_data.finalState[_idx]),
  statusFlags(_data.statusFlags[_idx]),
  parent(_data.parentContainer_, _data.parent_[_idx])
{
}

panda::UnpackedGenParticle::UnpackedGenParticle(ArrayBase* _array) :
  ParticleM(_array),
  pdgid(gStore.getData(this).pdgid[0]),
  finalState(gStore.getData(this).finalState[0]),
  statusFlags(gStore.getData(this).statusFlags[0]),
  parent(gStore.getData(this).parentContainer_, gStore.getData(this).parent_[0])
{
}

panda::UnpackedGenParticle::~UnpackedGenParticle()
{
  destructor();
}

void
panda::UnpackedGenParticle::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM UnpackedGenParticle.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    ParticleM::destructor(kTRUE);
}

panda::UnpackedGenParticle&
panda::UnpackedGenParticle::operator=(UnpackedGenParticle const& _src)
{
  ParticleM::operator=(_src);

  pdgid = _src.pdgid;
  finalState = _src.finalState;
  statusFlags = _src.statusFlags;
  parent = _src.parent;

  /* BEGIN CUSTOM UnpackedGenParticle.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::UnpackedGenParticle::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  ParticleM::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "pdgid", "", 'I', &pdgid, _branches);
  utils::book(_tree, _name, "finalState", "", 'O', &finalState, _branches);
  utils::book(_tree, _name, "statusFlags", "", 's', &statusFlags, _branches);
  utils::book(_tree, _name, "parent_", "", 'S', gStore.getData(this).parent_, _branches);
}

void
panda::UnpackedGenParticle::doInit_()
{
  ParticleM::doInit_();

  pdgid = 0;
  finalState = false;
  statusFlags = 0;
  parent.init();

  /* BEGIN CUSTOM UnpackedGenParticle.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::UnpackedGenParticle::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM UnpackedGenParticle.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::UnpackedGenParticle::dump(std::ostream& _out/* = std::cout*/) const
{
  ParticleM::dump(_out);

  _out << "pdgid = " << pdgid << std::endl;
  _out << "finalState = " << finalState << std::endl;
  _out << "statusFlags = " << statusFlags << std::endl;
  _out << "parent = " << parent << std::endl;
}


/* BEGIN CUSTOM UnpackedGenParticle.cc.global */
panda::UnpackedGenParticle&
panda::UnpackedGenParticle::operator=(GenParticle const& _rhs)
{
  setPtEtaPhiM(_rhs.pt(), _rhs.eta(), _rhs.phi(), _rhs.m());
  pdgid = _rhs.pdgid;
  finalState = _rhs.finalState;
  statusFlags = _rhs.statusFlags;

  return *this;
}

/* END CUSTOM */
