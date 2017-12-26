#include "../interface/GenJet.h"

/*static*/
panda::utils::BranchList
panda::GenJet::getListOfBranches()
{
  utils::BranchList blist;
  blist += ParticleM::getListOfBranches();
  blist += {"pdgid"};
  return blist;
}

void
panda::GenJet::datastore::allocate(UInt_t _nmax)
{
  ParticleM::datastore::allocate(_nmax);

  pdgid = new Short_t[nmax_];
}

void
panda::GenJet::datastore::deallocate()
{
  ParticleM::datastore::deallocate();

  delete [] pdgid;
  pdgid = 0;
}

void
panda::GenJet::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  ParticleM::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "pdgid", _branches);
}

panda::utils::BranchList
panda::GenJet::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(ParticleM::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "pdgid"));

  return blist;
}

void
panda::GenJet::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  ParticleM::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "pdgid", pdgid, _branches, _setStatus);
}

void
panda::GenJet::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  ParticleM::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "pdgid", size, 'S', pdgid, _branches);
}

void
panda::GenJet::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  ParticleM::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "pdgid");
}

void
panda::GenJet::datastore::resizeVectors_(UInt_t _size)
{
  ParticleM::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::GenJet::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return GenJet::getListOfBranches().fullNames(_name);
}

panda::GenJet::GenJet(char const* _name/* = ""*/) :
  ParticleM(new GenJetArray(1, _name)),
  pdgid(gStore.getData(this).pdgid[0])
{
}

panda::GenJet::GenJet(GenJet const& _src) :
  ParticleM(new GenJetArray(1, _src.getName())),
  pdgid(gStore.getData(this).pdgid[0])
{
  ParticleM::operator=(_src);

  pdgid = _src.pdgid;
}

panda::GenJet::GenJet(datastore& _data, UInt_t _idx) :
  ParticleM(_data, _idx),
  pdgid(_data.pdgid[_idx])
{
}

panda::GenJet::GenJet(ArrayBase* _array) :
  ParticleM(_array),
  pdgid(gStore.getData(this).pdgid[0])
{
}

panda::GenJet::~GenJet()
{
  destructor();
}

void
panda::GenJet::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM GenJet.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    ParticleM::destructor(kTRUE);
}

panda::GenJet&
panda::GenJet::operator=(GenJet const& _src)
{
  ParticleM::operator=(_src);

  pdgid = _src.pdgid;

  /* BEGIN CUSTOM GenJet.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::GenJet::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  ParticleM::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "pdgid", "", 'S', &pdgid, _branches);
}

void
panda::GenJet::doInit_()
{
  ParticleM::doInit_();

  pdgid = 0;

  /* BEGIN CUSTOM GenJet.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::GenJet::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM GenJet.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::GenJet::dump(std::ostream& _out/* = std::cout*/) const
{
  ParticleM::dump(_out);

  _out << "pdgid = " << pdgid << std::endl;
}

/* BEGIN CUSTOM GenJet.cc.global */
/* END CUSTOM */
