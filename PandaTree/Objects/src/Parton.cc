#include "../interface/Parton.h"

/*static*/
panda::utils::BranchList
panda::Parton::getListOfBranches()
{
  utils::BranchList blist;
  blist += ParticleM::getListOfBranches();
  blist += {"pdgid"};
  return blist;
}

void
panda::Parton::datastore::allocate(UInt_t _nmax)
{
  ParticleM::datastore::allocate(_nmax);

  pdgid = new Int_t[nmax_];
}

void
panda::Parton::datastore::deallocate()
{
  ParticleM::datastore::deallocate();

  delete [] pdgid;
  pdgid = 0;
}

void
panda::Parton::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  ParticleM::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "pdgid", _branches);
}

panda::utils::BranchList
panda::Parton::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(ParticleM::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "pdgid"));

  return blist;
}

void
panda::Parton::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  ParticleM::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "pdgid", pdgid, _branches, _setStatus);
}

void
panda::Parton::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  ParticleM::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "pdgid", size, 'I', pdgid, _branches);
}

void
panda::Parton::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  ParticleM::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "pdgid");
}

void
panda::Parton::datastore::resizeVectors_(UInt_t _size)
{
  ParticleM::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::Parton::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return Parton::getListOfBranches().fullNames(_name);
}

panda::Parton::Parton(char const* _name/* = ""*/) :
  ParticleM(new PartonArray(1, _name)),
  pdgid(gStore.getData(this).pdgid[0])
{
}

panda::Parton::Parton(Parton const& _src) :
  ParticleM(new PartonArray(1, _src.getName())),
  pdgid(gStore.getData(this).pdgid[0])
{
  ParticleM::operator=(_src);

  pdgid = _src.pdgid;
}

panda::Parton::Parton(datastore& _data, UInt_t _idx) :
  ParticleM(_data, _idx),
  pdgid(_data.pdgid[_idx])
{
}

panda::Parton::Parton(ArrayBase* _array) :
  ParticleM(_array),
  pdgid(gStore.getData(this).pdgid[0])
{
}

panda::Parton::~Parton()
{
  destructor();
}

void
panda::Parton::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM Parton.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    ParticleM::destructor(kTRUE);
}

panda::Parton&
panda::Parton::operator=(Parton const& _src)
{
  ParticleM::operator=(_src);

  pdgid = _src.pdgid;

  /* BEGIN CUSTOM Parton.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::Parton::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  ParticleM::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "pdgid", "", 'I', &pdgid, _branches);
}

void
panda::Parton::doInit_()
{
  ParticleM::doInit_();

  pdgid = 0;

  /* BEGIN CUSTOM Parton.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::Parton::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM Parton.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::Parton::dump(std::ostream& _out/* = std::cout*/) const
{
  ParticleM::dump(_out);

  _out << "pdgid = " << pdgid << std::endl;
}

/* BEGIN CUSTOM Parton.cc.global */
/* END CUSTOM */
