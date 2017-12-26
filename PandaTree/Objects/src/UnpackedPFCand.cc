#include "../interface/UnpackedPFCand.h"

/*static*/
panda::utils::BranchList
panda::UnpackedPFCand::getListOfBranches()
{
  utils::BranchList blist;
  blist += ParticleM::getListOfBranches();
  blist += {"puppiW", "puppiWNoLep", "ptype", "vertex_"};
  return blist;
}

void
panda::UnpackedPFCand::datastore::allocate(UInt_t _nmax)
{
  ParticleM::datastore::allocate(_nmax);

  puppiW = new Char_t[nmax_];
  puppiWNoLep = new Char_t[nmax_];
  ptype = new UChar_t[nmax_];
  vertex_ = new Short_t[nmax_];
}

void
panda::UnpackedPFCand::datastore::deallocate()
{
  ParticleM::datastore::deallocate();

  delete [] puppiW;
  puppiW = 0;
  delete [] puppiWNoLep;
  puppiWNoLep = 0;
  delete [] ptype;
  ptype = 0;
  delete [] vertex_;
  vertex_ = 0;
}

void
panda::UnpackedPFCand::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  ParticleM::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "puppiW", _branches);
  utils::setStatus(_tree, _name, "puppiWNoLep", _branches);
  utils::setStatus(_tree, _name, "ptype", _branches);
  utils::setStatus(_tree, _name, "vertex_", _branches);
}

panda::utils::BranchList
panda::UnpackedPFCand::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(ParticleM::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "puppiW"));
  blist.push_back(utils::getStatus(_tree, _name, "puppiWNoLep"));
  blist.push_back(utils::getStatus(_tree, _name, "ptype"));
  blist.push_back(utils::getStatus(_tree, _name, "vertex_"));

  return blist;
}

void
panda::UnpackedPFCand::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  ParticleM::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "puppiW", puppiW, _branches, _setStatus);
  utils::setAddress(_tree, _name, "puppiWNoLep", puppiWNoLep, _branches, _setStatus);
  utils::setAddress(_tree, _name, "ptype", ptype, _branches, _setStatus);
  utils::setAddress(_tree, _name, "vertex_", vertex_, _branches, _setStatus);
}

void
panda::UnpackedPFCand::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  ParticleM::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "puppiW", size, 'B', puppiW, _branches);
  utils::book(_tree, _name, "puppiWNoLep", size, 'B', puppiWNoLep, _branches);
  utils::book(_tree, _name, "ptype", size, 'b', ptype, _branches);
  utils::book(_tree, _name, "vertex_", size, 'S', vertex_, _branches);
}

void
panda::UnpackedPFCand::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  ParticleM::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "puppiW");
  utils::resetAddress(_tree, _name, "puppiWNoLep");
  utils::resetAddress(_tree, _name, "ptype");
  utils::resetAddress(_tree, _name, "vertex_");
}

void
panda::UnpackedPFCand::datastore::resizeVectors_(UInt_t _size)
{
  ParticleM::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::UnpackedPFCand::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return UnpackedPFCand::getListOfBranches().fullNames(_name);
}

panda::UnpackedPFCand::UnpackedPFCand(char const* _name/* = ""*/) :
  ParticleM(new UnpackedPFCandArray(1, _name)),
  puppiW(gStore.getData(this).puppiW[0]),
  puppiWNoLep(gStore.getData(this).puppiWNoLep[0]),
  ptype(gStore.getData(this).ptype[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0])
{
}

panda::UnpackedPFCand::UnpackedPFCand(UnpackedPFCand const& _src) :
  ParticleM(new UnpackedPFCandArray(1, _src.getName())),
  puppiW(gStore.getData(this).puppiW[0]),
  puppiWNoLep(gStore.getData(this).puppiWNoLep[0]),
  ptype(gStore.getData(this).ptype[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0])
{
  ParticleM::operator=(_src);

  puppiW = _src.puppiW;
  puppiWNoLep = _src.puppiWNoLep;
  ptype = _src.ptype;
  vertex = _src.vertex;
}

panda::UnpackedPFCand::UnpackedPFCand(datastore& _data, UInt_t _idx) :
  ParticleM(_data, _idx),
  puppiW(_data.puppiW[_idx]),
  puppiWNoLep(_data.puppiWNoLep[_idx]),
  ptype(_data.ptype[_idx]),
  vertex(_data.vertexContainer_, _data.vertex_[_idx])
{
}

panda::UnpackedPFCand::UnpackedPFCand(ArrayBase* _array) :
  ParticleM(_array),
  puppiW(gStore.getData(this).puppiW[0]),
  puppiWNoLep(gStore.getData(this).puppiWNoLep[0]),
  ptype(gStore.getData(this).ptype[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0])
{
}

panda::UnpackedPFCand::~UnpackedPFCand()
{
  destructor();
}

void
panda::UnpackedPFCand::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM UnpackedPFCand.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    ParticleM::destructor(kTRUE);
}

panda::UnpackedPFCand&
panda::UnpackedPFCand::operator=(UnpackedPFCand const& _src)
{
  ParticleM::operator=(_src);

  puppiW = _src.puppiW;
  puppiWNoLep = _src.puppiWNoLep;
  ptype = _src.ptype;
  vertex = _src.vertex;

  /* BEGIN CUSTOM UnpackedPFCand.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::UnpackedPFCand::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  ParticleM::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "puppiW", "", 'B', &puppiW, _branches);
  utils::book(_tree, _name, "puppiWNoLep", "", 'B', &puppiWNoLep, _branches);
  utils::book(_tree, _name, "ptype", "", 'b', &ptype, _branches);
  utils::book(_tree, _name, "vertex_", "", 'S', gStore.getData(this).vertex_, _branches);
}

void
panda::UnpackedPFCand::doInit_()
{
  ParticleM::doInit_();

  puppiW = 0;
  puppiWNoLep = 0;
  ptype = 0;
  vertex.init();

  /* BEGIN CUSTOM UnpackedPFCand.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::UnpackedPFCand::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM UnpackedPFCand.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::UnpackedPFCand::dump(std::ostream& _out/* = std::cout*/) const
{
  ParticleM::dump(_out);

  _out << "puppiW = " << puppiW << std::endl;
  _out << "puppiWNoLep = " << puppiWNoLep << std::endl;
  _out << "ptype = " << ptype << std::endl;
  _out << "vertex = " << vertex << std::endl;
}


/* BEGIN CUSTOM UnpackedPFCand.cc.global */
panda::UnpackedPFCand&
panda::UnpackedPFCand::operator=(PFCand const& _rhs)
{
  setPtEtaPhiM(_rhs.pt(), _rhs.eta(), _rhs.phi(), _rhs.m());

  puppiW = _rhs.puppiW();
  puppiWNoLep = _rhs.puppiWNoLep();
  ptype = _rhs.ptype;
  vertex = _rhs.vertex;

  return *this;
}

/* END CUSTOM */
