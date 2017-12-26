#include "../interface/Tau.h"

/*static*/
panda::utils::BranchList
panda::Tau::getListOfBranches()
{
  utils::BranchList blist;
  blist += ParticleM::getListOfBranches();
  blist += {"charge", "decayMode", "decayModeNew", "looseIsoMVA", "looseIsoMVAOld", "iso", "isoDeltaBetaCorr", "vertex_", "matchedGen_"};
  return blist;
}

void
panda::Tau::datastore::allocate(UInt_t _nmax)
{
  ParticleM::datastore::allocate(_nmax);

  charge = new Char_t[nmax_];
  decayMode = new Bool_t[nmax_];
  decayModeNew = new Bool_t[nmax_];
  looseIsoMVA = new Bool_t[nmax_];
  looseIsoMVAOld = new Bool_t[nmax_];
  iso = new Float_t[nmax_];
  isoDeltaBetaCorr = new Float_t[nmax_];
  vertex_ = new Short_t[nmax_];
  matchedGen_ = new Short_t[nmax_];
}

void
panda::Tau::datastore::deallocate()
{
  ParticleM::datastore::deallocate();

  delete [] charge;
  charge = 0;
  delete [] decayMode;
  decayMode = 0;
  delete [] decayModeNew;
  decayModeNew = 0;
  delete [] looseIsoMVA;
  looseIsoMVA = 0;
  delete [] looseIsoMVAOld;
  looseIsoMVAOld = 0;
  delete [] iso;
  iso = 0;
  delete [] isoDeltaBetaCorr;
  isoDeltaBetaCorr = 0;
  delete [] vertex_;
  vertex_ = 0;
  delete [] matchedGen_;
  matchedGen_ = 0;
}

void
panda::Tau::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  ParticleM::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "charge", _branches);
  utils::setStatus(_tree, _name, "decayMode", _branches);
  utils::setStatus(_tree, _name, "decayModeNew", _branches);
  utils::setStatus(_tree, _name, "looseIsoMVA", _branches);
  utils::setStatus(_tree, _name, "looseIsoMVAOld", _branches);
  utils::setStatus(_tree, _name, "iso", _branches);
  utils::setStatus(_tree, _name, "isoDeltaBetaCorr", _branches);
  utils::setStatus(_tree, _name, "vertex_", _branches);
  utils::setStatus(_tree, _name, "matchedGen_", _branches);
}

panda::utils::BranchList
panda::Tau::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(ParticleM::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "charge"));
  blist.push_back(utils::getStatus(_tree, _name, "decayMode"));
  blist.push_back(utils::getStatus(_tree, _name, "decayModeNew"));
  blist.push_back(utils::getStatus(_tree, _name, "looseIsoMVA"));
  blist.push_back(utils::getStatus(_tree, _name, "looseIsoMVAOld"));
  blist.push_back(utils::getStatus(_tree, _name, "iso"));
  blist.push_back(utils::getStatus(_tree, _name, "isoDeltaBetaCorr"));
  blist.push_back(utils::getStatus(_tree, _name, "vertex_"));
  blist.push_back(utils::getStatus(_tree, _name, "matchedGen_"));

  return blist;
}

void
panda::Tau::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  ParticleM::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "charge", charge, _branches, _setStatus);
  utils::setAddress(_tree, _name, "decayMode", decayMode, _branches, _setStatus);
  utils::setAddress(_tree, _name, "decayModeNew", decayModeNew, _branches, _setStatus);
  utils::setAddress(_tree, _name, "looseIsoMVA", looseIsoMVA, _branches, _setStatus);
  utils::setAddress(_tree, _name, "looseIsoMVAOld", looseIsoMVAOld, _branches, _setStatus);
  utils::setAddress(_tree, _name, "iso", iso, _branches, _setStatus);
  utils::setAddress(_tree, _name, "isoDeltaBetaCorr", isoDeltaBetaCorr, _branches, _setStatus);
  utils::setAddress(_tree, _name, "vertex_", vertex_, _branches, _setStatus);
  utils::setAddress(_tree, _name, "matchedGen_", matchedGen_, _branches, _setStatus);
}

void
panda::Tau::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  ParticleM::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "charge", size, 'B', charge, _branches);
  utils::book(_tree, _name, "decayMode", size, 'O', decayMode, _branches);
  utils::book(_tree, _name, "decayModeNew", size, 'O', decayModeNew, _branches);
  utils::book(_tree, _name, "looseIsoMVA", size, 'O', looseIsoMVA, _branches);
  utils::book(_tree, _name, "looseIsoMVAOld", size, 'O', looseIsoMVAOld, _branches);
  utils::book(_tree, _name, "iso", size, 'F', iso, _branches);
  utils::book(_tree, _name, "isoDeltaBetaCorr", size, 'F', isoDeltaBetaCorr, _branches);
  utils::book(_tree, _name, "vertex_", size, 'S', vertex_, _branches);
  utils::book(_tree, _name, "matchedGen_", size, 'S', matchedGen_, _branches);
}

void
panda::Tau::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  ParticleM::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "charge");
  utils::resetAddress(_tree, _name, "decayMode");
  utils::resetAddress(_tree, _name, "decayModeNew");
  utils::resetAddress(_tree, _name, "looseIsoMVA");
  utils::resetAddress(_tree, _name, "looseIsoMVAOld");
  utils::resetAddress(_tree, _name, "iso");
  utils::resetAddress(_tree, _name, "isoDeltaBetaCorr");
  utils::resetAddress(_tree, _name, "vertex_");
  utils::resetAddress(_tree, _name, "matchedGen_");
}

void
panda::Tau::datastore::resizeVectors_(UInt_t _size)
{
  ParticleM::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::Tau::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return Tau::getListOfBranches().fullNames(_name);
}

panda::Tau::Tau(char const* _name/* = ""*/) :
  ParticleM(new TauArray(1, _name)),
  charge(gStore.getData(this).charge[0]),
  decayMode(gStore.getData(this).decayMode[0]),
  decayModeNew(gStore.getData(this).decayModeNew[0]),
  looseIsoMVA(gStore.getData(this).looseIsoMVA[0]),
  looseIsoMVAOld(gStore.getData(this).looseIsoMVAOld[0]),
  iso(gStore.getData(this).iso[0]),
  isoDeltaBetaCorr(gStore.getData(this).isoDeltaBetaCorr[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0]),
  matchedGen(gStore.getData(this).matchedGenContainer_, gStore.getData(this).matchedGen_[0])
{
}

panda::Tau::Tau(Tau const& _src) :
  ParticleM(new TauArray(1, _src.getName())),
  charge(gStore.getData(this).charge[0]),
  decayMode(gStore.getData(this).decayMode[0]),
  decayModeNew(gStore.getData(this).decayModeNew[0]),
  looseIsoMVA(gStore.getData(this).looseIsoMVA[0]),
  looseIsoMVAOld(gStore.getData(this).looseIsoMVAOld[0]),
  iso(gStore.getData(this).iso[0]),
  isoDeltaBetaCorr(gStore.getData(this).isoDeltaBetaCorr[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0]),
  matchedGen(gStore.getData(this).matchedGenContainer_, gStore.getData(this).matchedGen_[0])
{
  ParticleM::operator=(_src);

  charge = _src.charge;
  decayMode = _src.decayMode;
  decayModeNew = _src.decayModeNew;
  looseIsoMVA = _src.looseIsoMVA;
  looseIsoMVAOld = _src.looseIsoMVAOld;
  iso = _src.iso;
  isoDeltaBetaCorr = _src.isoDeltaBetaCorr;
  vertex = _src.vertex;
  matchedGen = _src.matchedGen;
}

panda::Tau::Tau(datastore& _data, UInt_t _idx) :
  ParticleM(_data, _idx),
  charge(_data.charge[_idx]),
  decayMode(_data.decayMode[_idx]),
  decayModeNew(_data.decayModeNew[_idx]),
  looseIsoMVA(_data.looseIsoMVA[_idx]),
  looseIsoMVAOld(_data.looseIsoMVAOld[_idx]),
  iso(_data.iso[_idx]),
  isoDeltaBetaCorr(_data.isoDeltaBetaCorr[_idx]),
  vertex(_data.vertexContainer_, _data.vertex_[_idx]),
  matchedGen(_data.matchedGenContainer_, _data.matchedGen_[_idx])
{
}

panda::Tau::Tau(ArrayBase* _array) :
  ParticleM(_array),
  charge(gStore.getData(this).charge[0]),
  decayMode(gStore.getData(this).decayMode[0]),
  decayModeNew(gStore.getData(this).decayModeNew[0]),
  looseIsoMVA(gStore.getData(this).looseIsoMVA[0]),
  looseIsoMVAOld(gStore.getData(this).looseIsoMVAOld[0]),
  iso(gStore.getData(this).iso[0]),
  isoDeltaBetaCorr(gStore.getData(this).isoDeltaBetaCorr[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0]),
  matchedGen(gStore.getData(this).matchedGenContainer_, gStore.getData(this).matchedGen_[0])
{
}

panda::Tau::~Tau()
{
  destructor();
}

void
panda::Tau::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM Tau.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    ParticleM::destructor(kTRUE);
}

panda::Tau&
panda::Tau::operator=(Tau const& _src)
{
  ParticleM::operator=(_src);

  charge = _src.charge;
  decayMode = _src.decayMode;
  decayModeNew = _src.decayModeNew;
  looseIsoMVA = _src.looseIsoMVA;
  looseIsoMVAOld = _src.looseIsoMVAOld;
  iso = _src.iso;
  isoDeltaBetaCorr = _src.isoDeltaBetaCorr;
  vertex = _src.vertex;
  matchedGen = _src.matchedGen;

  /* BEGIN CUSTOM Tau.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::Tau::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  ParticleM::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "charge", "", 'B', &charge, _branches);
  utils::book(_tree, _name, "decayMode", "", 'O', &decayMode, _branches);
  utils::book(_tree, _name, "decayModeNew", "", 'O', &decayModeNew, _branches);
  utils::book(_tree, _name, "looseIsoMVA", "", 'O', &looseIsoMVA, _branches);
  utils::book(_tree, _name, "looseIsoMVAOld", "", 'O', &looseIsoMVAOld, _branches);
  utils::book(_tree, _name, "iso", "", 'F', &iso, _branches);
  utils::book(_tree, _name, "isoDeltaBetaCorr", "", 'F', &isoDeltaBetaCorr, _branches);
  utils::book(_tree, _name, "vertex_", "", 'S', gStore.getData(this).vertex_, _branches);
  utils::book(_tree, _name, "matchedGen_", "", 'S', gStore.getData(this).matchedGen_, _branches);
}

void
panda::Tau::doInit_()
{
  ParticleM::doInit_();

  charge = 0;
  decayMode = false;
  decayModeNew = false;
  looseIsoMVA = false;
  looseIsoMVAOld = false;
  iso = 0.;
  isoDeltaBetaCorr = 0.;
  vertex.init();
  matchedGen.init();

  /* BEGIN CUSTOM Tau.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::Tau::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM Tau.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::Tau::dump(std::ostream& _out/* = std::cout*/) const
{
  ParticleM::dump(_out);

  _out << "charge = " << charge << std::endl;
  _out << "decayMode = " << decayMode << std::endl;
  _out << "decayModeNew = " << decayModeNew << std::endl;
  _out << "looseIsoMVA = " << looseIsoMVA << std::endl;
  _out << "looseIsoMVAOld = " << looseIsoMVAOld << std::endl;
  _out << "iso = " << iso << std::endl;
  _out << "isoDeltaBetaCorr = " << isoDeltaBetaCorr << std::endl;
  _out << "vertex = " << vertex << std::endl;
  _out << "matchedGen = " << matchedGen << std::endl;
}

/* BEGIN CUSTOM Tau.cc.global */
/* END CUSTOM */
