#include "../interface/Lepton.h"

/*static*/
panda::utils::BranchList
panda::Lepton::getListOfBranches()
{
  utils::BranchList blist;
  blist += ParticleP::getListOfBranches();
  blist += {"pfPt", "charge", "loose", "medium", "tight", "chIso", "nhIso", "phIso", "puIso", "matchedPF_", "matchedGen_", "vertex_"};
  return blist;
}

void
panda::Lepton::datastore::allocate(UInt_t _nmax)
{
  ParticleP::datastore::allocate(_nmax);

  pfPt = new Float_t[nmax_];
  charge = new Char_t[nmax_];
  loose = new Bool_t[nmax_];
  medium = new Bool_t[nmax_];
  tight = new Bool_t[nmax_];
  chIso = new Float_t[nmax_];
  nhIso = new Float_t[nmax_];
  phIso = new Float_t[nmax_];
  puIso = new Float_t[nmax_];
  matchedPF_ = new Short_t[nmax_];
  matchedGen_ = new Short_t[nmax_];
  vertex_ = new Short_t[nmax_];
}

void
panda::Lepton::datastore::deallocate()
{
  ParticleP::datastore::deallocate();

  delete [] pfPt;
  pfPt = 0;
  delete [] charge;
  charge = 0;
  delete [] loose;
  loose = 0;
  delete [] medium;
  medium = 0;
  delete [] tight;
  tight = 0;
  delete [] chIso;
  chIso = 0;
  delete [] nhIso;
  nhIso = 0;
  delete [] phIso;
  phIso = 0;
  delete [] puIso;
  puIso = 0;
  delete [] matchedPF_;
  matchedPF_ = 0;
  delete [] matchedGen_;
  matchedGen_ = 0;
  delete [] vertex_;
  vertex_ = 0;
}

void
panda::Lepton::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  ParticleP::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "pfPt", _branches);
  utils::setStatus(_tree, _name, "charge", _branches);
  utils::setStatus(_tree, _name, "loose", _branches);
  utils::setStatus(_tree, _name, "medium", _branches);
  utils::setStatus(_tree, _name, "tight", _branches);
  utils::setStatus(_tree, _name, "chIso", _branches);
  utils::setStatus(_tree, _name, "nhIso", _branches);
  utils::setStatus(_tree, _name, "phIso", _branches);
  utils::setStatus(_tree, _name, "puIso", _branches);
  utils::setStatus(_tree, _name, "matchedPF_", _branches);
  utils::setStatus(_tree, _name, "matchedGen_", _branches);
  utils::setStatus(_tree, _name, "vertex_", _branches);
}

panda::utils::BranchList
panda::Lepton::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(ParticleP::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "pfPt"));
  blist.push_back(utils::getStatus(_tree, _name, "charge"));
  blist.push_back(utils::getStatus(_tree, _name, "loose"));
  blist.push_back(utils::getStatus(_tree, _name, "medium"));
  blist.push_back(utils::getStatus(_tree, _name, "tight"));
  blist.push_back(utils::getStatus(_tree, _name, "chIso"));
  blist.push_back(utils::getStatus(_tree, _name, "nhIso"));
  blist.push_back(utils::getStatus(_tree, _name, "phIso"));
  blist.push_back(utils::getStatus(_tree, _name, "puIso"));
  blist.push_back(utils::getStatus(_tree, _name, "matchedPF_"));
  blist.push_back(utils::getStatus(_tree, _name, "matchedGen_"));
  blist.push_back(utils::getStatus(_tree, _name, "vertex_"));

  return blist;
}

void
panda::Lepton::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  ParticleP::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "pfPt", pfPt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "charge", charge, _branches, _setStatus);
  utils::setAddress(_tree, _name, "loose", loose, _branches, _setStatus);
  utils::setAddress(_tree, _name, "medium", medium, _branches, _setStatus);
  utils::setAddress(_tree, _name, "tight", tight, _branches, _setStatus);
  utils::setAddress(_tree, _name, "chIso", chIso, _branches, _setStatus);
  utils::setAddress(_tree, _name, "nhIso", nhIso, _branches, _setStatus);
  utils::setAddress(_tree, _name, "phIso", phIso, _branches, _setStatus);
  utils::setAddress(_tree, _name, "puIso", puIso, _branches, _setStatus);
  utils::setAddress(_tree, _name, "matchedPF_", matchedPF_, _branches, _setStatus);
  utils::setAddress(_tree, _name, "matchedGen_", matchedGen_, _branches, _setStatus);
  utils::setAddress(_tree, _name, "vertex_", vertex_, _branches, _setStatus);
}

void
panda::Lepton::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  ParticleP::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "pfPt", size, 'F', pfPt, _branches);
  utils::book(_tree, _name, "charge", size, 'B', charge, _branches);
  utils::book(_tree, _name, "loose", size, 'O', loose, _branches);
  utils::book(_tree, _name, "medium", size, 'O', medium, _branches);
  utils::book(_tree, _name, "tight", size, 'O', tight, _branches);
  utils::book(_tree, _name, "chIso", size, 'F', chIso, _branches);
  utils::book(_tree, _name, "nhIso", size, 'F', nhIso, _branches);
  utils::book(_tree, _name, "phIso", size, 'F', phIso, _branches);
  utils::book(_tree, _name, "puIso", size, 'F', puIso, _branches);
  utils::book(_tree, _name, "matchedPF_", size, 'S', matchedPF_, _branches);
  utils::book(_tree, _name, "matchedGen_", size, 'S', matchedGen_, _branches);
  utils::book(_tree, _name, "vertex_", size, 'S', vertex_, _branches);
}

void
panda::Lepton::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  ParticleP::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "pfPt");
  utils::resetAddress(_tree, _name, "charge");
  utils::resetAddress(_tree, _name, "loose");
  utils::resetAddress(_tree, _name, "medium");
  utils::resetAddress(_tree, _name, "tight");
  utils::resetAddress(_tree, _name, "chIso");
  utils::resetAddress(_tree, _name, "nhIso");
  utils::resetAddress(_tree, _name, "phIso");
  utils::resetAddress(_tree, _name, "puIso");
  utils::resetAddress(_tree, _name, "matchedPF_");
  utils::resetAddress(_tree, _name, "matchedGen_");
  utils::resetAddress(_tree, _name, "vertex_");
}

void
panda::Lepton::datastore::resizeVectors_(UInt_t _size)
{
  ParticleP::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::Lepton::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return Lepton::getListOfBranches().fullNames(_name);
}

panda::Lepton::Lepton(char const* _name/* = ""*/) :
  ParticleP(new LeptonArray(1, _name)),
  pfPt(gStore.getData(this).pfPt[0]),
  charge(gStore.getData(this).charge[0]),
  loose(gStore.getData(this).loose[0]),
  medium(gStore.getData(this).medium[0]),
  tight(gStore.getData(this).tight[0]),
  chIso(gStore.getData(this).chIso[0]),
  nhIso(gStore.getData(this).nhIso[0]),
  phIso(gStore.getData(this).phIso[0]),
  puIso(gStore.getData(this).puIso[0]),
  matchedPF(gStore.getData(this).matchedPFContainer_, gStore.getData(this).matchedPF_[0]),
  matchedGen(gStore.getData(this).matchedGenContainer_, gStore.getData(this).matchedGen_[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0])
{
}

panda::Lepton::Lepton(Lepton const& _src) :
  ParticleP(new LeptonArray(1, _src.getName())),
  pfPt(gStore.getData(this).pfPt[0]),
  charge(gStore.getData(this).charge[0]),
  loose(gStore.getData(this).loose[0]),
  medium(gStore.getData(this).medium[0]),
  tight(gStore.getData(this).tight[0]),
  chIso(gStore.getData(this).chIso[0]),
  nhIso(gStore.getData(this).nhIso[0]),
  phIso(gStore.getData(this).phIso[0]),
  puIso(gStore.getData(this).puIso[0]),
  matchedPF(gStore.getData(this).matchedPFContainer_, gStore.getData(this).matchedPF_[0]),
  matchedGen(gStore.getData(this).matchedGenContainer_, gStore.getData(this).matchedGen_[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0])
{
  ParticleP::operator=(_src);

  pfPt = _src.pfPt;
  charge = _src.charge;
  loose = _src.loose;
  medium = _src.medium;
  tight = _src.tight;
  chIso = _src.chIso;
  nhIso = _src.nhIso;
  phIso = _src.phIso;
  puIso = _src.puIso;
  matchedPF = _src.matchedPF;
  matchedGen = _src.matchedGen;
  vertex = _src.vertex;
}

panda::Lepton::Lepton(datastore& _data, UInt_t _idx) :
  ParticleP(_data, _idx),
  pfPt(_data.pfPt[_idx]),
  charge(_data.charge[_idx]),
  loose(_data.loose[_idx]),
  medium(_data.medium[_idx]),
  tight(_data.tight[_idx]),
  chIso(_data.chIso[_idx]),
  nhIso(_data.nhIso[_idx]),
  phIso(_data.phIso[_idx]),
  puIso(_data.puIso[_idx]),
  matchedPF(_data.matchedPFContainer_, _data.matchedPF_[_idx]),
  matchedGen(_data.matchedGenContainer_, _data.matchedGen_[_idx]),
  vertex(_data.vertexContainer_, _data.vertex_[_idx])
{
}

panda::Lepton::Lepton(ArrayBase* _array) :
  ParticleP(_array),
  pfPt(gStore.getData(this).pfPt[0]),
  charge(gStore.getData(this).charge[0]),
  loose(gStore.getData(this).loose[0]),
  medium(gStore.getData(this).medium[0]),
  tight(gStore.getData(this).tight[0]),
  chIso(gStore.getData(this).chIso[0]),
  nhIso(gStore.getData(this).nhIso[0]),
  phIso(gStore.getData(this).phIso[0]),
  puIso(gStore.getData(this).puIso[0]),
  matchedPF(gStore.getData(this).matchedPFContainer_, gStore.getData(this).matchedPF_[0]),
  matchedGen(gStore.getData(this).matchedGenContainer_, gStore.getData(this).matchedGen_[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0])
{
}

panda::Lepton::~Lepton()
{
  destructor();
}

void
panda::Lepton::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM Lepton.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    ParticleP::destructor(kTRUE);
}

panda::Lepton&
panda::Lepton::operator=(Lepton const& _src)
{
  ParticleP::operator=(_src);

  pfPt = _src.pfPt;
  charge = _src.charge;
  loose = _src.loose;
  medium = _src.medium;
  tight = _src.tight;
  chIso = _src.chIso;
  nhIso = _src.nhIso;
  phIso = _src.phIso;
  puIso = _src.puIso;
  matchedPF = _src.matchedPF;
  matchedGen = _src.matchedGen;
  vertex = _src.vertex;

  /* BEGIN CUSTOM Lepton.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::Lepton::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  ParticleP::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "pfPt", "", 'F', &pfPt, _branches);
  utils::book(_tree, _name, "charge", "", 'B', &charge, _branches);
  utils::book(_tree, _name, "loose", "", 'O', &loose, _branches);
  utils::book(_tree, _name, "medium", "", 'O', &medium, _branches);
  utils::book(_tree, _name, "tight", "", 'O', &tight, _branches);
  utils::book(_tree, _name, "chIso", "", 'F', &chIso, _branches);
  utils::book(_tree, _name, "nhIso", "", 'F', &nhIso, _branches);
  utils::book(_tree, _name, "phIso", "", 'F', &phIso, _branches);
  utils::book(_tree, _name, "puIso", "", 'F', &puIso, _branches);
  utils::book(_tree, _name, "matchedPF_", "", 'S', gStore.getData(this).matchedPF_, _branches);
  utils::book(_tree, _name, "matchedGen_", "", 'S', gStore.getData(this).matchedGen_, _branches);
  utils::book(_tree, _name, "vertex_", "", 'S', gStore.getData(this).vertex_, _branches);
}

void
panda::Lepton::doInit_()
{
  ParticleP::doInit_();

  pfPt = -1.;
  charge = 0;
  loose = false;
  medium = false;
  tight = false;
  chIso = 0.;
  nhIso = 0.;
  phIso = 0.;
  puIso = 0.;
  matchedPF.init();
  matchedGen.init();
  vertex.init();

  /* BEGIN CUSTOM Lepton.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::Lepton::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM Lepton.cc.print */
  if (_level >= 3) {
    Lepton::dump(_out);
  }
  else if (_level == 2) {
    ParticleP::print(_out, _level);

    _out << "pfPt = " << pfPt << std::endl;
    _out << "charge = " << charge << std::endl;
    _out << "loose = " << loose << std::endl;
    _out << "medium = " << medium << std::endl;
    _out << "tight = " << tight << std::endl;
  }
  else
    return;
  /* END CUSTOM */
}

void
panda::Lepton::dump(std::ostream& _out/* = std::cout*/) const
{
  ParticleP::dump(_out);

  _out << "pfPt = " << pfPt << std::endl;
  _out << "charge = " << charge << std::endl;
  _out << "loose = " << loose << std::endl;
  _out << "medium = " << medium << std::endl;
  _out << "tight = " << tight << std::endl;
  _out << "chIso = " << chIso << std::endl;
  _out << "nhIso = " << nhIso << std::endl;
  _out << "phIso = " << phIso << std::endl;
  _out << "puIso = " << puIso << std::endl;
  _out << "matchedPF = " << matchedPF << std::endl;
  _out << "matchedGen = " << matchedGen << std::endl;
  _out << "vertex = " << vertex << std::endl;
}


/* BEGIN CUSTOM Lepton.cc.global */
/* END CUSTOM */
