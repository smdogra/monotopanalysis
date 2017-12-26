#include "../interface/ParticleM.h"

/*static*/
panda::utils::BranchList
panda::ParticleM::getListOfBranches()
{
  utils::BranchList blist;
  blist += ParticleP::getListOfBranches();
  blist += {"mass_"};
  return blist;
}

void
panda::ParticleM::datastore::allocate(UInt_t _nmax)
{
  ParticleP::datastore::allocate(_nmax);

  mass_ = new Float_t[nmax_];
}

void
panda::ParticleM::datastore::deallocate()
{
  ParticleP::datastore::deallocate();

  delete [] mass_;
  mass_ = 0;
}

void
panda::ParticleM::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  ParticleP::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "mass_", _branches);
}

panda::utils::BranchList
panda::ParticleM::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(ParticleP::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "mass_"));

  return blist;
}

void
panda::ParticleM::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  ParticleP::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "mass_", mass_, _branches, _setStatus);
}

void
panda::ParticleM::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  ParticleP::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "mass_", size, 'F', mass_, _branches);
}

void
panda::ParticleM::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  ParticleP::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "mass_");
}

void
panda::ParticleM::datastore::resizeVectors_(UInt_t _size)
{
  ParticleP::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::ParticleM::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return ParticleM::getListOfBranches().fullNames(_name);
}

panda::ParticleM::ParticleM(char const* _name/* = ""*/) :
  ParticleP(new ParticleMArray(1, _name)),
  mass_(gStore.getData(this).mass_[0])
{
}

panda::ParticleM::ParticleM(ParticleM const& _src) :
  ParticleP(new ParticleMArray(1, _src.getName())),
  mass_(gStore.getData(this).mass_[0])
{
  ParticleP::operator=(_src);

  mass_ = _src.mass_;
}

panda::ParticleM::ParticleM(datastore& _data, UInt_t _idx) :
  ParticleP(_data, _idx),
  mass_(_data.mass_[_idx])
{
}

panda::ParticleM::ParticleM(ArrayBase* _array) :
  ParticleP(_array),
  mass_(gStore.getData(this).mass_[0])
{
}

panda::ParticleM::~ParticleM()
{
  destructor();
}

void
panda::ParticleM::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM ParticleM.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    ParticleP::destructor(kTRUE);
}

panda::ParticleM&
panda::ParticleM::operator=(ParticleM const& _src)
{
  ParticleP::operator=(_src);

  mass_ = _src.mass_;

  /* BEGIN CUSTOM ParticleM.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::ParticleM::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  ParticleP::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "mass_", "", 'F', &mass_, _branches);
}

void
panda::ParticleM::doInit_()
{
  ParticleP::doInit_();

  mass_ = 0.;

  /* BEGIN CUSTOM ParticleM.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::ParticleM::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM ParticleM.cc.print */
  ParticleM::dump(_out);
  /* END CUSTOM */
}

void
panda::ParticleM::dump(std::ostream& _out/* = std::cout*/) const
{
  ParticleP::dump(_out);

  _out << "mass_ = " << mass_ << std::endl;
}

void
panda::ParticleM::setPtEtaPhiM(double pt, double eta, double phi, double m)
{
  pt_ = pt;
  eta_ = eta;
  phi_ = phi;
  mass_ = m;
}

void
panda::ParticleM::setXYZE(double px, double py, double pz, double e)
{
  pt_ = std::sqrt(px * px + py * py);
  double p(std::sqrt(px * px + py * py + pz * pz));
  eta_ = 0.5 * std::log((p + pz) / (p - pz));
  phi_ = std::atan2(py, px);
  mass_ = std::sqrt(e * e - p * p);
}


/* BEGIN CUSTOM ParticleM.cc.global */
/* END CUSTOM */
