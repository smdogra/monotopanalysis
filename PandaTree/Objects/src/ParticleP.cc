#include "../interface/ParticleP.h"

/*static*/
panda::utils::BranchList
panda::ParticleP::getListOfBranches()
{
  utils::BranchList blist;
  blist += Particle::getListOfBranches();
  blist += {"pt_", "eta_", "phi_"};
  return blist;
}

void
panda::ParticleP::datastore::allocate(UInt_t _nmax)
{
  Particle::datastore::allocate(_nmax);

  pt_ = new Float_t[nmax_];
  eta_ = new Float_t[nmax_];
  phi_ = new Float_t[nmax_];
}

void
panda::ParticleP::datastore::deallocate()
{
  Particle::datastore::deallocate();

  delete [] pt_;
  pt_ = 0;
  delete [] eta_;
  eta_ = 0;
  delete [] phi_;
  phi_ = 0;
}

void
panda::ParticleP::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  Particle::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "pt_", _branches);
  utils::setStatus(_tree, _name, "eta_", _branches);
  utils::setStatus(_tree, _name, "phi_", _branches);
}

panda::utils::BranchList
panda::ParticleP::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(Particle::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "pt_"));
  blist.push_back(utils::getStatus(_tree, _name, "eta_"));
  blist.push_back(utils::getStatus(_tree, _name, "phi_"));

  return blist;
}

void
panda::ParticleP::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  Particle::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "pt_", pt_, _branches, _setStatus);
  utils::setAddress(_tree, _name, "eta_", eta_, _branches, _setStatus);
  utils::setAddress(_tree, _name, "phi_", phi_, _branches, _setStatus);
}

void
panda::ParticleP::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  Particle::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "pt_", size, 'F', pt_, _branches);
  utils::book(_tree, _name, "eta_", size, 'F', eta_, _branches);
  utils::book(_tree, _name, "phi_", size, 'F', phi_, _branches);
}

void
panda::ParticleP::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  Particle::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "pt_");
  utils::resetAddress(_tree, _name, "eta_");
  utils::resetAddress(_tree, _name, "phi_");
}

void
panda::ParticleP::datastore::resizeVectors_(UInt_t _size)
{
  Particle::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::ParticleP::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return ParticleP::getListOfBranches().fullNames(_name);
}

panda::ParticleP::ParticleP(char const* _name/* = ""*/) :
  Particle(new ParticlePArray(1, _name)),
  pt_(gStore.getData(this).pt_[0]),
  eta_(gStore.getData(this).eta_[0]),
  phi_(gStore.getData(this).phi_[0])
{
}

panda::ParticleP::ParticleP(ParticleP const& _src) :
  Particle(new ParticlePArray(1, _src.getName())),
  pt_(gStore.getData(this).pt_[0]),
  eta_(gStore.getData(this).eta_[0]),
  phi_(gStore.getData(this).phi_[0])
{
  Particle::operator=(_src);

  pt_ = _src.pt_;
  eta_ = _src.eta_;
  phi_ = _src.phi_;
}

panda::ParticleP::ParticleP(datastore& _data, UInt_t _idx) :
  Particle(_data, _idx),
  pt_(_data.pt_[_idx]),
  eta_(_data.eta_[_idx]),
  phi_(_data.phi_[_idx])
{
}

panda::ParticleP::ParticleP(ArrayBase* _array) :
  Particle(_array),
  pt_(gStore.getData(this).pt_[0]),
  eta_(gStore.getData(this).eta_[0]),
  phi_(gStore.getData(this).phi_[0])
{
}

panda::ParticleP::~ParticleP()
{
  destructor();
}

void
panda::ParticleP::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM ParticleP.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    Particle::destructor(kTRUE);
}

panda::ParticleP&
panda::ParticleP::operator=(ParticleP const& _src)
{
  Particle::operator=(_src);

  pt_ = _src.pt_;
  eta_ = _src.eta_;
  phi_ = _src.phi_;

  /* BEGIN CUSTOM ParticleP.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::ParticleP::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  Particle::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "pt_", "", 'F', &pt_, _branches);
  utils::book(_tree, _name, "eta_", "", 'F', &eta_, _branches);
  utils::book(_tree, _name, "phi_", "", 'F', &phi_, _branches);
}

void
panda::ParticleP::doInit_()
{
  Particle::doInit_();

  pt_ = 0.;
  eta_ = 0.;
  phi_ = 0.;

  /* BEGIN CUSTOM ParticleP.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::ParticleP::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM ParticleP.cc.print */
  ParticleP::dump(_out);
  /* END CUSTOM */
}

void
panda::ParticleP::dump(std::ostream& _out/* = std::cout*/) const
{
  Particle::dump(_out);

  _out << "pt_ = " << pt_ << std::endl;
  _out << "eta_ = " << eta_ << std::endl;
  _out << "phi_ = " << phi_ << std::endl;
}

void
panda::ParticleP::setPtEtaPhiM(double pt, double eta, double phi, double)
{
  pt_ = pt;
  eta_ = eta;
  phi_ = phi;
}

void
panda::ParticleP::setXYZE(double px, double py, double pz, double)
{
  pt_ = std::sqrt(px * px + py * py);
  double p(std::sqrt(px * px + py * py + pz * pz));
  eta_ = 0.5 * std::log((p + pz) / (p - pz));
  phi_ = std::atan2(py, px);
}


/* BEGIN CUSTOM ParticleP.cc.global */
/* END CUSTOM */
