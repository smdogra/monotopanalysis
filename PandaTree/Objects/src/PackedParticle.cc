#include "../interface/PackedParticle.h"

/*static*/
panda::utils::BranchList
panda::PackedParticle::getListOfBranches()
{
  utils::BranchList blist;
  blist += Particle::getListOfBranches();
  blist += {"packedPt", "packedEta", "packedPhi", "packedM"};
  return blist;
}

void
panda::PackedParticle::datastore::allocate(UInt_t _nmax)
{
  Particle::datastore::allocate(_nmax);

  packedPt = new UShort_t[nmax_];
  packedEta = new Short_t[nmax_];
  packedPhi = new Short_t[nmax_];
  packedM = new UShort_t[nmax_];
}

void
panda::PackedParticle::datastore::deallocate()
{
  Particle::datastore::deallocate();

  delete [] packedPt;
  packedPt = 0;
  delete [] packedEta;
  packedEta = 0;
  delete [] packedPhi;
  packedPhi = 0;
  delete [] packedM;
  packedM = 0;
}

void
panda::PackedParticle::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  Particle::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "packedPt", _branches);
  utils::setStatus(_tree, _name, "packedEta", _branches);
  utils::setStatus(_tree, _name, "packedPhi", _branches);
  utils::setStatus(_tree, _name, "packedM", _branches);
}

panda::utils::BranchList
panda::PackedParticle::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(Particle::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "packedPt"));
  blist.push_back(utils::getStatus(_tree, _name, "packedEta"));
  blist.push_back(utils::getStatus(_tree, _name, "packedPhi"));
  blist.push_back(utils::getStatus(_tree, _name, "packedM"));

  return blist;
}

void
panda::PackedParticle::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  Particle::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "packedPt", packedPt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "packedEta", packedEta, _branches, _setStatus);
  utils::setAddress(_tree, _name, "packedPhi", packedPhi, _branches, _setStatus);
  utils::setAddress(_tree, _name, "packedM", packedM, _branches, _setStatus);
}

void
panda::PackedParticle::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  Particle::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "packedPt", size, 's', packedPt, _branches);
  utils::book(_tree, _name, "packedEta", size, 'S', packedEta, _branches);
  utils::book(_tree, _name, "packedPhi", size, 'S', packedPhi, _branches);
  utils::book(_tree, _name, "packedM", size, 's', packedM, _branches);
}

void
panda::PackedParticle::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  Particle::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "packedPt");
  utils::resetAddress(_tree, _name, "packedEta");
  utils::resetAddress(_tree, _name, "packedPhi");
  utils::resetAddress(_tree, _name, "packedM");
}

void
panda::PackedParticle::datastore::resizeVectors_(UInt_t _size)
{
  Particle::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::PackedParticle::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return PackedParticle::getListOfBranches().fullNames(_name);
}

panda::PackedParticle::PackedParticle(char const* _name/* = ""*/) :
  Particle(new PackedParticleArray(1, _name)),
  packedPt(gStore.getData(this).packedPt[0]),
  packedEta(gStore.getData(this).packedEta[0]),
  packedPhi(gStore.getData(this).packedPhi[0]),
  packedM(gStore.getData(this).packedM[0])
{
}

panda::PackedParticle::PackedParticle(PackedParticle const& _src) :
  Particle(new PackedParticleArray(1, _src.getName())),
  packedPt(gStore.getData(this).packedPt[0]),
  packedEta(gStore.getData(this).packedEta[0]),
  packedPhi(gStore.getData(this).packedPhi[0]),
  packedM(gStore.getData(this).packedM[0])
{
  Particle::operator=(_src);

  packedPt = _src.packedPt;
  packedEta = _src.packedEta;
  packedPhi = _src.packedPhi;
  packedM = _src.packedM;
}

panda::PackedParticle::PackedParticle(datastore& _data, UInt_t _idx) :
  Particle(_data, _idx),
  packedPt(_data.packedPt[_idx]),
  packedEta(_data.packedEta[_idx]),
  packedPhi(_data.packedPhi[_idx]),
  packedM(_data.packedM[_idx])
{
}

panda::PackedParticle::PackedParticle(ArrayBase* _array) :
  Particle(_array),
  packedPt(gStore.getData(this).packedPt[0]),
  packedEta(gStore.getData(this).packedEta[0]),
  packedPhi(gStore.getData(this).packedPhi[0]),
  packedM(gStore.getData(this).packedM[0])
{
}

panda::PackedParticle::~PackedParticle()
{
  destructor();
}

void
panda::PackedParticle::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM PackedParticle.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    Particle::destructor(kTRUE);
}

panda::PackedParticle&
panda::PackedParticle::operator=(PackedParticle const& _src)
{
  Particle::operator=(_src);

  packedPt = _src.packedPt;
  packedEta = _src.packedEta;
  packedPhi = _src.packedPhi;
  packedM = _src.packedM;

  /* BEGIN CUSTOM PackedParticle.cc.operator= */
  pt_ = _src.pt_;
  eta_ = _src.eta_;
  phi_ = _src.phi_;
  mass_ = _src.mass_;
  unpacked_ = _src.unpacked_;
  /* END CUSTOM */

  return *this;
}

void
panda::PackedParticle::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  Particle::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "packedPt", "", 's', &packedPt, _branches);
  utils::book(_tree, _name, "packedEta", "", 'S', &packedEta, _branches);
  utils::book(_tree, _name, "packedPhi", "", 'S', &packedPhi, _branches);
  utils::book(_tree, _name, "packedM", "", 's', &packedM, _branches);
}

void
panda::PackedParticle::doInit_()
{
  Particle::doInit_();

  packedPt = 0;
  packedEta = 0;
  packedPhi = 0;
  packedM = 0;

  /* BEGIN CUSTOM PackedParticle.cc.doInit_ */
  pt_ = 0.;
  eta_ = 0.;
  phi_ = 0.;
  mass_ = 0.;
  unpacked_ = kFALSE;
  /* END CUSTOM */
}

void
panda::PackedParticle::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM PackedParticle.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::PackedParticle::dump(std::ostream& _out/* = std::cout*/) const
{
  Particle::dump(_out);

  _out << "packedPt = " << packedPt << std::endl;
  _out << "packedEta = " << packedEta << std::endl;
  _out << "packedPhi = " << packedPhi << std::endl;
  _out << "packedM = " << packedM << std::endl;
}


/* BEGIN CUSTOM PackedParticle.cc.global */
#include "../interface/PackingHelper.h"

/*static*/
void
panda::PackedParticle::setPtEtaPhiM(double pt, double eta, double phi, double m)
{
  pt_ = pt;
  eta_ = eta;
  phi_ = phi;
  mass_ = m;
  unpacked_ = true;
  pack_();
}

void
panda::PackedParticle::setXYZE(double px, double py, double pz, double e)
{
  pt_ = std::sqrt(px * px + py * py);
  double p(std::sqrt(px * px + py * py + pz * pz));
  eta_ = 0.5 * std::log((p + pz) / (p - pz));
  phi_ = std::atan2(py, px);
  mass_ = std::sqrt(e * e - p * p);
  unpacked_ = true;
  pack_();
}

void
panda::PackedParticle::pack_()
{
  packedPt = PackingHelper::singleton().packUnbound(pt_);
  packedEta = std::round(eta_ / 6.0f * std::numeric_limits<Short_t>::max());
  packedPhi = std::round(phi_/3.2f*std::numeric_limits<Short_t>::max());
  packedM = PackingHelper::singleton().packUnbound(mass_);

  packMore_();
}

void
panda::PackedParticle::unpack_() const
{
  if (unpacked_)
    return;

  pt_ = PackingHelper::singleton().unpackUnbound(packedPt);
  // shift particle phi to break degeneracies in angular separations
  // plus introduce a pseudo-random sign of the shift
  double shift(pt_ < 1. ? 0.1 * pt_ : 0.1 / pt_);
  double sign((int(pt_ * 10.) % 2 == 0) ? 1 : -1);
  phi_ = (packedPhi + sign * shift) * 3.2f / std::numeric_limits<Short_t>::max();
  mass_ = PackingHelper::singleton().unpackUnbound(packedM);
  eta_ = packedEta * 6.0f / std::numeric_limits<Short_t>::max();

  unpackMore_();

  unpacked_ = true;
}

/* END CUSTOM */
