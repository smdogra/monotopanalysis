#include "../interface/PFCand.h"

TString panda::PFCand::PTypeName[] = {
  "hp",
  "hm",
  "ep",
  "em",
  "mup",
  "mum",
  "gamma",
  "h0",
  "h_HF",
  "egamma_HF",
  "Xp",
  "Xm",
  "X"
};

/*static*/
int const panda::PFCand::q_[nPTypes] = {1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 0};
/*static*/
int const panda::PFCand::pdgId_[nPTypes] = {211, -211, -11, 11, -13, 13, 22, 130, 1, 2, 0, 0, 0};

/*static*/
panda::utils::BranchList
panda::PFCand::getListOfBranches()
{
  utils::BranchList blist;
  blist += PackedParticle::getListOfBranches();
  blist += {"packedPuppiW", "packedPuppiWNoLepDiff", "ptype"};
  return blist;
}

void
panda::PFCand::datastore::allocate(UInt_t _nmax)
{
  PackedParticle::datastore::allocate(_nmax);

  packedPuppiW = new Char_t[nmax_];
  packedPuppiWNoLepDiff = new Char_t[nmax_];
  ptype = new UChar_t[nmax_];
  vertex_ = new Short_t[nmax_];
  track_ = new Short_t[nmax_];
}

void
panda::PFCand::datastore::deallocate()
{
  PackedParticle::datastore::deallocate();

  delete [] packedPuppiW;
  packedPuppiW = 0;
  delete [] packedPuppiWNoLepDiff;
  packedPuppiWNoLepDiff = 0;
  delete [] ptype;
  ptype = 0;
  delete [] vertex_;
  vertex_ = 0;
  delete [] track_;
  track_ = 0;
}

void
panda::PFCand::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  PackedParticle::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "packedPuppiW", _branches);
  utils::setStatus(_tree, _name, "packedPuppiWNoLepDiff", _branches);
  utils::setStatus(_tree, _name, "ptype", _branches);
}

panda::utils::BranchList
panda::PFCand::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(PackedParticle::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "packedPuppiW"));
  blist.push_back(utils::getStatus(_tree, _name, "packedPuppiWNoLepDiff"));
  blist.push_back(utils::getStatus(_tree, _name, "ptype"));

  return blist;
}

void
panda::PFCand::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  PackedParticle::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "packedPuppiW", packedPuppiW, _branches, _setStatus);
  utils::setAddress(_tree, _name, "packedPuppiWNoLepDiff", packedPuppiWNoLepDiff, _branches, _setStatus);
  utils::setAddress(_tree, _name, "ptype", ptype, _branches, _setStatus);
}

void
panda::PFCand::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  PackedParticle::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "packedPuppiW", size, 'B', packedPuppiW, _branches);
  utils::book(_tree, _name, "packedPuppiWNoLepDiff", size, 'B', packedPuppiWNoLepDiff, _branches);
  utils::book(_tree, _name, "ptype", size, 'b', ptype, _branches);
}

void
panda::PFCand::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  PackedParticle::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "packedPuppiW");
  utils::resetAddress(_tree, _name, "packedPuppiWNoLepDiff");
  utils::resetAddress(_tree, _name, "ptype");
}

void
panda::PFCand::datastore::resizeVectors_(UInt_t _size)
{
  PackedParticle::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::PFCand::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return PFCand::getListOfBranches().fullNames(_name);
}

panda::PFCand::PFCand(char const* _name/* = ""*/) :
  PackedParticle(new PFCandArray(1, _name)),
  packedPuppiW(gStore.getData(this).packedPuppiW[0]),
  packedPuppiWNoLepDiff(gStore.getData(this).packedPuppiWNoLepDiff[0]),
  ptype(gStore.getData(this).ptype[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0]),
  track(gStore.getData(this).trackContainer_, gStore.getData(this).track_[0])
{
}

panda::PFCand::PFCand(PFCand const& _src) :
  PackedParticle(new PFCandArray(1, _src.getName())),
  packedPuppiW(gStore.getData(this).packedPuppiW[0]),
  packedPuppiWNoLepDiff(gStore.getData(this).packedPuppiWNoLepDiff[0]),
  ptype(gStore.getData(this).ptype[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0]),
  track(gStore.getData(this).trackContainer_, gStore.getData(this).track_[0])
{
  PackedParticle::operator=(_src);

  packedPuppiW = _src.packedPuppiW;
  packedPuppiWNoLepDiff = _src.packedPuppiWNoLepDiff;
  ptype = _src.ptype;
  vertex = _src.vertex;
  track = _src.track;
}

panda::PFCand::PFCand(datastore& _data, UInt_t _idx) :
  PackedParticle(_data, _idx),
  packedPuppiW(_data.packedPuppiW[_idx]),
  packedPuppiWNoLepDiff(_data.packedPuppiWNoLepDiff[_idx]),
  ptype(_data.ptype[_idx]),
  vertex(_data.vertexContainer_, _data.vertex_[_idx]),
  track(_data.trackContainer_, _data.track_[_idx])
{
}

panda::PFCand::PFCand(ArrayBase* _array) :
  PackedParticle(_array),
  packedPuppiW(gStore.getData(this).packedPuppiW[0]),
  packedPuppiWNoLepDiff(gStore.getData(this).packedPuppiWNoLepDiff[0]),
  ptype(gStore.getData(this).ptype[0]),
  vertex(gStore.getData(this).vertexContainer_, gStore.getData(this).vertex_[0]),
  track(gStore.getData(this).trackContainer_, gStore.getData(this).track_[0])
{
}

panda::PFCand::~PFCand()
{
  destructor();
}

void
panda::PFCand::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM PFCand.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    PackedParticle::destructor(kTRUE);
}

panda::PFCand&
panda::PFCand::operator=(PFCand const& _src)
{
  PackedParticle::operator=(_src);

  packedPuppiW = _src.packedPuppiW;
  packedPuppiWNoLepDiff = _src.packedPuppiWNoLepDiff;
  ptype = _src.ptype;
  vertex = _src.vertex;
  track = _src.track;

  /* BEGIN CUSTOM PFCand.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::PFCand::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  PackedParticle::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "packedPuppiW", "", 'B', &packedPuppiW, _branches);
  utils::book(_tree, _name, "packedPuppiWNoLepDiff", "", 'B', &packedPuppiWNoLepDiff, _branches);
  utils::book(_tree, _name, "ptype", "", 'b', &ptype, _branches);
}

void
panda::PFCand::doInit_()
{
  PackedParticle::doInit_();

  packedPuppiW = 0;
  packedPuppiWNoLepDiff = 0;
  ptype = 0;
  vertex.init();
  track.init();

  /* BEGIN CUSTOM PFCand.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::PFCand::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM PFCand.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::PFCand::dump(std::ostream& _out/* = std::cout*/) const
{
  PackedParticle::dump(_out);

  _out << "packedPuppiW = " << packedPuppiW << std::endl;
  _out << "packedPuppiWNoLepDiff = " << packedPuppiWNoLepDiff << std::endl;
  _out << "ptype = " << ptype << std::endl;
  _out << "vertex = " << vertex << std::endl;
  _out << "track = " << track << std::endl;
}


/* BEGIN CUSTOM PFCand.cc.global */
TVector3
panda::PFCand::pca() const
{
  if (!vertex.isValid() || !track.isValid())
    return TVector3();

  unpack_();
  
  double trkPhi(phi_ + track->dPhi());

  // dir(pca - pv) is perpendicular to trkPhi by definition
  // -> relation between pca and pv simplifies

  return TVector3(vertex->x - track->dxy() * std::sin(trkPhi), vertex->y + track->dxy() * std::cos(trkPhi), vertex->z + track->dz());
}

double
panda::PFCand::dxy(TVector3 const& point) const
{
  if (!vertex.isValid() || !track.isValid())
    return 0.;

  auto ref(pca());

  double trkPhi(phi_ + track->dPhi());

  return -(ref.X() - point.X()) * std::sin(trkPhi) + (ref.Y() - point.Y()) * std::cos(trkPhi);
}

double
panda::PFCand::dz(TVector3 const& point) const
{
  if (!vertex.isValid() || !track.isValid())
    return 0.;

  auto ref(pca());

  double trkPhi(phi_ + track->dPhi());

  return (ref.Z() - point.Z()) - ((ref.X() - point.X()) * std::cos(trkPhi) + (ref.Y() - point.Y()) * std::sin(trkPhi)) * pz() / pt_;
}

#include "../interface/PackingHelper.h"

void
panda::PFCand::packMore_()
{
  packedPuppiW = PackingHelper::singleton().pack8LogBound((puppiW_ - 0.5) * 2., -2., 0., 64);
  packedPuppiWNoLepDiff = PackingHelper::singleton().pack8LogBound((puppiWNoLep_ - 0.5) * 2., -2., 0., 64) - packedPuppiW;
}

void
panda::PFCand::unpackMore_() const
{
  puppiW_ = PackingHelper::singleton().unpack8LogBound(packedPuppiW, -2., 0., 64) * 0.5 + 0.5;
  puppiWNoLep_ = PackingHelper::singleton().unpack8LogBound(packedPuppiWNoLepDiff + packedPuppiW, -2., 0., 64) * 0.5 + 0.5;
}

void
panda::PFCand::setPuppiW(double w, double wnl)
{
  puppiW_ = w;
  puppiWNoLep_ = wnl;
  packMore_();
}
/* END CUSTOM */
