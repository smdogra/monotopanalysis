#include "../interface/PackedTrack.h"

/*static*/
panda::utils::BranchList
panda::PackedTrack::getListOfBranches()
{
  utils::BranchList blist;
  blist += {"packedPtError", "packedDxy", "packedDz", "packedDPhi"};
  return blist;
}

void
panda::PackedTrack::datastore::allocate(UInt_t _nmax)
{
  Element::datastore::allocate(_nmax);

  packedPtError = new UShort_t[nmax_];
  packedDxy = new Short_t[nmax_];
  packedDz = new Short_t[nmax_];
  packedDPhi = new Short_t[nmax_];
}

void
panda::PackedTrack::datastore::deallocate()
{
  Element::datastore::deallocate();

  delete [] packedPtError;
  packedPtError = 0;
  delete [] packedDxy;
  packedDxy = 0;
  delete [] packedDz;
  packedDz = 0;
  delete [] packedDPhi;
  packedDPhi = 0;
}

void
panda::PackedTrack::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  Element::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "packedPtError", _branches);
  utils::setStatus(_tree, _name, "packedDxy", _branches);
  utils::setStatus(_tree, _name, "packedDz", _branches);
  utils::setStatus(_tree, _name, "packedDPhi", _branches);
}

panda::utils::BranchList
panda::PackedTrack::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(Element::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "packedPtError"));
  blist.push_back(utils::getStatus(_tree, _name, "packedDxy"));
  blist.push_back(utils::getStatus(_tree, _name, "packedDz"));
  blist.push_back(utils::getStatus(_tree, _name, "packedDPhi"));

  return blist;
}

void
panda::PackedTrack::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  Element::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "packedPtError", packedPtError, _branches, _setStatus);
  utils::setAddress(_tree, _name, "packedDxy", packedDxy, _branches, _setStatus);
  utils::setAddress(_tree, _name, "packedDz", packedDz, _branches, _setStatus);
  utils::setAddress(_tree, _name, "packedDPhi", packedDPhi, _branches, _setStatus);
}

void
panda::PackedTrack::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  Element::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "packedPtError", size, 's', packedPtError, _branches);
  utils::book(_tree, _name, "packedDxy", size, 'S', packedDxy, _branches);
  utils::book(_tree, _name, "packedDz", size, 'S', packedDz, _branches);
  utils::book(_tree, _name, "packedDPhi", size, 'S', packedDPhi, _branches);
}

void
panda::PackedTrack::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  Element::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "packedPtError");
  utils::resetAddress(_tree, _name, "packedDxy");
  utils::resetAddress(_tree, _name, "packedDz");
  utils::resetAddress(_tree, _name, "packedDPhi");
}

void
panda::PackedTrack::datastore::resizeVectors_(UInt_t _size)
{
  Element::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::PackedTrack::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return PackedTrack::getListOfBranches().fullNames(_name);
}

panda::PackedTrack::PackedTrack(char const* _name/* = ""*/) :
  Element(new PackedTrackArray(1, _name)),
  packedPtError(gStore.getData(this).packedPtError[0]),
  packedDxy(gStore.getData(this).packedDxy[0]),
  packedDz(gStore.getData(this).packedDz[0]),
  packedDPhi(gStore.getData(this).packedDPhi[0])
{
}

panda::PackedTrack::PackedTrack(PackedTrack const& _src) :
  Element(new PackedTrackArray(1, _src.getName())),
  packedPtError(gStore.getData(this).packedPtError[0]),
  packedDxy(gStore.getData(this).packedDxy[0]),
  packedDz(gStore.getData(this).packedDz[0]),
  packedDPhi(gStore.getData(this).packedDPhi[0])
{
  Element::operator=(_src);

  packedPtError = _src.packedPtError;
  packedDxy = _src.packedDxy;
  packedDz = _src.packedDz;
  packedDPhi = _src.packedDPhi;
}

panda::PackedTrack::PackedTrack(datastore& _data, UInt_t _idx) :
  Element(_data, _idx),
  packedPtError(_data.packedPtError[_idx]),
  packedDxy(_data.packedDxy[_idx]),
  packedDz(_data.packedDz[_idx]),
  packedDPhi(_data.packedDPhi[_idx])
{
}

panda::PackedTrack::PackedTrack(ArrayBase* _array) :
  Element(_array),
  packedPtError(gStore.getData(this).packedPtError[0]),
  packedDxy(gStore.getData(this).packedDxy[0]),
  packedDz(gStore.getData(this).packedDz[0]),
  packedDPhi(gStore.getData(this).packedDPhi[0])
{
}

panda::PackedTrack::~PackedTrack()
{
  destructor();
}

void
panda::PackedTrack::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM PackedTrack.cc.destructor */
  /* END CUSTOM */
}

panda::PackedTrack&
panda::PackedTrack::operator=(PackedTrack const& _src)
{
  packedPtError = _src.packedPtError;
  packedDxy = _src.packedDxy;
  packedDz = _src.packedDz;
  packedDPhi = _src.packedDPhi;

  /* BEGIN CUSTOM PackedTrack.cc.operator= */
  ptError_ = _src.ptError_;
  dxy_ = _src.dxy_;
  dz_ = _src.dz_;
  dPhi_ = _src.dPhi_;
  unpacked_ = _src.unpacked_;
  /* END CUSTOM */

  return *this;
}

void
panda::PackedTrack::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  utils::book(_tree, _name, "packedPtError", "", 's', &packedPtError, _branches);
  utils::book(_tree, _name, "packedDxy", "", 'S', &packedDxy, _branches);
  utils::book(_tree, _name, "packedDz", "", 'S', &packedDz, _branches);
  utils::book(_tree, _name, "packedDPhi", "", 'S', &packedDPhi, _branches);
}

void
panda::PackedTrack::doInit_()
{
  packedPtError = 0;
  packedDxy = 0;
  packedDz = 0;
  packedDPhi = 0;

  /* BEGIN CUSTOM PackedTrack.cc.doInit_ */
  ptError_ = 0.;
  dxy_ = 0.;
  dz_ = 0.;
  dPhi_ = 0.;
  unpacked_ = kFALSE;
  /* END CUSTOM */
}

void
panda::PackedTrack::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM PackedTrack.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::PackedTrack::dump(std::ostream& _out/* = std::cout*/) const
{
  _out << "packedPtError = " << packedPtError << std::endl;
  _out << "packedDxy = " << packedDxy << std::endl;
  _out << "packedDz = " << packedDz << std::endl;
  _out << "packedDPhi = " << packedDPhi << std::endl;
}


/* BEGIN CUSTOM PackedTrack.cc.global */
#include "../interface/PackingHelper.h"

void
panda::PackedTrack::pack_()
{
  packPtError_();
  packedDxy = PackingHelper::singleton().packUnbound(dxy_ * 100.);
  packedDz = PackingHelper::singleton().packUnbound(dz_ * 100.);
  packedDPhi = std::round(dPhi_/3.2f*std::numeric_limits<Short_t>::max());
}

void
panda::PackedTrack::packPtError_()
{
  packedPtError = PackingHelper::singleton().packUnbound(ptError_);
}

void
panda::PackedTrack::unpack_() const
{
  if (unpacked_)
    return;

  ptError_ = PackingHelper::singleton().unpackUnbound(packedPtError);
  dxy_ = PackingHelper::singleton().unpackUnbound(packedDxy) / 100.;
  dz_ = PackingHelper::singleton().unpackUnbound(packedDz) / 100.;
  dPhi_ = packedDPhi * 3.2f / std::numeric_limits<Short_t>::max();

  unpacked_ = true;
}

/* END CUSTOM */
