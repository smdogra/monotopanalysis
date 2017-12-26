#include "../interface/XPhoton.h"

TString panda::XPhoton::IDTuneName[] = {
  "kSpring15",
  "kSpring16",
  "kGJetsCWIso",
  "kZGCWIso"
};

/*static*/
double const panda::XPhoton::chIsoCuts[nIDTunes][2][4] = {
  {{3.32, 1.37, 0.76, 5.0}, {1.97, 1.10, 0.56, 5.0}},
  {{1.295, 0.441, 0.202, 5.0}, {1.011, 0.442, 0.034, 5.0}},
  {{1.295, 1.146, 0.202, 5.0}, {1.011, 0.442, 0.034, 5.0}},
  {{1.295, 1.163, 0.202, 5.0}, {1.011, 0.442, 0.034, 5.0}}
};
/*static*/
double const panda::XPhoton::nhIsoCuts[nIDTunes][2][4] = {
  {{1.92, 1.06, 0.97, 100000.}, {11.86, 2.69, 2.09, 100000.}},
  {{10.910, 2.725, 0.264, 100000.}, {5.931, 1.715, 0.586, 100000.}},
  {{10.910, 2.792, 0.264, 100000.}, {5.931, 1.715, 0.586, 100000.}},
  {{10.910, 7.005, 0.264, 100000.}, {5.931, 1.715, 0.586, 100000.}}
};
/*static*/
double const panda::XPhoton::phIsoCuts[nIDTunes][2][4] = {
  {{0.81, 0.28, 0.08, 2.75}, {0.83, 0.39, 0.16, 2.00}},
  {{3.630, 2.571, 2.362, 2.75}, {6.641, 3.863, 2.617, 2.00}},
  {{3.630, 2.176, 2.362, 2.75}, {6.641, 3.863, 2.617, 2.00}},
  {{3.630, 3.271, 2.362, 2.75}, {6.641, 3.863, 2.617, 2.00}}
};
/*static*/
double const panda::XPhoton::sieieCuts[nIDTunes][2][4] = {
  {{0.0102, 0.0102, 0.0100, 0.0105}, {0.0274, 0.0268, 0.0268, 0.028}},
  {{0.01031, 0.01022, 0.00994, 0.0105}, {0.03013, 0.03001, 0.03000, 0.028}},
  {{0.01031, 0.0104, 0.00994, 0.0105}, {0.03013, 0.03001, 0.03000, 0.028}},
  {{0.01031, 0.01002, 0.00994, 0.0105}, {0.03013, 0.03001, 0.03000, 0.028}}
};
/*static*/
double const panda::XPhoton::hOverECuts[nIDTunes][2][4] = {
  {{0.05, 0.05, 0.05, 0.05}, {0.05, 0.05, 0.05, 0.05}},
  {{0.0597, 0.0396, 0.0269, 0.05}, {0.0481, 0.0219, 0.0213, 0.05}},
  {{0.0597, 0.026, 0.0269, 0.05}, {0.0481, 0.0219, 0.0213, 0.05}},
  {{0.0597, 0.0263, 0.0269, 0.05}, {0.0481, 0.0219, 0.0213, 0.05}}
};

/*static*/
panda::utils::BranchList
panda::XPhoton::getListOfBranches()
{
  utils::BranchList blist;
  blist += Photon::getListOfBranches();
  blist += {"scEta", "scRawPt", "chIsoX", "chIsoMaxX", "nhIsoX", "phIsoX", "e4", "isEB", "looseX", "mediumX", "tightX", "matchedGenId"};
  return blist;
}

void
panda::XPhoton::datastore::allocate(UInt_t _nmax)
{
  Photon::datastore::allocate(_nmax);

  scEta = new Float_t[nmax_];
  scRawPt = new Float_t[nmax_];
  chIsoX = new Float_t[nmax_][nIDTunes];
  chIsoMaxX = new Float_t[nmax_][nIDTunes];
  nhIsoX = new Float_t[nmax_][nIDTunes];
  phIsoX = new Float_t[nmax_][nIDTunes];
  e4 = new Float_t[nmax_];
  isEB = new Bool_t[nmax_];
  looseX = new Bool_t[nmax_][nIDTunes];
  mediumX = new Bool_t[nmax_][nIDTunes];
  tightX = new Bool_t[nmax_][nIDTunes];
  matchedGenId = new Int_t[nmax_];
}

void
panda::XPhoton::datastore::deallocate()
{
  Photon::datastore::deallocate();

  delete [] scEta;
  scEta = 0;
  delete [] scRawPt;
  scRawPt = 0;
  delete [] chIsoX;
  chIsoX = 0;
  delete [] chIsoMaxX;
  chIsoMaxX = 0;
  delete [] nhIsoX;
  nhIsoX = 0;
  delete [] phIsoX;
  phIsoX = 0;
  delete [] e4;
  e4 = 0;
  delete [] isEB;
  isEB = 0;
  delete [] looseX;
  looseX = 0;
  delete [] mediumX;
  mediumX = 0;
  delete [] tightX;
  tightX = 0;
  delete [] matchedGenId;
  matchedGenId = 0;
}

void
panda::XPhoton::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  Photon::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "scEta", _branches);
  utils::setStatus(_tree, _name, "scRawPt", _branches);
  utils::setStatus(_tree, _name, "chIsoX", _branches);
  utils::setStatus(_tree, _name, "chIsoMaxX", _branches);
  utils::setStatus(_tree, _name, "nhIsoX", _branches);
  utils::setStatus(_tree, _name, "phIsoX", _branches);
  utils::setStatus(_tree, _name, "e4", _branches);
  utils::setStatus(_tree, _name, "isEB", _branches);
  utils::setStatus(_tree, _name, "looseX", _branches);
  utils::setStatus(_tree, _name, "mediumX", _branches);
  utils::setStatus(_tree, _name, "tightX", _branches);
  utils::setStatus(_tree, _name, "matchedGenId", _branches);
}

panda::utils::BranchList
panda::XPhoton::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(Photon::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "scEta"));
  blist.push_back(utils::getStatus(_tree, _name, "scRawPt"));
  blist.push_back(utils::getStatus(_tree, _name, "chIsoX"));
  blist.push_back(utils::getStatus(_tree, _name, "chIsoMaxX"));
  blist.push_back(utils::getStatus(_tree, _name, "nhIsoX"));
  blist.push_back(utils::getStatus(_tree, _name, "phIsoX"));
  blist.push_back(utils::getStatus(_tree, _name, "e4"));
  blist.push_back(utils::getStatus(_tree, _name, "isEB"));
  blist.push_back(utils::getStatus(_tree, _name, "looseX"));
  blist.push_back(utils::getStatus(_tree, _name, "mediumX"));
  blist.push_back(utils::getStatus(_tree, _name, "tightX"));
  blist.push_back(utils::getStatus(_tree, _name, "matchedGenId"));

  return blist;
}

void
panda::XPhoton::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  Photon::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "scEta", scEta, _branches, _setStatus);
  utils::setAddress(_tree, _name, "scRawPt", scRawPt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "chIsoX", chIsoX, _branches, _setStatus);
  utils::setAddress(_tree, _name, "chIsoMaxX", chIsoMaxX, _branches, _setStatus);
  utils::setAddress(_tree, _name, "nhIsoX", nhIsoX, _branches, _setStatus);
  utils::setAddress(_tree, _name, "phIsoX", phIsoX, _branches, _setStatus);
  utils::setAddress(_tree, _name, "e4", e4, _branches, _setStatus);
  utils::setAddress(_tree, _name, "isEB", isEB, _branches, _setStatus);
  utils::setAddress(_tree, _name, "looseX", looseX, _branches, _setStatus);
  utils::setAddress(_tree, _name, "mediumX", mediumX, _branches, _setStatus);
  utils::setAddress(_tree, _name, "tightX", tightX, _branches, _setStatus);
  utils::setAddress(_tree, _name, "matchedGenId", matchedGenId, _branches, _setStatus);
}

void
panda::XPhoton::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  Photon::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "scEta", size, 'F', scEta, _branches);
  utils::book(_tree, _name, "scRawPt", size, 'F', scRawPt, _branches);
  utils::book(_tree, _name, "chIsoX", size + TString::Format("[%d]", nIDTunes), 'F', chIsoX, _branches);
  utils::book(_tree, _name, "chIsoMaxX", size + TString::Format("[%d]", nIDTunes), 'F', chIsoMaxX, _branches);
  utils::book(_tree, _name, "nhIsoX", size + TString::Format("[%d]", nIDTunes), 'F', nhIsoX, _branches);
  utils::book(_tree, _name, "phIsoX", size + TString::Format("[%d]", nIDTunes), 'F', phIsoX, _branches);
  utils::book(_tree, _name, "e4", size, 'F', e4, _branches);
  utils::book(_tree, _name, "isEB", size, 'O', isEB, _branches);
  utils::book(_tree, _name, "looseX", size + TString::Format("[%d]", nIDTunes), 'O', looseX, _branches);
  utils::book(_tree, _name, "mediumX", size + TString::Format("[%d]", nIDTunes), 'O', mediumX, _branches);
  utils::book(_tree, _name, "tightX", size + TString::Format("[%d]", nIDTunes), 'O', tightX, _branches);
  utils::book(_tree, _name, "matchedGenId", size, 'I', matchedGenId, _branches);
}

void
panda::XPhoton::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  Photon::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "scEta");
  utils::resetAddress(_tree, _name, "scRawPt");
  utils::resetAddress(_tree, _name, "chIsoX");
  utils::resetAddress(_tree, _name, "chIsoMaxX");
  utils::resetAddress(_tree, _name, "nhIsoX");
  utils::resetAddress(_tree, _name, "phIsoX");
  utils::resetAddress(_tree, _name, "e4");
  utils::resetAddress(_tree, _name, "isEB");
  utils::resetAddress(_tree, _name, "looseX");
  utils::resetAddress(_tree, _name, "mediumX");
  utils::resetAddress(_tree, _name, "tightX");
  utils::resetAddress(_tree, _name, "matchedGenId");
}

void
panda::XPhoton::datastore::resizeVectors_(UInt_t _size)
{
  Photon::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::XPhoton::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return XPhoton::getListOfBranches().fullNames(_name);
}

panda::XPhoton::XPhoton(char const* _name/* = ""*/) :
  Photon(new XPhotonArray(1, _name)),
  scEta(gStore.getData(this).scEta[0]),
  scRawPt(gStore.getData(this).scRawPt[0]),
  chIsoX(gStore.getData(this).chIsoX[0]),
  chIsoMaxX(gStore.getData(this).chIsoMaxX[0]),
  nhIsoX(gStore.getData(this).nhIsoX[0]),
  phIsoX(gStore.getData(this).phIsoX[0]),
  e4(gStore.getData(this).e4[0]),
  isEB(gStore.getData(this).isEB[0]),
  looseX(gStore.getData(this).looseX[0]),
  mediumX(gStore.getData(this).mediumX[0]),
  tightX(gStore.getData(this).tightX[0]),
  matchedGenId(gStore.getData(this).matchedGenId[0])
{
}

panda::XPhoton::XPhoton(XPhoton const& _src) :
  Photon(new XPhotonArray(1, _src.getName())),
  scEta(gStore.getData(this).scEta[0]),
  scRawPt(gStore.getData(this).scRawPt[0]),
  chIsoX(gStore.getData(this).chIsoX[0]),
  chIsoMaxX(gStore.getData(this).chIsoMaxX[0]),
  nhIsoX(gStore.getData(this).nhIsoX[0]),
  phIsoX(gStore.getData(this).phIsoX[0]),
  e4(gStore.getData(this).e4[0]),
  isEB(gStore.getData(this).isEB[0]),
  looseX(gStore.getData(this).looseX[0]),
  mediumX(gStore.getData(this).mediumX[0]),
  tightX(gStore.getData(this).tightX[0]),
  matchedGenId(gStore.getData(this).matchedGenId[0])
{
  Photon::operator=(_src);

  scEta = _src.scEta;
  scRawPt = _src.scRawPt;
  std::memcpy(chIsoX, _src.chIsoX, sizeof(Float_t) * nIDTunes);
  std::memcpy(chIsoMaxX, _src.chIsoMaxX, sizeof(Float_t) * nIDTunes);
  std::memcpy(nhIsoX, _src.nhIsoX, sizeof(Float_t) * nIDTunes);
  std::memcpy(phIsoX, _src.phIsoX, sizeof(Float_t) * nIDTunes);
  e4 = _src.e4;
  isEB = _src.isEB;
  std::memcpy(looseX, _src.looseX, sizeof(Bool_t) * nIDTunes);
  std::memcpy(mediumX, _src.mediumX, sizeof(Bool_t) * nIDTunes);
  std::memcpy(tightX, _src.tightX, sizeof(Bool_t) * nIDTunes);
  matchedGenId = _src.matchedGenId;
}

panda::XPhoton::XPhoton(datastore& _data, UInt_t _idx) :
  Photon(_data, _idx),
  scEta(_data.scEta[_idx]),
  scRawPt(_data.scRawPt[_idx]),
  chIsoX(_data.chIsoX[_idx]),
  chIsoMaxX(_data.chIsoMaxX[_idx]),
  nhIsoX(_data.nhIsoX[_idx]),
  phIsoX(_data.phIsoX[_idx]),
  e4(_data.e4[_idx]),
  isEB(_data.isEB[_idx]),
  looseX(_data.looseX[_idx]),
  mediumX(_data.mediumX[_idx]),
  tightX(_data.tightX[_idx]),
  matchedGenId(_data.matchedGenId[_idx])
{
}

panda::XPhoton::XPhoton(ArrayBase* _array) :
  Photon(_array),
  scEta(gStore.getData(this).scEta[0]),
  scRawPt(gStore.getData(this).scRawPt[0]),
  chIsoX(gStore.getData(this).chIsoX[0]),
  chIsoMaxX(gStore.getData(this).chIsoMaxX[0]),
  nhIsoX(gStore.getData(this).nhIsoX[0]),
  phIsoX(gStore.getData(this).phIsoX[0]),
  e4(gStore.getData(this).e4[0]),
  isEB(gStore.getData(this).isEB[0]),
  looseX(gStore.getData(this).looseX[0]),
  mediumX(gStore.getData(this).mediumX[0]),
  tightX(gStore.getData(this).tightX[0]),
  matchedGenId(gStore.getData(this).matchedGenId[0])
{
}

panda::XPhoton::~XPhoton()
{
  destructor();
}

void
panda::XPhoton::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM XPhoton.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    Photon::destructor(kTRUE);
}

panda::XPhoton&
panda::XPhoton::operator=(XPhoton const& _src)
{
  Photon::operator=(_src);

  scEta = _src.scEta;
  scRawPt = _src.scRawPt;
  std::memcpy(chIsoX, _src.chIsoX, sizeof(Float_t) * nIDTunes);
  std::memcpy(chIsoMaxX, _src.chIsoMaxX, sizeof(Float_t) * nIDTunes);
  std::memcpy(nhIsoX, _src.nhIsoX, sizeof(Float_t) * nIDTunes);
  std::memcpy(phIsoX, _src.phIsoX, sizeof(Float_t) * nIDTunes);
  e4 = _src.e4;
  isEB = _src.isEB;
  std::memcpy(looseX, _src.looseX, sizeof(Bool_t) * nIDTunes);
  std::memcpy(mediumX, _src.mediumX, sizeof(Bool_t) * nIDTunes);
  std::memcpy(tightX, _src.tightX, sizeof(Bool_t) * nIDTunes);
  matchedGenId = _src.matchedGenId;

  /* BEGIN CUSTOM XPhoton.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::XPhoton::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  Photon::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "scEta", "", 'F', &scEta, _branches);
  utils::book(_tree, _name, "scRawPt", "", 'F', &scRawPt, _branches);
  utils::book(_tree, _name, "chIsoX", TString::Format("[%d]", nIDTunes), 'F', chIsoX, _branches);
  utils::book(_tree, _name, "chIsoMaxX", TString::Format("[%d]", nIDTunes), 'F', chIsoMaxX, _branches);
  utils::book(_tree, _name, "nhIsoX", TString::Format("[%d]", nIDTunes), 'F', nhIsoX, _branches);
  utils::book(_tree, _name, "phIsoX", TString::Format("[%d]", nIDTunes), 'F', phIsoX, _branches);
  utils::book(_tree, _name, "e4", "", 'F', &e4, _branches);
  utils::book(_tree, _name, "isEB", "", 'O', &isEB, _branches);
  utils::book(_tree, _name, "looseX", TString::Format("[%d]", nIDTunes), 'O', looseX, _branches);
  utils::book(_tree, _name, "mediumX", TString::Format("[%d]", nIDTunes), 'O', mediumX, _branches);
  utils::book(_tree, _name, "tightX", TString::Format("[%d]", nIDTunes), 'O', tightX, _branches);
  utils::book(_tree, _name, "matchedGenId", "", 'I', &matchedGenId, _branches);
}

void
panda::XPhoton::doInit_()
{
  Photon::doInit_();

  scEta = 0.;
  scRawPt = 0.;
  for (auto& p0 : chIsoX) p0 = 0.;
  for (auto& p0 : chIsoMaxX) p0 = 0.;
  for (auto& p0 : nhIsoX) p0 = 0.;
  for (auto& p0 : phIsoX) p0 = 0.;
  e4 = 0.;
  isEB = false;
  for (auto& p0 : looseX) p0 = false;
  for (auto& p0 : mediumX) p0 = false;
  for (auto& p0 : tightX) p0 = false;
  matchedGenId = 0;

  /* BEGIN CUSTOM XPhoton.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::XPhoton::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM XPhoton.cc.print */
  if (_level >= 3) {
    XPhoton::dump(_out);
  }
  else if (_level == 2) {
    Photon::print(_out, _level);

    _out << "scEta = " << scEta << std::endl;
    _out << "scRawPt = " << scRawPt << std::endl;
    _out << "chIso[S15] = " << chIsoX[kSpring15] << std::endl;
    _out << "nhIso[S15] = " << nhIsoX[kSpring15] << std::endl;
    _out << "phIso[S15] = " << phIsoX[kSpring15] << std::endl;
    _out << "e4 = " << e4 << std::endl;
    _out << "isEB = " << isEB << std::endl;
  }
  else
    return;
  /* END CUSTOM */
}

void
panda::XPhoton::dump(std::ostream& _out/* = std::cout*/) const
{
  Photon::dump(_out);

  _out << "scEta = " << scEta << std::endl;
  _out << "scRawPt = " << scRawPt << std::endl;
  _out << "chIsoX = " << chIsoX << std::endl;
  _out << "chIsoMaxX = " << chIsoMaxX << std::endl;
  _out << "nhIsoX = " << nhIsoX << std::endl;
  _out << "phIsoX = " << phIsoX << std::endl;
  _out << "e4 = " << e4 << std::endl;
  _out << "isEB = " << isEB << std::endl;
  _out << "looseX = " << looseX << std::endl;
  _out << "mediumX = " << mediumX << std::endl;
  _out << "tightX = " << tightX << std::endl;
  _out << "matchedGenId = " << matchedGenId << std::endl;
}


/* BEGIN CUSTOM XPhoton.cc.global */
/* END CUSTOM */
