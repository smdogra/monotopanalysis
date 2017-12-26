#include "../interface/RecoVertex.h"

/*static*/
panda::utils::BranchList
panda::RecoVertex::getListOfBranches()
{
  utils::BranchList blist;
  blist += Vertex::getListOfBranches();
  blist += {"score", "ntrk", "ndof", "chi2", "pfRangeMax"};
  return blist;
}

void
panda::RecoVertex::datastore::allocate(UInt_t _nmax)
{
  Vertex::datastore::allocate(_nmax);

  score = new Float_t[nmax_];
  ntrk = new UShort_t[nmax_];
  ndof = new Float_t[nmax_];
  chi2 = new Float_t[nmax_];
  pfRangeMax = new UShort_t[nmax_];
}

void
panda::RecoVertex::datastore::deallocate()
{
  Vertex::datastore::deallocate();

  delete [] score;
  score = 0;
  delete [] ntrk;
  ntrk = 0;
  delete [] ndof;
  ndof = 0;
  delete [] chi2;
  chi2 = 0;
  delete [] pfRangeMax;
  pfRangeMax = 0;
}

void
panda::RecoVertex::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  Vertex::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "score", _branches);
  utils::setStatus(_tree, _name, "ntrk", _branches);
  utils::setStatus(_tree, _name, "ndof", _branches);
  utils::setStatus(_tree, _name, "chi2", _branches);
  utils::setStatus(_tree, _name, "pfRangeMax", _branches);
}

panda::utils::BranchList
panda::RecoVertex::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(Vertex::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "score"));
  blist.push_back(utils::getStatus(_tree, _name, "ntrk"));
  blist.push_back(utils::getStatus(_tree, _name, "ndof"));
  blist.push_back(utils::getStatus(_tree, _name, "chi2"));
  blist.push_back(utils::getStatus(_tree, _name, "pfRangeMax"));

  return blist;
}

void
panda::RecoVertex::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  Vertex::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "score", score, _branches, _setStatus);
  utils::setAddress(_tree, _name, "ntrk", ntrk, _branches, _setStatus);
  utils::setAddress(_tree, _name, "ndof", ndof, _branches, _setStatus);
  utils::setAddress(_tree, _name, "chi2", chi2, _branches, _setStatus);
  utils::setAddress(_tree, _name, "pfRangeMax", pfRangeMax, _branches, _setStatus);
}

void
panda::RecoVertex::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  Vertex::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "score", size, 'F', score, _branches);
  utils::book(_tree, _name, "ntrk", size, 's', ntrk, _branches);
  utils::book(_tree, _name, "ndof", size, 'F', ndof, _branches);
  utils::book(_tree, _name, "chi2", size, 'F', chi2, _branches);
  utils::book(_tree, _name, "pfRangeMax", size, 's', pfRangeMax, _branches);
}

void
panda::RecoVertex::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  Vertex::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "score");
  utils::resetAddress(_tree, _name, "ntrk");
  utils::resetAddress(_tree, _name, "ndof");
  utils::resetAddress(_tree, _name, "chi2");
  utils::resetAddress(_tree, _name, "pfRangeMax");
}

void
panda::RecoVertex::datastore::resizeVectors_(UInt_t _size)
{
  Vertex::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::RecoVertex::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return RecoVertex::getListOfBranches().fullNames(_name);
}

panda::RecoVertex::RecoVertex(char const* _name/* = ""*/) :
  Vertex(new RecoVertexArray(1, _name)),
  score(gStore.getData(this).score[0]),
  ntrk(gStore.getData(this).ntrk[0]),
  ndof(gStore.getData(this).ndof[0]),
  chi2(gStore.getData(this).chi2[0]),
  pfRangeMax(gStore.getData(this).pfRangeMax[0])
{
}

panda::RecoVertex::RecoVertex(RecoVertex const& _src) :
  Vertex(new RecoVertexArray(1, _src.getName())),
  score(gStore.getData(this).score[0]),
  ntrk(gStore.getData(this).ntrk[0]),
  ndof(gStore.getData(this).ndof[0]),
  chi2(gStore.getData(this).chi2[0]),
  pfRangeMax(gStore.getData(this).pfRangeMax[0])
{
  Vertex::operator=(_src);

  score = _src.score;
  ntrk = _src.ntrk;
  ndof = _src.ndof;
  chi2 = _src.chi2;
  pfRangeMax = _src.pfRangeMax;
}

panda::RecoVertex::RecoVertex(datastore& _data, UInt_t _idx) :
  Vertex(_data, _idx),
  score(_data.score[_idx]),
  ntrk(_data.ntrk[_idx]),
  ndof(_data.ndof[_idx]),
  chi2(_data.chi2[_idx]),
  pfRangeMax(_data.pfRangeMax[_idx])
{
}

panda::RecoVertex::RecoVertex(ArrayBase* _array) :
  Vertex(_array),
  score(gStore.getData(this).score[0]),
  ntrk(gStore.getData(this).ntrk[0]),
  ndof(gStore.getData(this).ndof[0]),
  chi2(gStore.getData(this).chi2[0]),
  pfRangeMax(gStore.getData(this).pfRangeMax[0])
{
}

panda::RecoVertex::~RecoVertex()
{
  destructor();
}

void
panda::RecoVertex::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM RecoVertex.cc.destructor */
  /* END CUSTOM */

  if (_recursive)
    Vertex::destructor(kTRUE);
}

panda::RecoVertex&
panda::RecoVertex::operator=(RecoVertex const& _src)
{
  Vertex::operator=(_src);

  score = _src.score;
  ntrk = _src.ntrk;
  ndof = _src.ndof;
  chi2 = _src.chi2;
  pfRangeMax = _src.pfRangeMax;

  /* BEGIN CUSTOM RecoVertex.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::RecoVertex::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  Vertex::doBook_(_tree, _name, _branches);

  utils::book(_tree, _name, "score", "", 'F', &score, _branches);
  utils::book(_tree, _name, "ntrk", "", 's', &ntrk, _branches);
  utils::book(_tree, _name, "ndof", "", 'F', &ndof, _branches);
  utils::book(_tree, _name, "chi2", "", 'F', &chi2, _branches);
  utils::book(_tree, _name, "pfRangeMax", "", 's', &pfRangeMax, _branches);
}

void
panda::RecoVertex::doInit_()
{
  Vertex::doInit_();

  score = 0.;
  ntrk = 0;
  ndof = 0.;
  chi2 = 0.;
  pfRangeMax = 0;

  /* BEGIN CUSTOM RecoVertex.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::RecoVertex::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM RecoVertex.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::RecoVertex::dump(std::ostream& _out/* = std::cout*/) const
{
  Vertex::dump(_out);

  _out << "score = " << score << std::endl;
  _out << "ntrk = " << ntrk << std::endl;
  _out << "ndof = " << ndof << std::endl;
  _out << "chi2 = " << chi2 << std::endl;
  _out << "pfRangeMax = " << pfRangeMax << std::endl;
}


/* BEGIN CUSTOM RecoVertex.cc.global */
/* END CUSTOM */
