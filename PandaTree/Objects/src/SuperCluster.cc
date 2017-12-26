#include "../interface/SuperCluster.h"

/*static*/
panda::utils::BranchList
panda::SuperCluster::getListOfBranches()
{
  utils::BranchList blist;
  blist += {"rawPt", "eta", "phi"};
  return blist;
}

void
panda::SuperCluster::datastore::allocate(UInt_t _nmax)
{
  Element::datastore::allocate(_nmax);

  rawPt = new Float_t[nmax_];
  eta = new Float_t[nmax_];
  phi = new Float_t[nmax_];
}

void
panda::SuperCluster::datastore::deallocate()
{
  Element::datastore::deallocate();

  delete [] rawPt;
  rawPt = 0;
  delete [] eta;
  eta = 0;
  delete [] phi;
  phi = 0;
}

void
panda::SuperCluster::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  Element::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "rawPt", _branches);
  utils::setStatus(_tree, _name, "eta", _branches);
  utils::setStatus(_tree, _name, "phi", _branches);
}

panda::utils::BranchList
panda::SuperCluster::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(Element::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "rawPt"));
  blist.push_back(utils::getStatus(_tree, _name, "eta"));
  blist.push_back(utils::getStatus(_tree, _name, "phi"));

  return blist;
}

void
panda::SuperCluster::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  Element::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "rawPt", rawPt, _branches, _setStatus);
  utils::setAddress(_tree, _name, "eta", eta, _branches, _setStatus);
  utils::setAddress(_tree, _name, "phi", phi, _branches, _setStatus);
}

void
panda::SuperCluster::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  Element::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "rawPt", size, 'F', rawPt, _branches);
  utils::book(_tree, _name, "eta", size, 'F', eta, _branches);
  utils::book(_tree, _name, "phi", size, 'F', phi, _branches);
}

void
panda::SuperCluster::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  Element::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "rawPt");
  utils::resetAddress(_tree, _name, "eta");
  utils::resetAddress(_tree, _name, "phi");
}

void
panda::SuperCluster::datastore::resizeVectors_(UInt_t _size)
{
  Element::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::SuperCluster::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return SuperCluster::getListOfBranches().fullNames(_name);
}

panda::SuperCluster::SuperCluster(char const* _name/* = ""*/) :
  Element(new SuperClusterArray(1, _name)),
  rawPt(gStore.getData(this).rawPt[0]),
  eta(gStore.getData(this).eta[0]),
  phi(gStore.getData(this).phi[0])
{
}

panda::SuperCluster::SuperCluster(SuperCluster const& _src) :
  Element(new SuperClusterArray(1, _src.getName())),
  rawPt(gStore.getData(this).rawPt[0]),
  eta(gStore.getData(this).eta[0]),
  phi(gStore.getData(this).phi[0])
{
  Element::operator=(_src);

  rawPt = _src.rawPt;
  eta = _src.eta;
  phi = _src.phi;
}

panda::SuperCluster::SuperCluster(datastore& _data, UInt_t _idx) :
  Element(_data, _idx),
  rawPt(_data.rawPt[_idx]),
  eta(_data.eta[_idx]),
  phi(_data.phi[_idx])
{
}

panda::SuperCluster::SuperCluster(ArrayBase* _array) :
  Element(_array),
  rawPt(gStore.getData(this).rawPt[0]),
  eta(gStore.getData(this).eta[0]),
  phi(gStore.getData(this).phi[0])
{
}

panda::SuperCluster::~SuperCluster()
{
  destructor();
}

void
panda::SuperCluster::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM SuperCluster.cc.destructor */
  /* END CUSTOM */
}

panda::SuperCluster&
panda::SuperCluster::operator=(SuperCluster const& _src)
{
  rawPt = _src.rawPt;
  eta = _src.eta;
  phi = _src.phi;

  /* BEGIN CUSTOM SuperCluster.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::SuperCluster::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  utils::book(_tree, _name, "rawPt", "", 'F', &rawPt, _branches);
  utils::book(_tree, _name, "eta", "", 'F', &eta, _branches);
  utils::book(_tree, _name, "phi", "", 'F', &phi, _branches);
}

void
panda::SuperCluster::doInit_()
{
  rawPt = 0.;
  eta = 0.;
  phi = 0.;

  /* BEGIN CUSTOM SuperCluster.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::SuperCluster::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM SuperCluster.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::SuperCluster::dump(std::ostream& _out/* = std::cout*/) const
{
  _out << "rawPt = " << rawPt << std::endl;
  _out << "eta = " << eta << std::endl;
  _out << "phi = " << phi << std::endl;
}

/* BEGIN CUSTOM SuperCluster.cc.global */
/* END CUSTOM */
