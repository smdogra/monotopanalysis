#include "../interface/Vertex.h"

/*static*/
panda::utils::BranchList
panda::Vertex::getListOfBranches()
{
  utils::BranchList blist;
  blist += {"x", "y", "z"};
  return blist;
}

void
panda::Vertex::datastore::allocate(UInt_t _nmax)
{
  Element::datastore::allocate(_nmax);

  x = new Float_t[nmax_];
  y = new Float_t[nmax_];
  z = new Float_t[nmax_];
}

void
panda::Vertex::datastore::deallocate()
{
  Element::datastore::deallocate();

  delete [] x;
  x = 0;
  delete [] y;
  y = 0;
  delete [] z;
  z = 0;
}

void
panda::Vertex::datastore::setStatus(TTree& _tree, TString const& _name, utils::BranchList const& _branches)
{
  Element::datastore::setStatus(_tree, _name, _branches);

  utils::setStatus(_tree, _name, "x", _branches);
  utils::setStatus(_tree, _name, "y", _branches);
  utils::setStatus(_tree, _name, "z", _branches);
}

panda::utils::BranchList
panda::Vertex::datastore::getStatus(TTree& _tree, TString const& _name) const
{
  utils::BranchList blist(Element::datastore::getStatus(_tree, _name));

  blist.push_back(utils::getStatus(_tree, _name, "x"));
  blist.push_back(utils::getStatus(_tree, _name, "y"));
  blist.push_back(utils::getStatus(_tree, _name, "z"));

  return blist;
}

void
panda::Vertex::datastore::setAddress(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  Element::datastore::setAddress(_tree, _name, _branches, _setStatus);

  utils::setAddress(_tree, _name, "x", x, _branches, _setStatus);
  utils::setAddress(_tree, _name, "y", y, _branches, _setStatus);
  utils::setAddress(_tree, _name, "z", z, _branches, _setStatus);
}

void
panda::Vertex::datastore::book(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _dynamic/* = kTRUE*/)
{
  Element::datastore::book(_tree, _name, _branches, _dynamic);

  TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));

  utils::book(_tree, _name, "x", size, 'F', x, _branches);
  utils::book(_tree, _name, "y", size, 'F', y, _branches);
  utils::book(_tree, _name, "z", size, 'F', z, _branches);
}

void
panda::Vertex::datastore::releaseTree(TTree& _tree, TString const& _name)
{
  Element::datastore::releaseTree(_tree, _name);

  utils::resetAddress(_tree, _name, "x");
  utils::resetAddress(_tree, _name, "y");
  utils::resetAddress(_tree, _name, "z");
}

void
panda::Vertex::datastore::resizeVectors_(UInt_t _size)
{
  Element::datastore::resizeVectors_(_size);

}


panda::utils::BranchList
panda::Vertex::datastore::getBranchNames(TString const& _name/* = ""*/) const
{
  return Vertex::getListOfBranches().fullNames(_name);
}

panda::Vertex::Vertex(char const* _name/* = ""*/) :
  Element(new VertexArray(1, _name)),
  x(gStore.getData(this).x[0]),
  y(gStore.getData(this).y[0]),
  z(gStore.getData(this).z[0])
{
}

panda::Vertex::Vertex(Vertex const& _src) :
  Element(new VertexArray(1, _src.getName())),
  x(gStore.getData(this).x[0]),
  y(gStore.getData(this).y[0]),
  z(gStore.getData(this).z[0])
{
  Element::operator=(_src);

  x = _src.x;
  y = _src.y;
  z = _src.z;
}

panda::Vertex::Vertex(datastore& _data, UInt_t _idx) :
  Element(_data, _idx),
  x(_data.x[_idx]),
  y(_data.y[_idx]),
  z(_data.z[_idx])
{
}

panda::Vertex::Vertex(ArrayBase* _array) :
  Element(_array),
  x(gStore.getData(this).x[0]),
  y(gStore.getData(this).y[0]),
  z(gStore.getData(this).z[0])
{
}

panda::Vertex::~Vertex()
{
  destructor();
}

void
panda::Vertex::destructor(Bool_t _recursive/* = kFALSE*/)
{
  /* BEGIN CUSTOM Vertex.cc.destructor */
  /* END CUSTOM */
}

panda::Vertex&
panda::Vertex::operator=(Vertex const& _src)
{
  x = _src.x;
  y = _src.y;
  z = _src.z;

  /* BEGIN CUSTOM Vertex.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::Vertex::doBook_(TTree& _tree, TString const& _name, utils::BranchList const& _branches/* = {"*"}*/)
{
  utils::book(_tree, _name, "x", "", 'F', &x, _branches);
  utils::book(_tree, _name, "y", "", 'F', &y, _branches);
  utils::book(_tree, _name, "z", "", 'F', &z, _branches);
}

void
panda::Vertex::doInit_()
{
  x = 0.;
  y = 0.;
  z = 0.;

  /* BEGIN CUSTOM Vertex.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::Vertex::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM Vertex.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::Vertex::dump(std::ostream& _out/* = std::cout*/) const
{
  _out << "x = " << x << std::endl;
  _out << "y = " << y << std::endl;
  _out << "z = " << z << std::endl;
}


/* BEGIN CUSTOM Vertex.cc.global */
/* END CUSTOM */
