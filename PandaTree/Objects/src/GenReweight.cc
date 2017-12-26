#include "../interface/GenReweight.h"


/*static*/
panda::utils::BranchList
panda::GenReweight::getListOfBranches()
{
  utils::BranchList blist;
  blist += {"r1f2DW", "r1f5DW", "r2f1DW", "r2f2DW", "r5f1DW", "r5f5DW", "pdfDW", "genParam"};
  return blist;
}

panda::GenReweight::GenReweight(char const* _name/* = ""*/) :
  Singlet(_name)
{
}

panda::GenReweight::GenReweight(GenReweight const& _src) :
  Singlet(_src.name_),
  r1f2DW(_src.r1f2DW),
  r1f5DW(_src.r1f5DW),
  r2f1DW(_src.r2f1DW),
  r2f2DW(_src.r2f2DW),
  r5f1DW(_src.r5f1DW),
  r5f5DW(_src.r5f5DW),
  pdfDW(_src.pdfDW)
{
  r1f2DW = _src.r1f2DW;
  r1f5DW = _src.r1f5DW;
  r2f1DW = _src.r2f1DW;
  r2f2DW = _src.r2f2DW;
  r5f1DW = _src.r5f1DW;
  r5f5DW = _src.r5f5DW;
  pdfDW = _src.pdfDW;
  std::memcpy(genParam, _src.genParam, sizeof(Float_t) * NMAX);
}

panda::GenReweight::~GenReweight()
{
}

panda::GenReweight&
panda::GenReweight::operator=(GenReweight const& _src)
{
  r1f2DW = _src.r1f2DW;
  r1f5DW = _src.r1f5DW;
  r2f1DW = _src.r2f1DW;
  r2f2DW = _src.r2f2DW;
  r5f1DW = _src.r5f1DW;
  r5f5DW = _src.r5f5DW;
  pdfDW = _src.pdfDW;
  std::memcpy(genParam, _src.genParam, sizeof(Float_t) * NMAX);

  /* BEGIN CUSTOM GenReweight.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::GenReweight::doSetStatus_(TTree& _tree, utils::BranchList const& _branches)
{
  utils::setStatus(_tree, name_, "r1f2DW", _branches);
  utils::setStatus(_tree, name_, "r1f5DW", _branches);
  utils::setStatus(_tree, name_, "r2f1DW", _branches);
  utils::setStatus(_tree, name_, "r2f2DW", _branches);
  utils::setStatus(_tree, name_, "r5f1DW", _branches);
  utils::setStatus(_tree, name_, "r5f5DW", _branches);
  utils::setStatus(_tree, name_, "pdfDW", _branches);
  utils::setStatus(_tree, name_, "genParam", _branches);
}

panda::utils::BranchList
panda::GenReweight::doGetStatus_(TTree& _tree) const
{
  utils::BranchList blist;

  blist.push_back(utils::getStatus(_tree, name_, "r1f2DW"));
  blist.push_back(utils::getStatus(_tree, name_, "r1f5DW"));
  blist.push_back(utils::getStatus(_tree, name_, "r2f1DW"));
  blist.push_back(utils::getStatus(_tree, name_, "r2f2DW"));
  blist.push_back(utils::getStatus(_tree, name_, "r5f1DW"));
  blist.push_back(utils::getStatus(_tree, name_, "r5f5DW"));
  blist.push_back(utils::getStatus(_tree, name_, "pdfDW"));
  blist.push_back(utils::getStatus(_tree, name_, "genParam"));

  return blist;
}

void
panda::GenReweight::doSetAddress_(TTree& _tree, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  utils::setAddress(_tree, name_, "r1f2DW", &r1f2DW, _branches, _setStatus);
  utils::setAddress(_tree, name_, "r1f5DW", &r1f5DW, _branches, _setStatus);
  utils::setAddress(_tree, name_, "r2f1DW", &r2f1DW, _branches, _setStatus);
  utils::setAddress(_tree, name_, "r2f2DW", &r2f2DW, _branches, _setStatus);
  utils::setAddress(_tree, name_, "r5f1DW", &r5f1DW, _branches, _setStatus);
  utils::setAddress(_tree, name_, "r5f5DW", &r5f5DW, _branches, _setStatus);
  utils::setAddress(_tree, name_, "pdfDW", &pdfDW, _branches, _setStatus);
  utils::setAddress(_tree, name_, "genParam", genParam, _branches, _setStatus);
}

void
panda::GenReweight::doBook_(TTree& _tree, utils::BranchList const& _branches/* = {"*"}*/)
{
  utils::book(_tree, name_, "r1f2DW", "", 'F', &r1f2DW, _branches);
  utils::book(_tree, name_, "r1f5DW", "", 'F', &r1f5DW, _branches);
  utils::book(_tree, name_, "r2f1DW", "", 'F', &r2f1DW, _branches);
  utils::book(_tree, name_, "r2f2DW", "", 'F', &r2f2DW, _branches);
  utils::book(_tree, name_, "r5f1DW", "", 'F', &r5f1DW, _branches);
  utils::book(_tree, name_, "r5f5DW", "", 'F', &r5f5DW, _branches);
  utils::book(_tree, name_, "pdfDW", "", 'F', &pdfDW, _branches);
  utils::book(_tree, name_, "genParam", TString::Format("[%d]", NMAX), 'F', genParam, _branches);
}

void
panda::GenReweight::doInit_()
{
  r1f2DW = 0.;
  r1f5DW = 0.;
  r2f1DW = 0.;
  r2f2DW = 0.;
  r5f1DW = 0.;
  r5f5DW = 0.;
  pdfDW = 0.;
  for (auto& p0 : genParam) p0 = 0.;

  /* BEGIN CUSTOM GenReweight.cc.doInit_ */
  /* END CUSTOM */
}

panda::utils::BranchList
panda::GenReweight::doGetBranchNames_(Bool_t _fullName) const
{
  if (_fullName)
    return getListOfBranches().fullNames(name_);
  else
    return getListOfBranches().fullNames();
}

void
panda::GenReweight::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM GenReweight.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::GenReweight::dump(std::ostream& _out/* = std::cout*/) const
{
  _out << "r1f2DW = " << r1f2DW << std::endl;
  _out << "r1f5DW = " << r1f5DW << std::endl;
  _out << "r2f1DW = " << r2f1DW << std::endl;
  _out << "r2f2DW = " << r2f2DW << std::endl;
  _out << "r5f1DW = " << r5f1DW << std::endl;
  _out << "r5f5DW = " << r5f5DW << std::endl;
  _out << "pdfDW = " << pdfDW << std::endl;
  _out << "genParam = " << genParam << std::endl;
}

/* BEGIN CUSTOM GenReweight.cc.global */
/* END CUSTOM */
