#include "../interface/HLTBits.h"

/*static*/
panda::utils::BranchList
panda::HLTBits::getListOfBranches()
{
  utils::BranchList blist;
  blist += {"words"};
  return blist;
}

panda::HLTBits::HLTBits(char const* _name/* = ""*/) :
  Singlet(_name)
{
}

panda::HLTBits::HLTBits(HLTBits const& _src) :
  Singlet(_src.name_)
{
  std::memcpy(words, _src.words, sizeof(UInt_t) * 32);
}

panda::HLTBits::~HLTBits()
{
}

panda::HLTBits&
panda::HLTBits::operator=(HLTBits const& _src)
{
  std::memcpy(words, _src.words, sizeof(UInt_t) * 32);

  /* BEGIN CUSTOM HLTBits.cc.operator= */
  /* END CUSTOM */

  return *this;
}

void
panda::HLTBits::doSetStatus_(TTree& _tree, utils::BranchList const& _branches)
{
  utils::setStatus(_tree, name_, "words", _branches);
}

panda::utils::BranchList
panda::HLTBits::doGetStatus_(TTree& _tree) const
{
  utils::BranchList blist;

  blist.push_back(utils::getStatus(_tree, name_, "words"));

  return blist;
}

void
panda::HLTBits::doSetAddress_(TTree& _tree, utils::BranchList const& _branches/* = {"*"}*/, Bool_t _setStatus/* = kTRUE*/)
{
  utils::setAddress(_tree, name_, "words", words, _branches, _setStatus);
}

void
panda::HLTBits::doBook_(TTree& _tree, utils::BranchList const& _branches/* = {"*"}*/)
{
  utils::book(_tree, name_, "words", TString::Format("[32]"), 'i', words, _branches);
}

void
panda::HLTBits::doInit_()
{
  for (auto& p0 : words) p0 = 0;

  /* BEGIN CUSTOM HLTBits.cc.doInit_ */
  /* END CUSTOM */
}

panda::utils::BranchList
panda::HLTBits::doGetBranchNames_(Bool_t _fullName) const
{
  if (_fullName)
    return getListOfBranches().fullNames(name_);
  else
    return getListOfBranches().fullNames();
}

void
panda::HLTBits::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM HLTBits.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::HLTBits::dump(std::ostream& _out/* = std::cout*/) const
{
  _out << "words = " << words << std::endl;
}


/* BEGIN CUSTOM HLTBits.cc.global */
/* END CUSTOM */
