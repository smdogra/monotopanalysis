#include "../interface/EventTPMMG.h"

panda::EventTPMMG::EventTPMMG() :
  EventTPMG()
{
  std::vector<Object*> myObjects{{&looseTags}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&looseTags}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());
}

panda::EventTPMMG::EventTPMMG(EventTPMMG const& _src) :
  EventTPMG(_src),
  looseTags(_src.looseTags)
{
  std::vector<Object*> myObjects{{&looseTags}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&looseTags}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());
  /* BEGIN CUSTOM EventTPMMG.cc.copy_ctor */
  /* END CUSTOM */
}

panda::EventTPMMG::~EventTPMMG()
{
  /* BEGIN CUSTOM EventTPMMG.cc.dtor */
  /* END CUSTOM */
}

panda::EventTPMMG&
panda::EventTPMMG::operator=(EventTPMMG const& _src)
{
  EventTPMG::operator=(_src);

  /* BEGIN CUSTOM EventTPMMG.cc.operator= */
  /* END CUSTOM */

  looseTags = _src.looseTags;

  return *this;
}

void
panda::EventTPMMG::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM EventTPMMG.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::EventTPMMG::dump(std::ostream& _out/* = std::cout*/) const
{
  EventTPMG::dump(_out);

  looseTags.dump(_out);

}
/*static*/
panda::utils::BranchList
panda::EventTPMMG::getListOfBranches(Bool_t _direct/* = kFALSE*/)
{
  utils::BranchList blist;
  blist += EventTPMG::getListOfBranches(_direct);

  blist += {};
  if (!_direct) {
    blist += Muon::getListOfBranches().fullNames("looseTags");
  }
  return blist;
}

/*protected*/
void
panda::EventTPMMG::doSetStatus_(TTree& _tree, utils::BranchList const& _branches)
{
  EventTPMG::doSetStatus_(_tree, _branches);
}

/*protected*/
panda::utils::BranchList
panda::EventTPMMG::doGetStatus_(TTree& _tree) const
{
  utils::BranchList blist;
  blist += EventTPMG::doGetStatus_(_tree);

  return blist;
}

/*protected*/
panda::utils::BranchList
panda::EventTPMMG::doGetBranchNames_() const
{
  return getListOfBranches(true);
}

/*protected*/
void
panda::EventTPMMG::doSetAddress_(TTree& _tree, utils::BranchList const& _branches, Bool_t _setStatus)
{
  EventTPMG::doSetAddress_(_tree, _branches, _setStatus);

}

/*protected*/
void
panda::EventTPMMG::doBook_(TTree& _tree, utils::BranchList const& _branches)
{
  EventTPMG::doBook_(_tree, _branches);

}

/*protected*/
void
panda::EventTPMMG::doGetEntry_(TTree& _tree)
{
  EventTPMG::doGetEntry_(_tree);

  /* BEGIN CUSTOM EventTPMMG.cc.doGetEntry_ */
  /* END CUSTOM */
}

void
panda::EventTPMMG::doInit_()
{
  EventTPMG::doInit_();

  /* BEGIN CUSTOM EventTPMMG.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::EventTPMMG::doUnlink_(TTree& _tree)
{
  EventTPMG::doUnlink_(_tree);

  /* BEGIN CUSTOM EventTPMMG.cc.doUnlink_ */
  /* END CUSTOM */
}


/* BEGIN CUSTOM EventTPMMG.cc.global */
/* END CUSTOM */
