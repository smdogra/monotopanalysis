#include "../interface/EventTPEEG.h"

panda::EventTPEEG::EventTPEEG() :
  EventTPEG()
{
  std::vector<Object*> myObjects{{&looseTags}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&looseTags}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());
}

panda::EventTPEEG::EventTPEEG(EventTPEEG const& _src) :
  EventTPEG(_src),
  looseTags(_src.looseTags)
{
  std::vector<Object*> myObjects{{&looseTags}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&looseTags}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());
  /* BEGIN CUSTOM EventTPEEG.cc.copy_ctor */
  /* END CUSTOM */
}

panda::EventTPEEG::~EventTPEEG()
{
  /* BEGIN CUSTOM EventTPEEG.cc.dtor */
  /* END CUSTOM */
}

panda::EventTPEEG&
panda::EventTPEEG::operator=(EventTPEEG const& _src)
{
  EventTPEG::operator=(_src);

  /* BEGIN CUSTOM EventTPEEG.cc.operator= */
  /* END CUSTOM */

  looseTags = _src.looseTags;

  return *this;
}

void
panda::EventTPEEG::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM EventTPEEG.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::EventTPEEG::dump(std::ostream& _out/* = std::cout*/) const
{
  EventTPEG::dump(_out);

  looseTags.dump(_out);

}
/*static*/
panda::utils::BranchList
panda::EventTPEEG::getListOfBranches(Bool_t _direct/* = kFALSE*/)
{
  utils::BranchList blist;
  blist += EventTPEG::getListOfBranches(_direct);

  blist += {};
  if (!_direct) {
    blist += Electron::getListOfBranches().fullNames("looseTags");
  }
  return blist;
}

/*protected*/
void
panda::EventTPEEG::doSetStatus_(TTree& _tree, utils::BranchList const& _branches)
{
  EventTPEG::doSetStatus_(_tree, _branches);
}

/*protected*/
panda::utils::BranchList
panda::EventTPEEG::doGetStatus_(TTree& _tree) const
{
  utils::BranchList blist;
  blist += EventTPEG::doGetStatus_(_tree);

  return blist;
}

/*protected*/
panda::utils::BranchList
panda::EventTPEEG::doGetBranchNames_() const
{
  return getListOfBranches(true);
}

/*protected*/
void
panda::EventTPEEG::doSetAddress_(TTree& _tree, utils::BranchList const& _branches, Bool_t _setStatus)
{
  EventTPEG::doSetAddress_(_tree, _branches, _setStatus);

}

/*protected*/
void
panda::EventTPEEG::doBook_(TTree& _tree, utils::BranchList const& _branches)
{
  EventTPEG::doBook_(_tree, _branches);

}

/*protected*/
void
panda::EventTPEEG::doGetEntry_(TTree& _tree)
{
  EventTPEG::doGetEntry_(_tree);

  /* BEGIN CUSTOM EventTPEEG.cc.doGetEntry_ */
  /* END CUSTOM */
}

void
panda::EventTPEEG::doInit_()
{
  EventTPEG::doInit_();

  /* BEGIN CUSTOM EventTPEEG.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::EventTPEEG::doUnlink_(TTree& _tree)
{
  EventTPEG::doUnlink_(_tree);

  /* BEGIN CUSTOM EventTPEEG.cc.doUnlink_ */
  /* END CUSTOM */
}


/* BEGIN CUSTOM EventTPEEG.cc.global */
/* END CUSTOM */
