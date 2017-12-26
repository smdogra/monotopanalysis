#include "../interface/EventTPMG.h"

panda::EventTPMG::EventTPMG() :
  EventTP()
{
  std::vector<Object*> myObjects{{&tags, &probes}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&tags, &probes}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());
}

panda::EventTPMG::EventTPMG(EventTPMG const& _src) :
  EventTP(_src),
  tags(_src.tags),
  probes(_src.probes)
{
  std::vector<Object*> myObjects{{&tags, &probes}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&tags, &probes}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());
  /* BEGIN CUSTOM EventTPMG.cc.copy_ctor */
  /* END CUSTOM */
}

panda::EventTPMG::~EventTPMG()
{
  /* BEGIN CUSTOM EventTPMG.cc.dtor */
  /* END CUSTOM */
}

panda::EventTPMG&
panda::EventTPMG::operator=(EventTPMG const& _src)
{
  EventTP::operator=(_src);

  /* BEGIN CUSTOM EventTPMG.cc.operator= */
  /* END CUSTOM */

  tags = _src.tags;
  probes = _src.probes;

  return *this;
}

void
panda::EventTPMG::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM EventTPMG.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::EventTPMG::dump(std::ostream& _out/* = std::cout*/) const
{
  EventTP::dump(_out);

  tags.dump(_out);
  probes.dump(_out);

}
/*static*/
panda::utils::BranchList
panda::EventTPMG::getListOfBranches(Bool_t _direct/* = kFALSE*/)
{
  utils::BranchList blist;
  blist += EventTP::getListOfBranches(_direct);

  blist += {};
  if (!_direct) {
    blist += Muon::getListOfBranches().fullNames("tags");
    blist += XPhoton::getListOfBranches().fullNames("probes");
  }
  return blist;
}

/*protected*/
void
panda::EventTPMG::doSetStatus_(TTree& _tree, utils::BranchList const& _branches)
{
  EventTP::doSetStatus_(_tree, _branches);
}

/*protected*/
panda::utils::BranchList
panda::EventTPMG::doGetStatus_(TTree& _tree) const
{
  utils::BranchList blist;
  blist += EventTP::doGetStatus_(_tree);

  return blist;
}

/*protected*/
panda::utils::BranchList
panda::EventTPMG::doGetBranchNames_() const
{
  return getListOfBranches(true);
}

/*protected*/
void
panda::EventTPMG::doSetAddress_(TTree& _tree, utils::BranchList const& _branches, Bool_t _setStatus)
{
  EventTP::doSetAddress_(_tree, _branches, _setStatus);

}

/*protected*/
void
panda::EventTPMG::doBook_(TTree& _tree, utils::BranchList const& _branches)
{
  EventTP::doBook_(_tree, _branches);

}

/*protected*/
void
panda::EventTPMG::doGetEntry_(TTree& _tree)
{
  EventTP::doGetEntry_(_tree);

  /* BEGIN CUSTOM EventTPMG.cc.doGetEntry_ */
  /* END CUSTOM */
}

void
panda::EventTPMG::doInit_()
{
  EventTP::doInit_();

  /* BEGIN CUSTOM EventTPMG.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::EventTPMG::doUnlink_(TTree& _tree)
{
  EventTP::doUnlink_(_tree);

  /* BEGIN CUSTOM EventTPMG.cc.doUnlink_ */
  /* END CUSTOM */
}


/* BEGIN CUSTOM EventTPMG.cc.global */
/* END CUSTOM */
