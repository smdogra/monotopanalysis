#include "../interface/EventTP2M.h"

panda::EventTP2M::EventTP2M() :
  EventTP()
{
  std::vector<Object*> myObjects{{&tags, &probes}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&tags, &probes}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());
}

panda::EventTP2M::EventTP2M(EventTP2M const& _src) :
  EventTP(_src),
  tags(_src.tags),
  probes(_src.probes)
{
  std::vector<Object*> myObjects{{&tags, &probes}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
  std::vector<CollectionBase*> myCollections{{&tags, &probes}};
  collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());
  /* BEGIN CUSTOM EventTP2M.cc.copy_ctor */
  /* END CUSTOM */
}

panda::EventTP2M::~EventTP2M()
{
  /* BEGIN CUSTOM EventTP2M.cc.dtor */
  /* END CUSTOM */
}

panda::EventTP2M&
panda::EventTP2M::operator=(EventTP2M const& _src)
{
  EventTP::operator=(_src);

  /* BEGIN CUSTOM EventTP2M.cc.operator= */
  /* END CUSTOM */

  tags = _src.tags;
  probes = _src.probes;

  return *this;
}

void
panda::EventTP2M::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM EventTP2M.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::EventTP2M::dump(std::ostream& _out/* = std::cout*/) const
{
  EventTP::dump(_out);

  tags.dump(_out);
  probes.dump(_out);

}
/*static*/
panda::utils::BranchList
panda::EventTP2M::getListOfBranches(Bool_t _direct/* = kFALSE*/)
{
  utils::BranchList blist;
  blist += EventTP::getListOfBranches(_direct);

  blist += {};
  if (!_direct) {
    blist += Muon::getListOfBranches().fullNames("tags");
    blist += Muon::getListOfBranches().fullNames("probes");
  }
  return blist;
}

/*protected*/
void
panda::EventTP2M::doSetStatus_(TTree& _tree, utils::BranchList const& _branches)
{
  EventTP::doSetStatus_(_tree, _branches);
}

/*protected*/
panda::utils::BranchList
panda::EventTP2M::doGetStatus_(TTree& _tree) const
{
  utils::BranchList blist;
  blist += EventTP::doGetStatus_(_tree);

  return blist;
}

/*protected*/
panda::utils::BranchList
panda::EventTP2M::doGetBranchNames_() const
{
  return getListOfBranches(true);
}

/*protected*/
void
panda::EventTP2M::doSetAddress_(TTree& _tree, utils::BranchList const& _branches, Bool_t _setStatus)
{
  EventTP::doSetAddress_(_tree, _branches, _setStatus);

}

/*protected*/
void
panda::EventTP2M::doBook_(TTree& _tree, utils::BranchList const& _branches)
{
  EventTP::doBook_(_tree, _branches);

}

/*protected*/
void
panda::EventTP2M::doGetEntry_(TTree& _tree)
{
  EventTP::doGetEntry_(_tree);

  /* BEGIN CUSTOM EventTP2M.cc.doGetEntry_ */
  /* END CUSTOM */
}

void
panda::EventTP2M::doInit_()
{
  EventTP::doInit_();

  /* BEGIN CUSTOM EventTP2M.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::EventTP2M::doUnlink_(TTree& _tree)
{
  EventTP::doUnlink_(_tree);

  /* BEGIN CUSTOM EventTP2M.cc.doUnlink_ */
  /* END CUSTOM */
}


/* BEGIN CUSTOM EventTP2M.cc.global */
/* END CUSTOM */
