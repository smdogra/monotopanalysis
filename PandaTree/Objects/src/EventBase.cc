#include "../interface/EventBase.h"

panda::EventBase::EventBase() :
  TreeEntry()
{
  std::vector<Object*> myObjects{{&triggers}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());
}

panda::EventBase::EventBase(EventBase const& _src) :
  TreeEntry(_src),
  triggers(_src.triggers),
  runNumber(_src.runNumber),
  lumiNumber(_src.lumiNumber),
  eventNumber(_src.eventNumber),
  isData(_src.isData),
  weight(_src.weight)
{
  std::vector<Object*> myObjects{{&triggers}};
  objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());

  /* BEGIN CUSTOM EventBase.cc.copy_ctor */
  run = _src.run;
  /* END CUSTOM */
}

panda::EventBase::~EventBase()
{
  /* BEGIN CUSTOM EventBase.cc.dtor */
  /* END CUSTOM */
}

panda::EventBase&
panda::EventBase::operator=(EventBase const& _src)
{
  TreeEntry::operator=(_src);

  /* BEGIN CUSTOM EventBase.cc.operator= */
  run = _src.run;
  /* END CUSTOM */

  runNumber = _src.runNumber;
  lumiNumber = _src.lumiNumber;
  eventNumber = _src.eventNumber;
  isData = _src.isData;
  weight = _src.weight;

  triggers = _src.triggers;

  return *this;
}

void
panda::EventBase::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM EventBase.cc.print */
  if (_level >= 3) {
    EventBase::dump(_out);
  }
  else if (_level == 2) {
    // debug level
    _out << "runNumber = " << runNumber << std::endl;
    _out << "lumiNumber = " << lumiNumber << std::endl;
    _out << "eventNumber = " << eventNumber << std::endl;
    _out << "isData = " << isData << std::endl;
    _out << "weight = " << weight << std::endl;
  }
  else
    return;
  /* END CUSTOM */
}

void
panda::EventBase::dump(std::ostream& _out/* = std::cout*/) const
{
  _out << "runNumber = " << runNumber << std::endl;
  _out << "lumiNumber = " << lumiNumber << std::endl;
  _out << "eventNumber = " << eventNumber << std::endl;
  _out << "isData = " << isData << std::endl;
  _out << "weight = " << weight << std::endl;

  triggers.dump(_out);

}
/*static*/
panda::utils::BranchList
panda::EventBase::getListOfBranches(Bool_t _direct/* = kFALSE*/)
{
  utils::BranchList blist;
  blist += {"runNumber", "lumiNumber", "eventNumber", "isData", "weight"};
  if (!_direct) {
    blist += HLTBits::getListOfBranches().fullNames("triggers");
  }
  return blist;
}

/*protected*/
void
panda::EventBase::doSetStatus_(TTree& _tree, utils::BranchList const& _branches)
{
  utils::setStatus(_tree, "", "runNumber", _branches);
  utils::setStatus(_tree, "", "lumiNumber", _branches);
  utils::setStatus(_tree, "", "eventNumber", _branches);
  utils::setStatus(_tree, "", "isData", _branches);
  utils::setStatus(_tree, "", "weight", _branches);
}

/*protected*/
panda::utils::BranchList
panda::EventBase::doGetStatus_(TTree& _tree) const
{
  utils::BranchList blist;

  blist.push_back(utils::getStatus(_tree, "", "runNumber"));
  blist.push_back(utils::getStatus(_tree, "", "lumiNumber"));
  blist.push_back(utils::getStatus(_tree, "", "eventNumber"));
  blist.push_back(utils::getStatus(_tree, "", "isData"));
  blist.push_back(utils::getStatus(_tree, "", "weight"));
  return blist;
}

/*protected*/
panda::utils::BranchList
panda::EventBase::doGetBranchNames_() const
{
  return getListOfBranches(true);
}

/*protected*/
void
panda::EventBase::doSetAddress_(TTree& _tree, utils::BranchList const& _branches, Bool_t _setStatus)
{
  utils::setAddress(_tree, "", "runNumber", &runNumber, _branches, _setStatus);
  utils::setAddress(_tree, "", "lumiNumber", &lumiNumber, _branches, _setStatus);
  utils::setAddress(_tree, "", "eventNumber", &eventNumber, _branches, _setStatus);
  utils::setAddress(_tree, "", "isData", &isData, _branches, _setStatus);
  utils::setAddress(_tree, "", "weight", &weight, _branches, _setStatus);
}

/*protected*/
void
panda::EventBase::doBook_(TTree& _tree, utils::BranchList const& _branches)
{
  utils::book(_tree, "", "runNumber", "", 'i', &runNumber, _branches);
  utils::book(_tree, "", "lumiNumber", "", 'i', &lumiNumber, _branches);
  utils::book(_tree, "", "eventNumber", "", 'i', &eventNumber, _branches);
  utils::book(_tree, "", "isData", "", 'O', &isData, _branches);
  utils::book(_tree, "", "weight", "", 'F', &weight, _branches);
}

/*protected*/
void
panda::EventBase::doGetEntry_(TTree& _tree)
{
  /* BEGIN CUSTOM EventBase.cc.doGetEntry_ */
  if (readRunTree_ && _tree.GetCurrentFile() && runNumber != 0) {
    // If our Run object is enabled, we need to check for file transitions

    auto rItr(runTrees_.find(&_tree));
    if (rItr == runTrees_.end()) {
      // First time dealing with this tree
      rItr = runTrees_.emplace(&_tree, std::make_pair<Int_t, TTree*>(-1, 0)).first;
    }

    if (_tree.GetTreeNumber() == rItr->second.first) {
      // We are on the same tree. Just check for run number transition (and update the trigger table if necessary)
      run.findEntry(*rItr->second.second, runNumber);
    }
    else {
      // There was a file transition in the input
      rItr->second.first = _tree.GetTreeNumber();

      // First check that the runNumber branch is turned on
      if (_tree.GetBranchStatus("runNumber")) {
        auto& file(*_tree.GetCurrentFile());

        // read out the runs tree
        // using GetKey to create a "fresh" object - Get can fetch an in-memory object that may already be in use
        auto* key(file.GetKey("runs"));
        if (!key) {
          std::cerr << "File " << file.GetName() << " does not have a run tree." << std::endl;
          std::cerr << "If this is not a problem, turn off run tree synchronization" << std::endl;
          std::cerr << "by calling setReadRunTree(false) on the event object." << std::endl;
          throw std::runtime_error("InputError");
        }

        auto* runTree(static_cast<TTree*>(key->ReadObj()));

        rItr->second.second = runTree;

        run.setAddress(*runTree);
        run.init();
        run.resetCache();

        // Now cue the run object to the given run number
        run.findEntry(*rItr->second.second, runNumber);
      }
      else {
        rItr->second.second = 0;
        std::cerr << "Branch runNumber is turned off in the input events tree. Skipping run update." << std::endl;
      }
    }
  }
  /* END CUSTOM */
}

void
panda::EventBase::doInit_()
{
  runNumber = 0;
  lumiNumber = 0;
  eventNumber = 0;
  isData = false;
  weight = 0.;
  /* BEGIN CUSTOM EventBase.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::EventBase::doUnlink_(TTree& _tree)
{
  /* BEGIN CUSTOM EventBase.cc.doUnlink_ */
  runTrees_.erase(&_tree);
  /* END CUSTOM */
}


/* BEGIN CUSTOM EventBase.cc.global */

Bool_t
panda::EventBase::triggerFired(UInt_t _token) const
{
  UInt_t idx(run.getTriggerIndex(_token));
  if (idx < run.triggerSize())
    return triggers.pass(idx);
  else
    return false;
}

/* END CUSTOM */
