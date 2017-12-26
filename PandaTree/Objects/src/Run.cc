#include "../interface/Run.h"

panda::Run::Run() :
  TreeEntry()
{
}

panda::Run::Run(Run const& _src) :
  TreeEntry(_src),
  runNumber(_src.runNumber),
  hltMenu(_src.hltMenu),
  hltSize(_src.hltSize)
{

  /* BEGIN CUSTOM Run.cc.copy_ctor */
  loadTrigger_ = _src.loadTrigger_;
  hltMenuCache_ = _src.hltMenuCache_;
  registeredTriggers_ = _src.registeredTriggers_;
  triggerIndices_ = _src.triggerIndices_;

  if (_src.hlt.menu) {
    hlt.create();
    *hlt.menu = *_src.hlt.menu;
    hlt.paths->assign(_src.hlt.paths->begin(), _src.hlt.paths->end());
  }

  inputTree_ = 0;
  inputTreeNumber_ = -1;
  /* END CUSTOM */
}

panda::Run::~Run()
{
  /* BEGIN CUSTOM Run.cc.dtor */
  resetCache();
  /* END CUSTOM */
}

panda::Run&
panda::Run::operator=(Run const& _src)
{
  TreeEntry::operator=(_src);

  /* BEGIN CUSTOM Run.cc.operator= */
  loadTrigger_ = _src.loadTrigger_;
  hltMenuCache_ = _src.hltMenuCache_;
  registeredTriggers_ = _src.registeredTriggers_;
  triggerIndices_ = _src.triggerIndices_;

  if (!hlt.menu && _src.hlt.menu)
    hlt.create();

  if (hlt.menu && *_src.hlt.menu != *hlt.menu) {
    *hlt.menu = *_src.hlt.menu;
    hlt.paths->assign(_src.hlt.paths->begin(), _src.hlt.paths->end());
  }

  inputTree_ = 0;
  inputTreeNumber_ = -1;
  /* END CUSTOM */

  runNumber = _src.runNumber;
  hltMenu = _src.hltMenu;
  hltSize = _src.hltSize;

  return *this;
}

void
panda::Run::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const
{
  /* BEGIN CUSTOM Run.cc.print */
  dump(_out);
  /* END CUSTOM */
}

void
panda::Run::dump(std::ostream& _out/* = std::cout*/) const
{
  _out << "runNumber = " << runNumber << std::endl;
  _out << "hltMenu = " << hltMenu << std::endl;
  _out << "hltSize = " << hltSize << std::endl;

}
/*static*/
panda::utils::BranchList
panda::Run::getListOfBranches(Bool_t _direct/* = kFALSE*/)
{
  utils::BranchList blist;
  blist += {"runNumber", "hltMenu"};
  return blist;
}

/*protected*/
void
panda::Run::doSetStatus_(TTree& _tree, utils::BranchList const& _branches)
{
  utils::setStatus(_tree, "", "runNumber", _branches);
  utils::setStatus(_tree, "", "hltMenu", _branches);
}

/*protected*/
panda::utils::BranchList
panda::Run::doGetStatus_(TTree& _tree) const
{
  utils::BranchList blist;

  blist.push_back(utils::getStatus(_tree, "", "runNumber"));
  blist.push_back(utils::getStatus(_tree, "", "hltMenu"));
  return blist;
}

/*protected*/
panda::utils::BranchList
panda::Run::doGetBranchNames_() const
{
  return getListOfBranches(true);
}

/*protected*/
void
panda::Run::doSetAddress_(TTree& _tree, utils::BranchList const& _branches, Bool_t _setStatus)
{
  utils::setAddress(_tree, "", "runNumber", &runNumber, _branches, _setStatus);
  utils::setAddress(_tree, "", "hltMenu", &hltMenu, _branches, _setStatus);
}

/*protected*/
void
panda::Run::doBook_(TTree& _tree, utils::BranchList const& _branches)
{
  utils::book(_tree, "", "runNumber", "", 'i', &runNumber, _branches);
  utils::book(_tree, "", "hltMenu", "", 'i', &hltMenu, _branches);
}

/*protected*/
void
panda::Run::doGetEntry_(TTree& _tree)
{
  /* BEGIN CUSTOM Run.cc.doGetEntry_ */
  if (loadTrigger_)
    updateTriggerTable_(_tree);
  /* END CUSTOM */
}

void
panda::Run::doInit_()
{
  runNumber = 0;
  hltMenu = 0;
  hltSize = 0;
  /* BEGIN CUSTOM Run.cc.doInit_ */
  /* END CUSTOM */
}

void
panda::Run::doUnlink_(TTree& _tree)
{
  /* BEGIN CUSTOM Run.cc.doUnlink_ */
  if (inputTree_ == &_tree) {
    inputTree_ = 0;
    inputTreeNumber_ = -1;
  }
  /* END CUSTOM */
}


/* BEGIN CUSTOM Run.cc.global */

#include "TFile.h"
#include "TKey.h"

void
panda::Run::setLoadTrigger(Bool_t _l/* = kTRUE*/)
{
  loadTrigger_ = _l;
  if (!_l) {
    hlt.destroy();
    hltMenuCache_ = -1;
  }
}

UInt_t
panda::Run::registerTrigger(char const* _path)
{
  loadTrigger_ = true;

  TString path(_path);
  path += "_v";

  auto itr(std::find(registeredTriggers_.begin(), registeredTriggers_.end(), path));
  if (itr == registeredTriggers_.end()) {
    registeredTriggers_.push_back(path);

    // need to update the input
    hlt.destroy();
    hltMenuCache_ = -1;

    return registeredTriggers_.size() - 1;
  }
  else
    return itr -registeredTriggers_.begin();
}

char const*
panda::Run::getRegisteredPath(UInt_t _token) const
{
  if (_token < registeredTriggers_.size()) {
    auto& path(registeredTriggers_[_token]);
    return path(0, path.Length() - 2).Data();
  }
  else
    return "";
}

UInt_t
panda::Run::getTriggerIndex(UInt_t _token) const
{
  try {
    return triggerIndices_.at(_token);
  }
  catch (std::exception& ex) {
    std::cerr << "Trigger menu is not up to date." << std::endl;
    std::cerr << "This happens when the runs or hlt tree is not in the input file" << std::endl;
    std::cerr << "or runNumber branch of the events tree is not read. Please check" << std::endl;
    std::cerr << "your input." << std::endl;
    throw;
  }
}

char const*
panda::Run::triggerMenu() const
{
  if (!hlt.menu)
    throw std::runtime_error("Trigger menu is not read from the input.");

  return *hlt.menu;
}

std::vector<TString> const&
panda::Run::triggerPaths() const
{
  if (!hlt.paths)
    throw std::runtime_error("Trigger menu is not read from the input.");

  return *hlt.paths;
}

void
panda::Run::findEntry(TTree& _runTree, UInt_t _runNumber)
{
  // Known issue: if this function is called with a new tree but with the same run number as the previous call,
  // nothing happens and the tree is not updated. This is such a rare situation that (in my opinion) does not warrant
  // covering for. To avoid it, set runNumber to 0 before calling this function if there was a file transition.
  // (EventBase::doGetEntry_ does that).

  if (_runNumber == runNumber)
    return;

  long iEntry(0);
  while (_runTree.GetEntry(iEntry++) > 0) {
    if (runNumber == _runNumber)
      break;
  }
  if (iEntry == _runTree.GetEntries()+1) {
    std::cerr << "Run " << _runNumber << " not found in run tree" << std::endl;
    throw std::runtime_error("InputError");
  }

  doGetEntry_(_runTree);
}

void
panda::Run::resetCache()
{
  if (inputTree_)
    inputTree_ = 0;

  inputTreeNumber_ = -1;
  
  hltMenuCache_ = -1;
}
    
/*private*/
void
panda::Run::updateTriggerTable_(TTree& _tree)
{
  if (&_tree != inputTree_) {
    inputTree_ = &_tree;
    inputTreeNumber_ = -1;
  }

  if (_tree.GetTreeNumber() != inputTreeNumber_) {
    inputTreeNumber_ = _tree.GetTreeNumber();
    hltMenuCache_ = -1;
  }

  if (hltMenu == hltMenuCache_)
    return;
  
  hltMenuCache_ = hltMenu;

  auto* inputFile(inputTree_->GetCurrentFile());
  if (!inputFile) {
    std::cerr << "No input file associated to the run tree" << std::endl;
    throw std::runtime_error("InputError");
  }

  auto* key(inputFile->GetKey("hlt"));
  if (!key) {
    std::cerr << "File " << inputFile->GetName() << " does not have an hlt tree" << std::endl;
    throw std::runtime_error("InputError");
  }

  auto* hltTree(static_cast<TTree*>(key->ReadObj()));

  if (!hlt.menu)
    hlt.create();

  hltTree->SetBranchAddress("menu", &hlt.menu);
  hltTree->SetBranchAddress("paths", &hlt.paths);

  if (hltTree->GetEntry(hltMenu) <= 0) {
    std::cerr << "Failed to read HLT menu " << hltMenu << " from " << inputFile->GetName() << std::endl;
    throw std::runtime_error("InputError");
  }

  delete hltTree;

  triggerIndices_.clear();
  // for each registered path, find the matching trigger in the given menu
  for (auto& path : registeredTriggers_) {
    unsigned iT(0);
    for (; iT != hlt.paths->size(); ++iT) {
      if ((*hlt.paths)[iT].BeginsWith(path)) {
        triggerIndices_.push_back(iT);
        break;
      }
    }
    if (iT == hlt.paths->size())
      triggerIndices_.push_back(-1);
  }
}

/* END CUSTOM */
