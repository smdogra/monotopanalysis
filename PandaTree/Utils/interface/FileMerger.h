#ifndef PandaTree_Utils_PandaMerger_h
#define PandaTree_Utils_PandaMerger_h

#include "../../Framework/interface/IOUtils.h"

#include "TString.h"

#include <vector>
#include <functional>

namespace panda {

  class EventBase;

  //! Skimmer base class
  /*!
   * Pass a child class of Skimmer to FileMerger to run skimming during file merge process.
   * When a skimmer is present FileMerger, will read and write using the event object returned by getEvent().
   */
  class Skimmer {
  public:
    Skimmer() {}
    virtual ~Skimmer() {}

    virtual Bool_t skim() = 0;
    virtual EventBase* getEvent() = 0;
  };

  //! Merge multiple panda files with optional skimming, slimming, and editing.
  class FileMerger {
  public:
    enum TreeType {
      kEvent,
      kRun,
      nTreeTypes
    };

    FileMerger() {}
    ~FileMerger() {}

    //! Add input paths.
    /*!
     * \param path  Path to input panda files. Wildcard is allowed.
     */
    void addInput(char const* path);

    //! Select branches to write.
    /*!
     * By default, branches in the input trees are copied.
     *
     * \param blist    Branch list (see BranchList syntax in README).
     * \param onRead   Apply the branch list on input (default false)
     * \param treeType Tree to apply the selection
     */
    void selectBranches(utils::BranchList const& blist, Bool_t onRead = kFALSE, TreeType treeType = kEvent);

    //! Set event selection (as interpreted by TTree::Draw)
    void setEventSelection(char const* sel) { eventSelection_ = sel; }

    //! Run the merger.
    /*!
     * \param outPath   Output file name.
     * \param nEvents   The number of events to process (default -1 = process all).
     *
     * \return Number of events in the output file.
     */
    unsigned merge(char const* outPath, long nEvents = -1);

    //! Introduce a skim.
    /*!
     * Pass a subclass of Skimmer. This will change the I/O behavior in addition to enabling skim.
     *
     * \param skimmer   Skimmer object.
     */
    void setSkimmer(Skimmer& skimmer) { skimmer_ = &skimmer; }

    //! Set the number of seconds to wait before an input file appears (default 0).
    void setInputTimeout(UInt_t t) { timeout_ = t; }

    //! Set print level.
    void setPrintLevel(UInt_t l) { printLevel_ = l; }

  private:
    std::vector<TString> paths_{};
    utils::BranchList branchList_[nTreeTypes]{};
    Bool_t applyBranchListOnRead_[nTreeTypes]{};

    TString eventSelection_{};
    Skimmer* skimmer_{0};

    UInt_t timeout_{0};

    UInt_t printLevel_{1};
  };

}

#endif
