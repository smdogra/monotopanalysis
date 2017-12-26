#ifndef PandaTree_Objects_EventTPEEG_h
#define PandaTree_Objects_EventTPEEG_h
#include "EventTPEG.h"
#include "Constants.h"
#include "Electron.h"

namespace panda {

  class EventTPEEG : public EventTPEG {
  public:
    EventTPEEG();
    EventTPEEG(EventTPEEG const&);
    ~EventTPEEG();
    EventTPEEG& operator=(EventTPEEG const&);

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    ElectronCollection looseTags = ElectronCollection("looseTags", 32);

    static utils::BranchList getListOfBranches(Bool_t direct = kFALSE);

  protected:
    void doSetStatus_(TTree&, utils::BranchList const&) override;
    utils::BranchList doGetStatus_(TTree&) const override;
    utils::BranchList doGetBranchNames_() const override;
    void doSetAddress_(TTree&, utils::BranchList const&, Bool_t setStatus) override;
    void doBook_(TTree&, utils::BranchList const&) override;
    void doGetEntry_(TTree&) override;
    void doInit_() override;
    void doUnlink_(TTree&) override;

  public:
    /* BEGIN CUSTOM EventTPEEG.h.classdef */
    /* END CUSTOM */
  };

  /* BEGIN CUSTOM EventTPEEG.h.global */
  /* END CUSTOM */

}

#endif
