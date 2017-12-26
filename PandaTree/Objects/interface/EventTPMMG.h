#ifndef PandaTree_Objects_EventTPMMG_h
#define PandaTree_Objects_EventTPMMG_h
#include "EventTPMG.h"
#include "Constants.h"
#include "Muon.h"

namespace panda {

  class EventTPMMG : public EventTPMG {
  public:
    EventTPMMG();
    EventTPMMG(EventTPMMG const&);
    ~EventTPMMG();
    EventTPMMG& operator=(EventTPMMG const&);

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    MuonCollection looseTags = MuonCollection("looseTags", 32);

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
    /* BEGIN CUSTOM EventTPMMG.h.classdef */
    /* END CUSTOM */
  };

  /* BEGIN CUSTOM EventTPMMG.h.global */
  /* END CUSTOM */

}

#endif
