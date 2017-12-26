#ifndef PandaTree_Objects_EventTPMG_h
#define PandaTree_Objects_EventTPMG_h
#include "EventTP.h"
#include "Constants.h"
#include "Muon.h"
#include "XPhoton.h"

namespace panda {

  class EventTPMG : public EventTP {
  public:
    EventTPMG();
    EventTPMG(EventTPMG const&);
    ~EventTPMG();
    EventTPMG& operator=(EventTPMG const&);

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    MuonCollection tags = MuonCollection("tags", 32);
    XPhotonCollection probes = XPhotonCollection("probes", 32);

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
    /* BEGIN CUSTOM EventTPMG.h.classdef */
    /* END CUSTOM */
  };

  /* BEGIN CUSTOM EventTPMG.h.global */
  /* END CUSTOM */

}

#endif
