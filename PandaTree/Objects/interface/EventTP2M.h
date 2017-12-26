#ifndef PandaTree_Objects_EventTP2M_h
#define PandaTree_Objects_EventTP2M_h
#include "EventTP.h"
#include "Constants.h"
#include "Muon.h"

namespace panda {

  class EventTP2M : public EventTP {
  public:
    EventTP2M();
    EventTP2M(EventTP2M const&);
    ~EventTP2M();
    EventTP2M& operator=(EventTP2M const&);

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    MuonCollection tags = MuonCollection("tags", 32);
    MuonCollection probes = MuonCollection("probes", 32);

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
    /* BEGIN CUSTOM EventTP2M.h.classdef */
    /* END CUSTOM */
  };

  /* BEGIN CUSTOM EventTP2M.h.global */
  /* END CUSTOM */

}

#endif
