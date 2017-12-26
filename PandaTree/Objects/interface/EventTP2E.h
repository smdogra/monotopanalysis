#ifndef PandaTree_Objects_EventTP2E_h
#define PandaTree_Objects_EventTP2E_h
#include "EventTP.h"
#include "Constants.h"
#include "Electron.h"

namespace panda {

  class EventTP2E : public EventTP {
  public:
    EventTP2E();
    EventTP2E(EventTP2E const&);
    ~EventTP2E();
    EventTP2E& operator=(EventTP2E const&);

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    ElectronCollection tags = ElectronCollection("tags", 32);
    ElectronCollection probes = ElectronCollection("probes", 32);

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
    /* BEGIN CUSTOM EventTP2E.h.classdef */
    /* END CUSTOM */
  };

  /* BEGIN CUSTOM EventTP2E.h.global */
  /* END CUSTOM */

}

#endif
