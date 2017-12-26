#ifndef PandaTree_Objects_EventTP_h
#define PandaTree_Objects_EventTP_h
#include "EventBase.h"
#include "Constants.h"
#include "TPPair.h"
#include "Jet.h"
#include "RecoMet.h"

namespace panda {

  class EventTP : public EventBase {
  public:
    EventTP();
    EventTP(EventTP const&);
    ~EventTP();
    EventTP& operator=(EventTP const&);

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    TPPairCollection tp = TPPairCollection("tp", 32);
    JetCollection jets = JetCollection("jets", 64);
    RecoMet t1Met = RecoMet("t1Met");

    UShort_t npv{};
    UShort_t npvTrue{};
    Float_t rho{};
    UInt_t sample{};

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
    /* BEGIN CUSTOM EventTP.h.classdef */
    /* END CUSTOM */
  };

  /* BEGIN CUSTOM EventTP.h.global */
  /* END CUSTOM */

}

#endif
