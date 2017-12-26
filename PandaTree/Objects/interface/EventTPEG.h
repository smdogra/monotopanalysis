#ifndef PandaTree_Objects_EventTPEG_h
#define PandaTree_Objects_EventTPEG_h
#include "EventTP.h"
#include "Constants.h"
#include "Electron.h"
#include "XPhoton.h"

namespace panda {

  class EventTPEG : public EventTP {
  public:
    EventTPEG();
    EventTPEG(EventTPEG const&);
    ~EventTPEG();
    EventTPEG& operator=(EventTPEG const&);

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    ElectronCollection tags = ElectronCollection("tags", 32);
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
    /* BEGIN CUSTOM EventTPEG.h.classdef */
    /* END CUSTOM */
  };

  /* BEGIN CUSTOM EventTPEG.h.global */
  /* END CUSTOM */

}

#endif
