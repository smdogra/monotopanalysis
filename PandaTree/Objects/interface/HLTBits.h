#ifndef PandaTree_Objects_HLTBits_h
#define PandaTree_Objects_HLTBits_h
#include "Constants.h"
#include "../../Framework/interface/Singlet.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"

namespace panda {

  class HLTBits : public Singlet {
  public:
    typedef Singlet base_type;

    HLTBits(char const* name = "");
    HLTBits(HLTBits const&);
    ~HLTBits();
    HLTBits& operator=(HLTBits const&);

    static char const* typeName() { return "HLTBits"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    void set(unsigned iB) { if (iB >= 1024) return; words[iB / 32] |= (1 << (iB % 32)); }
    bool pass(unsigned iB) const { if (iB >= 1024) return false; return (words[iB / 32] & (1 << (iB % 32))) != 0; }

    UInt_t words[32]{};

    /* BEGIN CUSTOM HLTBits.h.classdef */
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

  protected:
    void doSetStatus_(TTree&, utils::BranchList const&) override;
    utils::BranchList doGetStatus_(TTree&) const override;
    utils::BranchList doGetBranchNames_(Bool_t) const override;
    void doSetAddress_(TTree&, utils::BranchList const& = {"*"}, Bool_t setStatus = kTRUE) override;
    void doBook_(TTree&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  /* BEGIN CUSTOM HLTBits.h.global */
  /* END CUSTOM */

}

#endif
