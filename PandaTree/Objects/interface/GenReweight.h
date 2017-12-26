#ifndef PandaTree_Objects_GenReweight_h
#define PandaTree_Objects_GenReweight_h
#include "Constants.h"
#include "../../Framework/interface/Singlet.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"

namespace panda {

  class GenReweight : public Singlet {
  public:
    static unsigned const NMAX = 1024;

    typedef Singlet base_type;

    GenReweight(char const* name = "");
    GenReweight(GenReweight const&);
    ~GenReweight();
    GenReweight& operator=(GenReweight const&);

    static char const* typeName() { return "GenReweight"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    Float_t r1f2DW{};
    Float_t r1f5DW{};
    Float_t r2f1DW{};
    Float_t r2f2DW{};
    Float_t r5f1DW{};
    Float_t r5f5DW{};
    Float_t pdfDW{};
    Float_t genParam[NMAX]{};

    /* BEGIN CUSTOM GenReweight.h.classdef */
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

  /* BEGIN CUSTOM GenReweight.h.global */
  /* END CUSTOM */

}

#endif
