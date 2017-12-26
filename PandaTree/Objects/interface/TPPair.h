#ifndef PandaTree_Objects_TPPair_h
#define PandaTree_Objects_TPPair_h
#include "Constants.h"
#include "../../Framework/interface/Element.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"

namespace panda {

  class TPPair : public Element {
  public:
    struct datastore : public Element::datastore {
      datastore() : Element::datastore() {}
      ~datastore() { deallocate(); }

      Float_t* mass{0};
      Float_t* mass2{0};

      void allocate(UInt_t n) override;
      void deallocate() override;
      void setStatus(TTree&, TString const&, utils::BranchList const&) override;
      utils::BranchList getStatus(TTree&, TString const&) const override;
      utils::BranchList getBranchNames(TString const& = "") const override;
      void setAddress(TTree&, TString const&, utils::BranchList const& = {"*"}, Bool_t setStatus = kTRUE) override;
      void book(TTree&, TString const&, utils::BranchList const& = {"*"}, Bool_t dynamic = kTRUE) override;
      void releaseTree(TTree&, TString const&) override;
      void resizeVectors_(UInt_t) override;
    };

    typedef Array<TPPair> array_type;
    typedef Collection<TPPair> collection_type;

    typedef Element base_type;

    TPPair(char const* name = "");
    TPPair(TPPair const&);
    TPPair(datastore&, UInt_t idx);
    ~TPPair();
    TPPair& operator=(TPPair const&);

    static char const* typeName() { return "TPPair"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    Float_t& mass;
    Float_t& mass2;

    /* BEGIN CUSTOM TPPair.h.classdef */
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    TPPair(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<TPPair> TPPairArray;
  typedef Collection<TPPair> TPPairCollection;
  typedef Ref<TPPair> TPPairRef;
  typedef RefVector<TPPair> TPPairRefVector;

  /* BEGIN CUSTOM TPPair.h.global */
  /* END CUSTOM */

}

#endif
