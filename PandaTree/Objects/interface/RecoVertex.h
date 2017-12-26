#ifndef PandaTree_Objects_RecoVertex_h
#define PandaTree_Objects_RecoVertex_h
#include "Constants.h"
#include "Vertex.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"

namespace panda {

  class RecoVertex : public Vertex {
  public:
    struct datastore : public Vertex::datastore {
      datastore() : Vertex::datastore() {}
      ~datastore() { deallocate(); }

      /* Vertex
      Float_t* x{0};
      Float_t* y{0};
      Float_t* z{0};
      */
      Float_t* score{0};
      UShort_t* ntrk{0};
      Float_t* ndof{0};
      Float_t* chi2{0};
      UShort_t* pfRangeMax{0};

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

    typedef Array<RecoVertex> array_type;
    typedef Collection<RecoVertex> collection_type;

    typedef Vertex base_type;

    RecoVertex(char const* name = "");
    RecoVertex(RecoVertex const&);
    RecoVertex(datastore&, UInt_t idx);
    ~RecoVertex();
    RecoVertex& operator=(RecoVertex const&);

    static char const* typeName() { return "RecoVertex"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    double normChi2() const { return ndof != 0. ? chi2 / ndof : chi2 * 1.e+6; }

    /* Vertex
    Float_t& x;
    Float_t& y;
    Float_t& z;
    */
    Float_t& score;
    UShort_t& ntrk;
    Float_t& ndof;
    Float_t& chi2;
    UShort_t& pfRangeMax;

    /* BEGIN CUSTOM RecoVertex.h.classdef */
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    RecoVertex(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<RecoVertex> RecoVertexArray;
  typedef Collection<RecoVertex> RecoVertexCollection;
  typedef Ref<RecoVertex> RecoVertexRef;
  typedef RefVector<RecoVertex> RecoVertexRefVector;

  /* BEGIN CUSTOM RecoVertex.h.global */
  /* END CUSTOM */

}

#endif
