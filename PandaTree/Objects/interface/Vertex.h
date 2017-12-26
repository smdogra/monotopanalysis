#ifndef PandaTree_Objects_Vertex_h
#define PandaTree_Objects_Vertex_h
#include "Constants.h"
#include "../../Framework/interface/Element.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"

namespace panda {

  class Vertex : public Element {
  public:
    struct datastore : public Element::datastore {
      datastore() : Element::datastore() {}
      ~datastore() { deallocate(); }

      Float_t* x{0};
      Float_t* y{0};
      Float_t* z{0};

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

    typedef Array<Vertex> array_type;
    typedef Collection<Vertex> collection_type;

    typedef Element base_type;

    Vertex(char const* name = "");
    Vertex(Vertex const&);
    Vertex(datastore&, UInt_t idx);
    ~Vertex();
    Vertex& operator=(Vertex const&);

    static char const* typeName() { return "Vertex"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    TVector3 position() const { return TVector3(x, y, z); }

    Float_t& x;
    Float_t& y;
    Float_t& z;

    /* BEGIN CUSTOM Vertex.h.classdef */
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    Vertex(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<Vertex> VertexArray;
  typedef Collection<Vertex> VertexCollection;
  typedef Ref<Vertex> VertexRef;
  typedef RefVector<Vertex> VertexRefVector;

  /* BEGIN CUSTOM Vertex.h.global */
  /* END CUSTOM */

}

#endif
