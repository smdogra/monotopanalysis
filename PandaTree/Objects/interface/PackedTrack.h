#ifndef PandaTree_Objects_PackedTrack_h
#define PandaTree_Objects_PackedTrack_h
#include "Constants.h"
#include "../../Framework/interface/Element.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"

namespace panda {

  class PackedTrack : public Element {
  public:
    struct datastore : public Element::datastore {
      datastore() : Element::datastore() {}
      ~datastore() { deallocate(); }

      UShort_t* packedPtError{0};
      Short_t* packedDxy{0};
      Short_t* packedDz{0};
      Short_t* packedDPhi{0};

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

    typedef Array<PackedTrack> array_type;
    typedef Collection<PackedTrack> collection_type;

    typedef Element base_type;

    PackedTrack(char const* name = "");
    PackedTrack(PackedTrack const&);
    PackedTrack(datastore&, UInt_t idx);
    ~PackedTrack();
    PackedTrack& operator=(PackedTrack const&);

    static char const* typeName() { return "PackedTrack"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    void setPtError(double e) { ptError_ = e; packPtError_(); }
    double ptError() const { unpack_(); return ptError_; }
    double dxy() const { unpack_(); return dxy_; }
    double dz() const { unpack_(); return dz_; }
    double dPhi() const { unpack_(); return dPhi_; }

    UShort_t& packedPtError;
    Short_t& packedDxy;
    Short_t& packedDz;
    Short_t& packedDPhi;

    /* BEGIN CUSTOM PackedTrack.h.classdef */
  protected:
    void pack_();
    void packPtError_();
    void unpack_() const;

    mutable Double_t ptError_{0.};
    mutable Double_t dxy_{0.};
    mutable Double_t dz_{0.};
    mutable Double_t dPhi_{0.};
    mutable Bool_t unpacked_{kFALSE};

  public:
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    PackedTrack(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<PackedTrack> PackedTrackArray;
  typedef Collection<PackedTrack> PackedTrackCollection;
  typedef Ref<PackedTrack> PackedTrackRef;
  typedef RefVector<PackedTrack> PackedTrackRefVector;

  /* BEGIN CUSTOM PackedTrack.h.global */
  /* END CUSTOM */

}

#endif
