#ifndef PandaTree_Objects_GenParticle_h
#define PandaTree_Objects_GenParticle_h
#include "Constants.h"
#include "PackedParticle.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"

namespace panda {

  class GenParticle : public PackedParticle {
  public:
    enum StatusFlag {
      kIsPrompt,
      kIsDecayedLeptonHadron,
      kIsTauDecayProduct,
      kIsPromptTauDecayProduct,
      kIsDirectTauDecayProduct,
      kIsDirectPromptTauDecayProduct,
      kIsDirectHadronDecayProduct,
      kIsHardProcess,
      kFromHardProcess,
      kIsHardProcessTauDecayProduct,
      kIsDirectHardProcessTauDecayProduct,
      kFromHardProcessBeforeFSR,
      kIsFirstCopy,
      kIsLastCopy,
      kIsLastCopyBeforeFSR,
      nStatusFlags
    };

    static TString StatusFlagName[nStatusFlags];

    struct datastore : public PackedParticle::datastore {
      datastore() : PackedParticle::datastore() {}
      ~datastore() { deallocate(); }

      /* PackedParticle
      UShort_t* packedPt{0};
      Short_t* packedEta{0};
      Short_t* packedPhi{0};
      UShort_t* packedM{0};
      */
      Int_t* pdgid{0};
      Bool_t* finalState{0};
      UShort_t* statusFlags{0};
      ContainerBase const* parentContainer_{0};
      Short_t* parent_{0};

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

    typedef Array<GenParticle> array_type;
    typedef Collection<GenParticle> collection_type;

    typedef PackedParticle base_type;

    GenParticle(char const* name = "");
    GenParticle(GenParticle const&);
    GenParticle(datastore&, UInt_t idx);
    ~GenParticle();
    GenParticle& operator=(GenParticle const&);

    static char const* typeName() { return "GenParticle"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    bool testFlag(StatusFlag f) const { return ((statusFlags >> f) & 1) == 1; }

    /* PackedParticle
    UShort_t& packedPt;
    Short_t& packedEta;
    Short_t& packedPhi;
    UShort_t& packedM;
    */
    Int_t& pdgid;
    Bool_t& finalState;
    UShort_t& statusFlags;
    Ref<GenParticle> parent;

    /* BEGIN CUSTOM GenParticle.h.classdef */
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    GenParticle(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<GenParticle> GenParticleArray;
  typedef Collection<GenParticle> GenParticleCollection;
  typedef Ref<GenParticle> GenParticleRef;
  typedef RefVector<GenParticle> GenParticleRefVector;

  /* BEGIN CUSTOM GenParticle.h.global */
  /* END CUSTOM */

}

#endif
