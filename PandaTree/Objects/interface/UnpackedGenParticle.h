#ifndef PandaTree_Objects_UnpackedGenParticle_h
#define PandaTree_Objects_UnpackedGenParticle_h
#include "Constants.h"
#include "ParticleM.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"
#include "GenParticle.h"

namespace panda {

  class UnpackedGenParticle : public ParticleM {
  public:
    struct datastore : public ParticleM::datastore {
      datastore() : ParticleM::datastore() {}
      ~datastore() { deallocate(); }

      /* ParticleP
      Float_t* pt_{0};
      Float_t* eta_{0};
      Float_t* phi_{0};
      */
      /* ParticleM
      Float_t* mass_{0};
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

    typedef Array<UnpackedGenParticle> array_type;
    typedef Collection<UnpackedGenParticle> collection_type;

    typedef ParticleM base_type;

    UnpackedGenParticle(char const* name = "");
    UnpackedGenParticle(UnpackedGenParticle const&);
    UnpackedGenParticle(datastore&, UInt_t idx);
    ~UnpackedGenParticle();
    UnpackedGenParticle& operator=(UnpackedGenParticle const&);

    static char const* typeName() { return "UnpackedGenParticle"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    bool testFlag(GenParticle::StatusFlag f) const { return ((statusFlags >> f) & 1) == 1; }

    Int_t& pdgid;
    Bool_t& finalState;
    UShort_t& statusFlags;
    Ref<UnpackedGenParticle> parent;

  protected:
    /* ParticleP
    Float_t& pt_;
    Float_t& eta_;
    Float_t& phi_;
    */
    /* ParticleM
    Float_t& mass_;
    */

  public:
    /* BEGIN CUSTOM UnpackedGenParticle.h.classdef */
    UnpackedGenParticle& operator=(GenParticle const&);
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    UnpackedGenParticle(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<UnpackedGenParticle> UnpackedGenParticleArray;
  typedef Collection<UnpackedGenParticle> UnpackedGenParticleCollection;
  typedef Ref<UnpackedGenParticle> UnpackedGenParticleRef;
  typedef RefVector<UnpackedGenParticle> UnpackedGenParticleRefVector;

  /* BEGIN CUSTOM UnpackedGenParticle.h.global */
  /* END CUSTOM */

}

#endif
