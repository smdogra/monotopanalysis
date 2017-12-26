#ifndef PandaTree_Objects_Lepton_h
#define PandaTree_Objects_Lepton_h
#include "Constants.h"
#include "ParticleP.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"
#include "PFCand.h"
#include "GenParticle.h"
#include "Vertex.h"

namespace panda {

  class Lepton : public ParticleP {
  public:
    struct datastore : public ParticleP::datastore {
      datastore() : ParticleP::datastore() {}
      ~datastore() { deallocate(); }

      /* ParticleP
      Float_t* pt_{0};
      Float_t* eta_{0};
      Float_t* phi_{0};
      */
      Float_t* pfPt{0};
      Char_t* charge{0};
      Bool_t* loose{0};
      Bool_t* medium{0};
      Bool_t* tight{0};
      Float_t* chIso{0};
      Float_t* nhIso{0};
      Float_t* phIso{0};
      Float_t* puIso{0};
      ContainerBase const* matchedPFContainer_{0};
      Short_t* matchedPF_{0};
      ContainerBase const* matchedGenContainer_{0};
      Short_t* matchedGen_{0};
      ContainerBase const* vertexContainer_{0};
      Short_t* vertex_{0};

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

    typedef Array<Lepton> array_type;
    typedef Collection<Lepton> collection_type;

    typedef ParticleP base_type;

    Lepton(char const* name = "");
    Lepton(Lepton const&);
    Lepton(datastore&, UInt_t idx);
    ~Lepton();
    Lepton& operator=(Lepton const&);

    static char const* typeName() { return "Lepton"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    virtual double combIso() const { return 0.; }

    Float_t& pfPt; // for E: Pt of the dR-closest PF candidate; for Mu: pfP4().pt()
    Char_t& charge;
    Bool_t& loose;
    Bool_t& medium;
    Bool_t& tight;
    Float_t& chIso;
    Float_t& nhIso;
    Float_t& phIso;
    Float_t& puIso;
    Ref<PFCand> matchedPF;
    Ref<GenParticle> matchedGen;
    Ref<Vertex> vertex;

  protected:
    /* ParticleP
    Float_t& pt_;
    Float_t& eta_;
    Float_t& phi_;
    */

  public:
    /* BEGIN CUSTOM Lepton.h.classdef */
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    Lepton(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<Lepton> LeptonArray;
  typedef Collection<Lepton> LeptonCollection;
  typedef Ref<Lepton> LeptonRef;
  typedef RefVector<Lepton> LeptonRefVector;

  /* BEGIN CUSTOM Lepton.h.global */
  /* END CUSTOM */

}

#endif
