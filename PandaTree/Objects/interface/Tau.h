#ifndef PandaTree_Objects_Tau_h
#define PandaTree_Objects_Tau_h
#include "Constants.h"
#include "ParticleM.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"
#include "Vertex.h"
#include "GenParticle.h"

namespace panda {

  class Tau : public ParticleM {
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
      Char_t* charge{0};
      Bool_t* decayMode{0};
      Bool_t* decayModeNew{0};
      Bool_t* looseIsoMVA{0};
      Bool_t* looseIsoMVAOld{0};
      Float_t* iso{0};
      Float_t* isoDeltaBetaCorr{0};
      ContainerBase const* vertexContainer_{0};
      Short_t* vertex_{0};
      ContainerBase const* matchedGenContainer_{0};
      Short_t* matchedGen_{0};

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

    typedef Array<Tau> array_type;
    typedef Collection<Tau> collection_type;

    typedef ParticleM base_type;

    Tau(char const* name = "");
    Tau(Tau const&);
    Tau(datastore&, UInt_t idx);
    ~Tau();
    Tau& operator=(Tau const&);

    static char const* typeName() { return "Tau"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    Char_t& charge;
    Bool_t& decayMode;
    Bool_t& decayModeNew;
    Bool_t& looseIsoMVA;
    Bool_t& looseIsoMVAOld;
    Float_t& iso;
    Float_t& isoDeltaBetaCorr;
    Ref<Vertex> vertex;
    Ref<GenParticle> matchedGen;

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
    /* BEGIN CUSTOM Tau.h.classdef */
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    Tau(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<Tau> TauArray;
  typedef Collection<Tau> TauCollection;
  typedef Ref<Tau> TauRef;
  typedef RefVector<Tau> TauRefVector;

  /* BEGIN CUSTOM Tau.h.global */
  /* END CUSTOM */

}

#endif
