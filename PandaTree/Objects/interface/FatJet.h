#ifndef PandaTree_Objects_FatJet_h
#define PandaTree_Objects_FatJet_h
#include "Constants.h"
#include "Jet.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"
#include "MicroJet.h"

namespace panda {

  class FatJet : public Jet {
  public:
    struct datastore : public Jet::datastore {
      datastore() : Jet::datastore() {}
      ~datastore() { deallocate(); }

      /* ParticleP
      Float_t* pt_{0};
      Float_t* eta_{0};
      Float_t* phi_{0};
      */
      /* ParticleM
      Float_t* mass_{0};
      */
      /* MicroJet
      Float_t* csv{0};
      Float_t* qgl{0};
      */
      /* Jet
      Float_t* rawPt{0};
      Float_t* ptCorrUp{0};
      Float_t* ptCorrDown{0};
      Float_t* ptSmear{0};
      Float_t* ptSmearUp{0};
      Float_t* ptSmearDown{0};
      Float_t* area{0};
      Float_t* nhf{0};
      Float_t* chf{0};
      Float_t* puid{0};
      Bool_t* loose{0};
      Bool_t* tight{0};
      Bool_t* monojet{0};
      ContainerBase const* matchedGenJetContainer_{0};
      Short_t* matchedGenJet_{0};
      ContainerBase const* constituentsContainer_{0};
      std::vector<std::vector<Short_t>>* constituents_{0};
      */
      Float_t* tau1{0};
      Float_t* tau2{0};
      Float_t* tau3{0};
      Float_t* mSD{0};
      Float_t* mPruned{0};
      Float_t* tau1SD{0};
      Float_t* tau2SD{0};
      Float_t* tau3SD{0};
      Float_t* htt_mass{0};
      Float_t* htt_frec{0};
      Float_t* double_sub{0};
      Float_t (*ecfs)[3][4][4]{0};
      ContainerBase const* subjetsContainer_{0};
      std::vector<std::vector<Short_t>>* subjets_{0};

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

    typedef Array<FatJet> array_type;
    typedef Collection<FatJet> collection_type;

    typedef Jet base_type;

    FatJet(char const* name = "");
    FatJet(FatJet const&);
    FatJet(datastore&, UInt_t idx);
    ~FatJet();
    FatJet& operator=(FatJet const&);

    static char const* typeName() { return "FatJet"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    double get_ecf(int o_, int N_, int ib_) const;
    bool set_ecf(int o_, int N_, int ib_, float x_);

    /* MicroJet
    Float_t& csv;
    Float_t& qgl;
    */
    /* Jet
    Float_t& rawPt;
    Float_t& ptCorrUp;
    Float_t& ptCorrDown;
    Float_t& ptSmear;
    Float_t& ptSmearUp;
    Float_t& ptSmearDown;
    Float_t& area;
    Float_t& nhf;
    Float_t& chf;
    Float_t& puid;
    Bool_t& loose;
    Bool_t& tight;
    Bool_t& monojet;
    Ref<GenJet> matchedGenJet;
    RefVector<PFCand> constituents;
    */
    Float_t& tau1;
    Float_t& tau2;
    Float_t& tau3;
    Float_t& mSD;
    Float_t& mPruned;
    Float_t& tau1SD;
    Float_t& tau2SD;
    Float_t& tau3SD;
    Float_t& htt_mass;
    Float_t& htt_frec;
    Float_t& double_sub;
    Float_t (&ecfs)[3][4][4];
    RefVector<MicroJet> subjets;

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
    /* BEGIN CUSTOM FatJet.h.classdef */
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    FatJet(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<FatJet> FatJetArray;
  typedef Collection<FatJet> FatJetCollection;
  typedef Ref<FatJet> FatJetRef;
  typedef RefVector<FatJet> FatJetRefVector;

  /* BEGIN CUSTOM FatJet.h.global */
  /* END CUSTOM */

}

#endif
