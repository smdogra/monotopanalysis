#ifndef PandaTree_Objects_Jet_h
#define PandaTree_Objects_Jet_h
#include "Constants.h"
#include "MicroJet.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"
#include "GenJet.h"
#include "PFCand.h"

namespace panda {

  class Jet : public MicroJet {
  public:
    struct datastore : public MicroJet::datastore {
      datastore() : MicroJet::datastore() {}
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

    typedef Array<Jet> array_type;
    typedef Collection<Jet> collection_type;

    typedef MicroJet base_type;

    Jet(char const* name = "");
    Jet(Jet const&);
    Jet(datastore&, UInt_t idx);
    ~Jet();
    Jet& operator=(Jet const&);

    static char const* typeName() { return "Jet"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    /* MicroJet
    Float_t& csv;
    Float_t& qgl;
    */
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
    /* BEGIN CUSTOM Jet.h.classdef */
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    Jet(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<Jet> JetArray;
  typedef Collection<Jet> JetCollection;
  typedef Ref<Jet> JetRef;
  typedef RefVector<Jet> JetRefVector;

  /* BEGIN CUSTOM Jet.h.global */
  /* END CUSTOM */

}

#endif
