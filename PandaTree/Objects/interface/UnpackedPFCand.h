#ifndef PandaTree_Objects_UnpackedPFCand_h
#define PandaTree_Objects_UnpackedPFCand_h
#include "Constants.h"
#include "ParticleM.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"
#include "PFCand.h"
#include "RecoVertex.h"

namespace panda {

  class UnpackedPFCand : public ParticleM {
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
      Char_t* puppiW{0};
      Char_t* puppiWNoLep{0};
      UChar_t* ptype{0};
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

    typedef Array<UnpackedPFCand> array_type;
    typedef Collection<UnpackedPFCand> collection_type;

    typedef ParticleM base_type;

    UnpackedPFCand(char const* name = "");
    UnpackedPFCand(UnpackedPFCand const&);
    UnpackedPFCand(datastore&, UInt_t idx);
    ~UnpackedPFCand();
    UnpackedPFCand& operator=(UnpackedPFCand const&);

    static char const* typeName() { return "UnpackedPFCand"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    TLorentzVector puppiP4() const { TLorentzVector p4; p4.SetPtEtaPhiM(pt() * puppiW, eta(), phi(), m() * puppiW); return p4; }
    TLorentzVector puppiNoLepP4() const { TLorentzVector p4; p4.SetPtEtaPhiM(pt() * puppiWNoLep, eta(), phi(), m() * puppiWNoLep); return p4; }
    int q() const { return PFCand::q_[ptype]; }
    int pdgId() const { return PFCand::pdgId_[ptype]; }

    Char_t& puppiW;
    Char_t& puppiWNoLep;
    UChar_t& ptype;
    Ref<RecoVertex> vertex;

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
    /* BEGIN CUSTOM UnpackedPFCand.h.classdef */
    UnpackedPFCand& operator=(PFCand const&);
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    UnpackedPFCand(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<UnpackedPFCand> UnpackedPFCandArray;
  typedef Collection<UnpackedPFCand> UnpackedPFCandCollection;
  typedef Ref<UnpackedPFCand> UnpackedPFCandRef;
  typedef RefVector<UnpackedPFCand> UnpackedPFCandRefVector;

  /* BEGIN CUSTOM UnpackedPFCand.h.global */
  /* END CUSTOM */

}

#endif
