#ifndef PandaTree_Objects_PackedParticle_h
#define PandaTree_Objects_PackedParticle_h
#include "Constants.h"
#include "Particle.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"

namespace panda {

  class PackedParticle : public Particle {
  public:
    struct datastore : public Particle::datastore {
      datastore() : Particle::datastore() {}
      ~datastore() { deallocate(); }

      UShort_t* packedPt{0};
      Short_t* packedEta{0};
      Short_t* packedPhi{0};
      UShort_t* packedM{0};

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

    typedef Array<PackedParticle> array_type;
    typedef Collection<PackedParticle> collection_type;

    typedef Particle base_type;

    PackedParticle(char const* name = "");
    PackedParticle(PackedParticle const&);
    PackedParticle(datastore&, UInt_t idx);
    ~PackedParticle();
    PackedParticle& operator=(PackedParticle const&);

    static char const* typeName() { return "PackedParticle"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    double pt() const override { unpack_(); return pt_; }
    double eta() const override { unpack_(); return eta_; }
    double phi() const override { unpack_(); return phi_; }
    double m() const override { unpack_(); return mass_; }
    void setPtEtaPhiM(double pt, double eta, double phi, double m) override;
    void setXYZE(double px, double py, double pz, double e) override;

    UShort_t& packedPt;
    Short_t& packedEta;
    Short_t& packedPhi;
    UShort_t& packedM;

    /* BEGIN CUSTOM PackedParticle.h.classdef */
  protected:
    void pack_();
    void unpack_() const;
    virtual void packMore_() {}
    virtual void unpackMore_() const {}

    mutable Double_t pt_{0.};
    mutable Double_t eta_{0.};
    mutable Double_t phi_{0.};
    mutable Double_t mass_{0.};
    mutable Bool_t unpacked_{kFALSE};

  public:
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    PackedParticle(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<PackedParticle> PackedParticleArray;
  typedef Collection<PackedParticle> PackedParticleCollection;
  typedef Ref<PackedParticle> PackedParticleRef;
  typedef RefVector<PackedParticle> PackedParticleRefVector;

  /* BEGIN CUSTOM PackedParticle.h.global */
  /* END CUSTOM */

}

#endif
