#ifndef PandaTree_Objects_Particle_h
#define PandaTree_Objects_Particle_h
#include "Constants.h"
#include "../../Framework/interface/Element.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"

namespace panda {

  class Particle : public Element {
  public:
    struct datastore : public Element::datastore {
      datastore() : Element::datastore() {}
      ~datastore() { deallocate(); }

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

    typedef Array<Particle> array_type;
    typedef Collection<Particle> collection_type;

    typedef Element base_type;

    Particle(datastore&, UInt_t idx);
    ~Particle();
    Particle& operator=(Particle const&);

    static char const* typeName() { return "Particle"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    TLorentzVector p4() const { TLorentzVector p4; p4.SetPtEtaPhiM(pt(), eta(), phi(), m()); return p4; }
    double e() const { return std::sqrt(std::pow(pt() * std::cosh(eta()), 2.) + m() * m()); }
    double p() const { return pt() * std::cosh(eta()); }
    double px() const { return pt() * std::cos(phi()); }
    double py() const { return pt() * std::sin(phi()); }
    double pz() const { return pt() * std::sinh(eta()); }
    virtual double pt() const = 0;
    virtual double eta() const = 0;
    virtual double phi() const = 0;
    virtual double m() const = 0;
    double dEta(Particle const& p) const { return eta() - p.eta(); }
    double dPhi(Particle const& p) const { return TVector2::Phi_mpi_pi(phi() - p.phi()); }
    double dR2(Particle const& p) const { double d1(dEta(p)); double d2(dPhi(p)); return d1 * d1 + d2 * d2; }
    double dR(Particle const& p) const { return std::sqrt(dR2(p)); }
    virtual void setPtEtaPhiM(double pt, double eta, double phi, double m) = 0;
    virtual void setXYZE(double px, double py, double pz, double e) = 0;

    /* BEGIN CUSTOM Particle.h.classdef */
    //! sort comparison by pt
    /*!
     * This function takes Elements as arguments to conform with ContainerBase::Comparison
     * but assumes that both arguments are static_castable to Particle.
     */
    static Bool_t PtGreater(Element const& p1, Element const& p2) {
      return static_cast<Particle const&>(p1).pt() > static_cast<Particle const&>(p2).pt();
    }
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    Particle(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<Particle> ParticleArray;
  typedef Collection<Particle> ParticleCollection;
  typedef Ref<Particle> ParticleRef;
  typedef RefVector<Particle> ParticleRefVector;

  /* BEGIN CUSTOM Particle.h.global */
  /* END CUSTOM */

}

#endif
