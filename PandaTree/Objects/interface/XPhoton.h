#ifndef PandaTree_Objects_XPhoton_h
#define PandaTree_Objects_XPhoton_h
#include "Constants.h"
#include "Photon.h"
#include "../../Framework/interface/Array.h"
#include "../../Framework/interface/Collection.h"
#include "../../Framework/interface/Ref.h"
#include "../../Framework/interface/RefVector.h"

namespace panda {

  class XPhoton : public Photon {
  public:
    enum IDTune {
      kSpring15,
      kSpring16,
      kGJetsCWIso,
      kZGCWIso,
      nIDTunes
    };

    static TString IDTuneName[nIDTunes];

    static double const chIsoCuts[nIDTunes][2][4];
    static double const nhIsoCuts[nIDTunes][2][4];
    static double const phIsoCuts[nIDTunes][2][4];
    static double const sieieCuts[nIDTunes][2][4];
    static double const hOverECuts[nIDTunes][2][4];

    struct datastore : public Photon::datastore {
      datastore() : Photon::datastore() {}
      ~datastore() { deallocate(); }

      /* ParticleP
      Float_t* pt_{0};
      Float_t* eta_{0};
      Float_t* phi_{0};
      */
      /* Photon
      Float_t* pfPt{0};
      Float_t* chIso{0};
      Float_t* chIsoMax{0};
      Float_t* nhIso{0};
      Float_t* phIso{0};
      Float_t* sieie{0};
      Float_t* sipip{0};
      Float_t* hOverE{0};
      Float_t* genIso{0};
      Float_t* mipEnergy{0};
      Float_t* emax{0};
      Float_t* e2nd{0};
      Float_t* eleft{0};
      Float_t* eright{0};
      Float_t* etop{0};
      Float_t* ebottom{0};
      Float_t* r9{0};
      Float_t* etaWidth{0};
      Float_t* phiWidth{0};
      Float_t* time{0};
      Float_t* timeSpan{0};
      Float_t* regPt{0};
      Float_t* smearedPt{0};
      Float_t* originalPt{0};
      Bool_t* loose{0};
      Bool_t* medium{0};
      Bool_t* tight{0};
      Bool_t* highpt{0};
      Bool_t* pixelVeto{0};
      Bool_t* csafeVeto{0};
      Bool_t (*triggerMatch)[nTriggerObjects]{0};
      ContainerBase const* superClusterContainer_{0};
      Short_t* superCluster_{0};
      ContainerBase const* matchedPFContainer_{0};
      Short_t* matchedPF_{0};
      ContainerBase const* matchedGenContainer_{0};
      Short_t* matchedGen_{0};
      */
      Float_t* scEta{0};
      Float_t* scRawPt{0};
      Float_t (*chIsoX)[nIDTunes]{0};
      Float_t (*chIsoMaxX)[nIDTunes]{0};
      Float_t (*nhIsoX)[nIDTunes]{0};
      Float_t (*phIsoX)[nIDTunes]{0};
      Float_t* e4{0};
      Bool_t* isEB{0};
      Bool_t (*looseX)[nIDTunes]{0};
      Bool_t (*mediumX)[nIDTunes]{0};
      Bool_t (*tightX)[nIDTunes]{0};
      Int_t* matchedGenId{0};

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

    typedef Array<XPhoton> array_type;
    typedef Collection<XPhoton> collection_type;

    typedef Photon base_type;

    XPhoton(char const* name = "");
    XPhoton(XPhoton const&);
    XPhoton(datastore&, UInt_t idx);
    ~XPhoton();
    XPhoton& operator=(XPhoton const&);

    static char const* typeName() { return "XPhoton"; }

    void print(std::ostream& = std::cout, UInt_t level = 1) const override;
    void dump(std::ostream& = std::cout) const override;

    bool passCHIso(UInt_t wp, UInt_t t) const { return chIsoX[t] < chIsoCuts[t][isEB ? 0 : 1][wp]; }
    bool passCHIsoMax(UInt_t wp, UInt_t t) const { return chIsoMaxX[t] < chIsoCuts[t][isEB ? 0 : 1][wp]; }
    bool passNHIso(UInt_t wp, UInt_t t) const { return nhIsoX[t] < nhIsoCuts[t][isEB ? 0 : 1][wp]; }
    bool passPhIso(UInt_t wp, UInt_t t) const { return phIsoX[t] < phIsoCuts[t][isEB ? 0 : 1][wp]; }
    bool passSieie(UInt_t wp, UInt_t t) const { return sieie < sieieCuts[t][isEB ? 0 : 1][wp]; }
    bool passHOverE(UInt_t wp, UInt_t t) const { return hOverE < hOverECuts[t][isEB ? 0 : 1][wp]; }

    /* Photon
    Float_t& pfPt; // Pt of the dR-closest PF candidate
    Float_t& chIso;
    Float_t& chIsoMax;
    Float_t& nhIso;
    Float_t& phIso;
    Float_t& sieie;
    Float_t& sipip;
    Float_t& hOverE;
    Float_t& genIso;
    Float_t& mipEnergy;
    Float_t& emax;
    Float_t& e2nd;
    Float_t& eleft;
    Float_t& eright;
    Float_t& etop;
    Float_t& ebottom;
    Float_t& r9;
    Float_t& etaWidth;
    Float_t& phiWidth;
    Float_t& time;
    Float_t& timeSpan;
    Float_t& regPt;
    Float_t& smearedPt;
    Float_t& originalPt;
    Bool_t& loose;
    Bool_t& medium;
    Bool_t& tight;
    Bool_t& highpt;
    Bool_t& pixelVeto;
    Bool_t& csafeVeto;
    Bool_t (&triggerMatch)[nTriggerObjects];
    Ref<SuperCluster> superCluster;
    Ref<PFCand> matchedPF;
    Ref<GenParticle> matchedGen;
    */
    Float_t& scEta;
    Float_t& scRawPt;
    Float_t (&chIsoX)[nIDTunes];
    Float_t (&chIsoMaxX)[nIDTunes];
    Float_t (&nhIsoX)[nIDTunes];
    Float_t (&phIsoX)[nIDTunes];
    Float_t& e4;
    Bool_t& isEB;
    Bool_t (&looseX)[nIDTunes];
    Bool_t (&mediumX)[nIDTunes];
    Bool_t (&tightX)[nIDTunes];
    Int_t& matchedGenId;

  protected:
    /* ParticleP
    Float_t& pt_;
    Float_t& eta_;
    Float_t& phi_;
    */

  public:
    /* BEGIN CUSTOM XPhoton.h.classdef */
    /* END CUSTOM */

    static utils::BranchList getListOfBranches();

    void destructor(Bool_t recursive = kFALSE);

  protected:
    XPhoton(ArrayBase*);

    void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;
    void doInit_() override;
  };

  typedef Array<XPhoton> XPhotonArray;
  typedef Collection<XPhoton> XPhotonCollection;
  typedef Ref<XPhoton> XPhotonRef;
  typedef RefVector<XPhoton> XPhotonRefVector;

  /* BEGIN CUSTOM XPhoton.h.global */
  /* END CUSTOM */

}

#endif
