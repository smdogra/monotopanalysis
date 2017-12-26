#ifndef PandaProd_Utilities_PNode_h
#define PandaProd_Utilities_PNode_h

#include "PandaTree/Objects/interface/GenParticle.h"

#include <vector>
#include <bitset>
#include <string>
#include <functional>

struct PNode {
  enum MomentumFormat {
    kPtEtaPhi,
    kPxPyPzE
  };

  enum MomentumDispMode {
    kShowAllP,
    kShowFinalP,
    kNoP
  };

  enum MassDispMode {
    kShowAllM,
    kShowHardScatM,
    kNoM
  };

  PNode* mother{0};
  std::vector<PNode*> daughters{};
  int pdgId{0};
  int status{0};
  std::bitset<15> statusBits{};
  double mass{0.};
  double pt{0.};
  double eta{0.};
  double phi{0.};
  bool ownDaughters{false};

  static MomentumFormat gMomentumFormat; // = kPtEtaPhi
  static MomentumDispMode gMomentumDispMode; // = kShowAllP
  static MassDispMode gMassDispMode; // = kShowAllM
  static double gMatchEta;
  static double gMatchPhi;
  static double gMatchDR;

  // true -> node is to be removed
  static std::function<bool(PNode const&)> gPruningFunction;

  PNode() {}
  ~PNode();
  PNode& operator=(PNode const&);

  std::string print() const;

  bool hasDescendant(int pdgId, bool ignoreCharge = true) const;
  bool hasAncestor(int pdgId, bool ignoreCharge = true) const;

  bool isIntermediateTerminal() const;
  bool isNoDecay() const;
  bool isHadronic() const;
  bool isHeavyHadron() const;
  bool isHadronicIntermediate() const;
  bool isFirstHeavyHadron() const;
  bool isLightDecayingToLight() const;

  //! Recursively prune the decay tree
  /*!
   * Pruning can be configured by defining gPruningFunction.
   * By default, each of the following daughter d is removed from the daughters list:
   *  . Terminal non-final state (d.daughters.size() == 0 && d.status != 1)
   *  . Repeating link (d.daughters.size() == 1 && d.daughters[0].pdgId == d.pdgId)
   *  . Hadronic intermediates (status != 1 and (the hundreds place of |d.pdgId| is nonzero, or pdgId == 21, 81-100)) that is not the first heavy-flavor hadron in the chain
   *  . |d.pdgId| <= 3 and |d.daughters[n].pdgId| <= 3 for all n
   * Daughters of removed d is connected back to the current node.
   */
  void pruneDaughters();

private:
  //! Write information string into info_
  void generateInfo_() const;
  std::string doPrint_(std::vector<std::string>* spacers, bool firstDaughter, bool lastDaughter) const;
  mutable std::string info_{""};
};

#endif
