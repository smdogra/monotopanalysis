#include "../interface/PNode.h"

#include "TVector2.h"

#include <sstream>
#include <iomanip>
#include <cmath>

PNode::MomentumFormat PNode::gMomentumFormat(PNode::kPtEtaPhi);
PNode::MomentumDispMode PNode::gMomentumDispMode(PNode::kShowAllP);
PNode::MassDispMode PNode::gMassDispMode(PNode::kShowAllM);
double PNode::gMatchEta(0.);
double PNode::gMatchPhi(0.);
double PNode::gMatchDR(0.);

std::function<bool(PNode const&)> PNode::gPruningFunction([](PNode const& node)->bool {
    return node.isIntermediateTerminal() ||
      node.isNoDecay() ||
      (node.isHadronicIntermediate() && !node.isFirstHeavyHadron()) ||
      node.isLightDecayingToLight();
  });

PNode::~PNode()
{
  if (ownDaughters) {
    for (PNode* daughter : daughters)
      delete daughter;
  }
}

PNode&
PNode::operator=(PNode const& _rhs)
{
  mother = _rhs.mother;
  daughters = _rhs.daughters;
  pdgId = _rhs.pdgId;
  status = _rhs.status;
  statusBits = _rhs.statusBits;
  mass = _rhs.mass;
  pt = _rhs.pt;
  eta = _rhs.eta;
  phi = _rhs.phi;

  ownDaughters = _rhs.ownDaughters;
  info_ = "";
  return *this;
}

std::string
PNode::print() const
{
  generateInfo_();
  return doPrint_(0, true, true);
}

bool
PNode::hasDescendant(int _pdgId, bool _ignoreCharge/* = true*/) const
{
  for (PNode* daughter : daughters) {
    if (_ignoreCharge) {
      if (std::abs(daughter->pdgId) == _pdgId)
        return true;
    }
    else {
      if (daughter->pdgId == _pdgId)
        return true;
    }
    if (daughter->hasDescendant(_pdgId, _ignoreCharge))
      return true;
  }
  return false;
}

bool
PNode::hasAncestor(int _pdgId, bool _ignoreCharge/* = false*/) const
{
  if (!mother)
    return false;

  if (_ignoreCharge) {
    if (std::abs(mother->pdgId) == _pdgId)
      return true;
  }
  else {
    if (mother->pdgId == _pdgId)
      return true;
  }
  return mother->hasAncestor(_pdgId, _ignoreCharge);
}

bool
PNode::isIntermediateTerminal() const
{
  return daughters.size() == 0 && status != 1;
}

bool
PNode::isNoDecay() const
{
  return daughters.size() == 1 && pdgId == daughters[0]->pdgId;
}

bool
PNode::isHadronic() const
{
  unsigned pdg(std::abs(pdgId));
  return (pdg / 100) % 10 != 0 || pdg == 21 || (pdg > 80 && pdg < 101);
}

bool
PNode::isHeavyHadron() const
{
  unsigned pdg(std::abs(pdgId));
  return (pdg / 1000) % 10 >= 4 || (pdg / 100) % 10 >= 4;
}

bool
PNode::isHadronicIntermediate() const
{
  return status != 1 && isHadronic();
}

bool
PNode::isFirstHeavyHadron() const
{
  if (mother && mother->isHeavyHadron())
    return false;

  return isHeavyHadron();
}

bool
PNode::isLightDecayingToLight() const
{
  if (pdgId >= 4 || pdgId <= -4)
    return false;

  if (daughters.size() == 0)
    return false;

  for (auto* d : daughters)
    if (d->pdgId >= 4 || d->pdgId <= -4)
      return false;

  return true;
}

void
PNode::pruneDaughters()
{
  for (unsigned iD(0); iD < daughters.size(); ++iD) {
    PNode* dnode(daughters[iD]);
    dnode->pruneDaughters();

    if (gPruningFunction(*dnode)) {
      for (auto* gd : dnode->daughters)
        gd->mother = this;

      daughters.erase(daughters.begin() + iD);
      daughters.insert(daughters.begin() + iD, dnode->daughters.begin(), dnode->daughters.end());
      dnode->daughters.clear();
      if (ownDaughters)
        delete dnode;

      --iD;
    }
  }
}

void
PNode::generateInfo_() const
{
  if (!info_.empty())
    return;

  std::stringstream ss;
  ss << std::setw(9) << pdgId;

  if (gMassDispMode == kShowAllM || (gMassDispMode == kShowHardScatM && (statusBits[panda::GenParticle::kIsHardProcess] || statusBits[panda::GenParticle::kFromHardProcess])))
    ss << " [" << std::setw(6) << std::fixed << std::setprecision(2) << mass << "]";

  if (gMomentumDispMode == kShowAllP || (gMomentumDispMode == kShowFinalP && status == 1)) {
    if (gMomentumFormat == kPtEtaPhi) {
      ss << " (";
      ss << std::setw(5) << std::fixed << std::setprecision(pt < 1000. ? 1 : 0) << pt;
      ss << ",";
      if (eta > 10.)
        ss << "  inf";
      else if (eta < -10.)
        ss << " -inf";
      else
        ss << std::setw(5) << std::fixed << std::setprecision(1) << eta;
      ss << ",";
      ss << std::setw(5) << std::fixed << std::setprecision(1) << phi;
      ss << ")";
    } 
    else {
      double p4[4] = {pt * std::cos(phi), pt * std::sin(phi), pt * std::sinh(eta), std::sqrt(mass * mass + std::pow(pt * std::cosh(eta), 2.))};
      ss << " (";
      for (unsigned iP(0); iP != 4; ++iP) {
        ss << std::setw(5) << std::fixed << std::setprecision(p4[iP] < 1000. ? 1 : 0) << p4[iP];
        if (iP != 3)
          ss << ",";
      }
      ss << ")";
    }
  }

  if (status == 1) {
    bool first(true);
    for (unsigned iB(0); iB != panda::GenParticle::nStatusFlags; ++iB) {
      if (!statusBits[iB])
        continue;

      if (first) {
        ss << "{";
        first = false;
      }
      else
        ss << ",";
      ss << iB;
    }
    if (!first)
      ss << "}";

    if (gMatchDR > 0.) {
      double dEta(eta - gMatchEta);
      double dPhi(TVector2::Phi_mpi_pi(phi - gMatchPhi));
      if (dEta * dEta + dPhi * dPhi < gMatchDR * gMatchDR)
        ss << " *";
    }
  }
  else {
    ss << " " << status << " ";
    bool first(true);
    for (unsigned iB(0); iB != panda::GenParticle::nStatusFlags; ++iB) {
      if (!statusBits[iB])
        continue;

      if (first) {
        ss << "{";
        first = false;
      }
      else
        ss << ",";
      ss << iB;
    }
    if (!first)
      ss << "}";
  }

  info_ = ss.str();

  for (auto* daughter : daughters)
    daughter->generateInfo_();
}

std::string
PNode::doPrint_(std::vector<std::string>* _spacers, bool _firstDaughter, bool _lastDaughter) const
{
  std::stringstream ss;

  if (_spacers && !_firstDaughter) {
    for (std::string& s : *_spacers)
      ss << s;
  }

  ss << "+ " << info_ << " ";

  std::string spacer(info_.size() + 1, ' ');
  if (_lastDaughter)
    spacer = "  " + spacer;
  else
    spacer = "| " + spacer;

  if (_spacers)
    _spacers->push_back(spacer);
  else
    _spacers = new std::vector<std::string>(1, spacer);

  if (daughters.size() != 0) {
    for (unsigned iD(0); iD != daughters.size(); ++iD)
      ss << daughters[iD]->doPrint_(_spacers, iD == 0, iD == daughters.size() - 1);
  }
  else {
    ss << std::endl;
    if (_lastDaughter && _spacers) {
      for (std::string& s : *_spacers)
        ss << s;
      ss << std::endl;
    }
  }

  _spacers->pop_back();
  if (_spacers->size() == 0)
    delete _spacers;

  return ss.str();
}
