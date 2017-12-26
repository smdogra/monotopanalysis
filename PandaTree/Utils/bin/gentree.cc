#include "PandaTree/Utils/interface/PNode.h"
#include "PandaTree/Objects/interface/Event.h"

#include "TFile.h"
#include "TTree.h"
#include "TPRegexp.h"

#include <iostream>
#include <vector>

struct PNodePanda : public PNode {
  short motherPanda{-1};

  PNodePanda(panda::GenParticle const& _part) : PNode() {
    pdgId = _part.pdgid;
    status = _part.finalState ? 1 : 2;
    statusBits = std::bitset<15>(_part.statusFlags);
    mass = _part.m();
    pt = _part.pt();
    eta = _part.eta();
    phi = _part.phi();
    motherPanda = _part.parent.idx();

    ownDaughters = false;
  }
};

int
main(int argc, char* argv[])
{
  PNode::gMomentumDispMode = PNode::kShowAllP;
  PNode::gMassDispMode = PNode::kShowHardScatM;

  if (argc != 3) {
    std::cerr << "Usage: gentree PATH EVENT" << std::endl;
    std::cerr << " EVENT can be an event id (run:lumi:event) or a number of events to display." << std::endl;
    return 0;
  }

  auto* source(TFile::Open(argv[1]));
  if (!source || source->IsZombie())
    return 1;

  auto* tree(dynamic_cast<TTree*>(source->Get("events")));
  if (!tree) {
    std::cerr << "No valid events tree found" << std::endl;
    return 1;
  }

  auto* matches(TPRegexp("^([0-9]+):([0-9]+):([0-9]+)$").MatchS(argv[2]));
  long nEntries(0);
  unsigned runNumber(0);
  unsigned lumiNumber(0);
  unsigned eventNumber(0);
  if (matches->GetEntries() == 0)
    nEntries = TString(argv[2]).Atoi();
  else {
    nEntries = -1;
    runNumber = TString(matches->At(1)->GetName()).Atoi();
    lumiNumber = TString(matches->At(2)->GetName()).Atoi();
    eventNumber = TString(matches->At(3)->GetName()).Atoi();
  }
  delete matches;

  panda::Event event;
  event.setAddress(*tree, {"runNumber", "lumiNumber", "eventNumber", "genParticles"});

  long iEntry(0);
  while (iEntry != nEntries && event.getEntry(*tree, iEntry++) > 0) {
    if (runNumber != 0 && (event.runNumber != runNumber || event.lumiNumber != lumiNumber || event.eventNumber != eventNumber))
      continue;

    std::vector<PNodePanda> pnodes;
    pnodes.reserve(event.genParticles.size());

    for (auto& gen : event.genParticles)
      pnodes.emplace_back(gen);

    std::vector<PNode*> rootNodes;
    for (auto& pnode : pnodes) {
      if (pnode.motherPanda == -1) {
        rootNodes.push_back(&pnode);
      }
      else {
        PNodePanda& mother(pnodes[pnode.motherPanda]);
        pnode.mother = &mother;
        mother.daughters.push_back(&pnode);
      }
    }

    std::cout << "[" << event.runNumber << ":" << event.lumiNumber << ":" << event.eventNumber << "]" << std::endl;
    for (auto* rootNode : rootNodes) {
      rootNode->pruneDaughters();
      std::cout << rootNode->print() << std::endl;
    }
    std::cout << std::endl << std::endl;
  }

  return 0;
}
