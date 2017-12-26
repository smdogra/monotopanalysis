#include "../interface/genericTree.h"

void
genericTree::RemoveBranches(std::vector<TString> droppable, 
                            std::vector<TString> keeppable)
{

  for (auto &s : droppable)
    r_droppable.emplace_back(s);
  for (auto &s : keeppable)
    r_keeppable.emplace_back(s);


}

bool
genericTree::Book(TString bname, void *address, TString leaf)
{

  if (!treePtr)
    return false;

  bool mustKeep = false; // if there's an override
  for (auto &r : r_keeppable) {
    if (bname.Contains(r)) {
      mustKeep = true;
      break;
    }
  }

  if (!mustKeep) {
    for (auto &r : r_droppable) {
      if (bname.Contains(r)) 
        return false;
    }
  }

  treePtr->Branch(bname,address,leaf);
  return true;

}
