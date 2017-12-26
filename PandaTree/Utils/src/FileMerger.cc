#include "../interface/FileMerger.h"
#include "../../Objects/interface/Event.h"
#include "../../Objects/interface/Run.h"

#include "TFile.h"
#include "TSystem.h"
#include "TRegexp.h"
#include "TError.h"
#include "TEntryList.h"
#include "TObjArray.h"
#include "TH1.h"

#include <iostream>

void
panda::FileMerger::addInput(char const* _path)
{
  // Implementation from TChain::Add

  TString basename(_path);

  // case with one single file
  if (!basename.MaybeWildcard()) {
    if (printLevel_ >= 1)
      std::cout << " " << basename << std::endl;
    paths_.push_back(basename);
    return;
  }
 
  // wildcarding used in name

  Int_t slashpos = basename.Last('/');
  TString directory;
  if (slashpos>=0) {
    directory = basename(0,slashpos); // Copy the directory name
    basename.Remove(0,slashpos+1);      // and remove it from basename
  } else {
    directory = gSystem->UnixPathName(gSystem->WorkingDirectory());
  }
 
  const char *file;
  const char *epath = gSystem->ExpandPathName(directory.Data());
  void *dir = gSystem->OpenDirectory(epath);
  delete [] epath;
  if (dir) {
    TRegexp re(basename,kTRUE);
    while ((file = gSystem->GetDirEntry(dir))) {
      if (!strcmp(file,".") || !strcmp(file,"..")) continue;
      TString s = file;
      if ( (basename!=file) && s.Index(re) == kNPOS) continue;
      if (printLevel_ >= 1)
        std::cout << " " << directory << "/" << s << std::endl;
      paths_.push_back(directory + "/" + s);
    }
    gSystem->FreeDirectory(dir);
  }
}

void
panda::FileMerger::selectBranches(utils::BranchList const& _blist, Bool_t _onRead/* = kFALSE*/, TreeType _ttype/* = kEvent*/)
{
  branchList_[_ttype] = _blist;
  applyBranchListOnRead_[_ttype] = _onRead;
}

unsigned
panda::FileMerger::merge(char const* _outPath, long _nEvents/* = -1*/)
{
  auto* outputFile(TFile::Open(_outPath, "recreate"));
  if (!outputFile || outputFile->IsZombie())
    throw std::runtime_error("Invalid path");

  // same run can appear in multiple files
  // assuming everything except the hlt index pointer is identical between these entries
  // we save the first occurence (file index, entry number) and read them out later
  // (currently Run only has the run number and HLT index, so this is a total overkill)
  std::map<UInt_t, std::pair<unsigned, long>> runSources{};
  std::map<UInt_t, unsigned> runToMenuMap{};
  std::vector<TString> hltMenuList{};
  std::vector<std::vector<TString>> hltPathsList{};

  // Assuming the runs and hlt tree contents are unchanged
  panda::Run run;
  run.hlt.create();

  // loop over files to
  // . fill events (simple)
  // . collect run info
  // . collect hlt info

  long nTotal(0);
  long nRead(0);
  long nWritten(0);

  TObjArray mergeables;

  // output tree constructed only at the first valid input
  TTree* outEventTree(0);

  for (unsigned iS(0); iS != paths_.size(); ++iS) {
    auto& path(paths_[iS]);

    if (printLevel_ >= 2)
      std::cout << "Opening " << path << std::endl;

    TFile* source(0);

    auto originalErrorIgnoreLevel(gErrorIgnoreLevel);
    if (timeout_ > 0) {
      // silence the TFile error message
      gErrorIgnoreLevel = kError + 1;
    }

    unsigned const tryEvery(30);
    for (unsigned iAtt(0); iAtt <= timeout_ / tryEvery; ++iAtt) {
      source = TFile::Open(path);
      if (source) {
        if (!source->IsZombie())
          break;
        delete source;
      }

      gSystem->Sleep(tryEvery * 1000.);
    }

    if (timeout_ > 0)
      gErrorIgnoreLevel = originalErrorIgnoreLevel;

    if (!source || source->IsZombie()) {
      std::cerr << "Cannot open file " << path << std::endl;
      delete source;
      throw std::runtime_error("source");
    }

    auto* inEventTree(static_cast<TTree*>(source->Get("events")));
    if (!inEventTree) {
      std::cerr << "File " << path << " is not a valid input" << std::endl;
      throw std::runtime_error("events");
    }

    if (eventSelection_ != "") {
      inEventTree->Draw(">>elist", eventSelection_, "entrylist");
      auto* elist(static_cast<TEntryList*>(gDirectory->Get("elist")));
      if (elist)
        inEventTree->SetEntryList(elist);
      else {
        std::cerr << "Failed to compile event selection " << eventSelection_ << std::endl;
        throw std::runtime_error("event selection");
      }
    }

    if (printLevel_ >= 2) {
      std::cout << "Event tree has " << inEventTree->GetEntries();
      if (inEventTree->GetEntryList())
        std::cout << " (" << inEventTree->GetEntryList()->GetN() << " selected)";
      std::cout << " events" << std::endl;
    }

    // first event - book a tree
    if (!outEventTree) {
      // pick up everything that is not events, runs, or hlt

      std::set<TString> keyNames;
      for (auto* key : *source->GetListOfKeys()) {
        TString keyName(key->GetName());

        if (keyNames.count(keyName) != 0)
          continue;

        if (keyName == "events" || keyName == "runs" || keyName == "hlt")
          continue;

        keyNames.insert(keyName);

        outputFile->cd();
        auto* obj(source->Get(keyName));
        TObject* clone(0);
        if (obj->InheritsFrom(TTree::Class())) {
          // if the obj is a tree, a simple clone will only copy the keys but not the tree data
          clone = static_cast<TTree*>(obj)->CloneTree(-1, "fast");
        }
        else
          clone = outputFile->CloneObject(obj);

        if (obj->InheritsFrom(TTree::Class()) || obj->InheritsFrom(TH1::Class()))
          mergeables.Add(clone);
        else {
          clone->Write();
          delete clone;
          delete obj;
        }
      }

      // list of branches in the original tree filtered/augmented by branchList_
      auto blist(utils::BranchList::makeList(*inEventTree));
      blist += branchList_[kEvent];

      if (printLevel_ >= 2)
        std::cout << "Creating output event tree" << std::endl;
      if (printLevel_ >= 3)
        std::cout << blist << std::endl;

      outputFile->cd();

      if (skimmer_) {
        // when a skimmer is given, book branches using the event object it uses
        outEventTree = new TTree("events", "Events");
        skimmer_->getEvent()->book(*outEventTree, blist);
      }
      else {
        // otherwise copy the branches
        for (auto* br : *inEventTree->GetListOfBranches()) {
          if (!utils::BranchName(br->GetName()).in(blist))
            inEventTree->SetBranchStatus(br->GetName(), false);
        }

        // branches vetoed in blist are turned off
        outEventTree = inEventTree->CloneTree(0);

        inEventTree->SetBranchStatus("*", true);
      }
    }
    else {
      // >= second input tree. Merge mergeable objects

      for (auto* mergeable : mergeables) {
        TString objName(mergeable->GetName());

        auto* obj(source->Get(objName));
        if (!obj) {
          std::cerr << objName << " missing from " << path << std::endl;
          continue;
        }

        if (obj->InheritsFrom(TTree::Class())) {
          // if the obj is a tree, a simple clone will only copy the keys but not the tree data
          static_cast<TTree*>(mergeable)->CopyEntries(static_cast<TTree*>(obj), -1, "fast");
        }
        else if (obj->InheritsFrom(TH1::Class()))
          static_cast<TH1*>(mergeable)->Add(static_cast<TH1*>(obj));
      }
    }

    if (printLevel_ >= 2)
      std::cout << "Binding input" << std::endl;

    // set input
    utils::BranchList blist({"*"});
    if (applyBranchListOnRead_[kEvent])
      blist += branchList_[kEvent];

    if (skimmer_) {
      // when a skimmer is given, set branch addresses to the event object it uses
      skimmer_->getEvent()->setAddress(*inEventTree, blist, true);
    }

    auto* inRunTree(static_cast<TTree*>(source->Get("runs")));

    std::set<UInt_t> savedRuns;

    if (printLevel_ >= 3)
      std::cout << "Processing events" << std::endl;

    nTotal += inEventTree->GetEntries();

    // fill events
    if (eventSelection_ == "" && !skimmer_) {
      // No event selection whatsoever; just copying entries one-to-one.
      // We can speed up the copy in this case by simply copying the baskets without decompression

      outEventTree->CopyEntries(inEventTree, -1, "fast");
      nRead += inEventTree->GetEntries();
      nWritten += inEventTree->GetEntries();

      if (printLevel_ > 0)
        (std::cout << "\rProcessed " << nRead << " events").flush();

      if (inRunTree) {
        // All runs are saved - just read the full list of runs from the runs tree

        run.setAddress(*inRunTree, {"runNumber"});
        long iEntry(0);
        while (inRunTree->GetEntry(iEntry++) > 0)
          savedRuns.insert(run.runNumber);

        inRunTree->ResetBranchAddresses();
      }
    }
    else {
      // Some sort of event selection exists. Need to unpack and process each event.

      unsigned runNumber(0);
      if (!skimmer_)
        inEventTree->SetBranchAddress("runNumber", &runNumber);

      Long64_t iEntry(0);
      while (nRead != _nEvents) {
        Long64_t entryNumber(inEventTree->GetEntryNumber(iEntry++));
        if (entryNumber < 0)
          break;

        switch (printLevel_) {
        case 0:
          break;
        case 1:
          if (nRead % 100000 != 0)
            break;
        case 2:
          if (nRead % 1000 != 0)
            break;
        case 3:
          if (nRead % 10 != 0)
            break;
        default:
          (std::cout << "\rProcessed " << nRead << " events").flush();
        }
  
        ++nRead;

        if (skimmer_) {
          auto& event(*skimmer_->getEvent());
          if (event.getEntry(*inEventTree, entryNumber) <= 0)
            break;
          if (!skimmer_->skim())
            continue;

          event.fill(*outEventTree);

          runNumber = event.runNumber;
        }
        else {
          if (inEventTree->GetEntry(entryNumber) <= 0)
            break;

          outEventTree->Fill();
        }

        ++nWritten;
  
        savedRuns.insert(runNumber);
      }
    }

    if (printLevel_ >= 2)
      std::cout << std::endl;

    if (inRunTree) {
      // Now we process the runs

      // collect HLT and run info

      if (printLevel_ >= 3)
        std::cout << "Run tree found" << std::endl;
      
      if (applyBranchListOnRead_[kRun])
        run.setAddress(*inRunTree, branchList_[kRun]);
      else
        run.setAddress(*inRunTree);

      auto* inHLTTree(static_cast<TTree*>(source->Get("hlt")));

      if (inHLTTree) {
        if (printLevel_ >= 3)
          std::cout << "HLT tree found" << std::endl;

        inHLTTree->SetBranchAddress("menu", &run.hlt.menu);
        inHLTTree->SetBranchAddress("paths", &run.hlt.paths);
      }

      if (printLevel_ >= 3)
        std::cout << "Processing runs" << std::endl;

      long iEntry(0);
      while (run.getEntry(*inRunTree, iEntry++) > 0) {
        if (savedRuns.count(run.runNumber) == 0)
          continue;

        if (printLevel_ >= 3)
          std::cout << " run " << run.runNumber << std::endl;

        if (runSources.count(run.runNumber) == 0)
          runSources.emplace(run.runNumber, std::make_pair(iS, iEntry - 1));

        if (inHLTTree) {
          inHLTTree->GetEntry(run.hltMenu);

          if (printLevel_ >= 3)
            std::cout << "  HLT menu name " << *run.hlt.menu << std::endl;

          auto mItr(std::find(hltMenuList.begin(), hltMenuList.end(), *run.hlt.menu));
          unsigned menuId(0);
          if (mItr == hltMenuList.end()) {
            hltMenuList.push_back(*run.hlt.menu);
            hltPathsList.emplace_back(run.hlt.paths->begin(), run.hlt.paths->end());
            menuId = hltMenuList.size() - 1;
          }
          else
            menuId = mItr - hltMenuList.begin();

          auto hltItr(runToMenuMap.find(run.runNumber));
          if (hltItr == runToMenuMap.end())
            runToMenuMap.emplace(run.runNumber, menuId);
          else if (hltItr->second != menuId) {
            std::cerr << hltItr->second << " " << menuId << std::endl;
            std::cerr << "Inconsistent HLT menu found for run " << run.runNumber << " in file " << path << std::endl;
            throw std::runtime_error("HLT");
          }
        }
      }
    }
    else if (runSources.size() != 0) {
      std::cerr << "Run tree missing in file " << path << std::endl;
      throw std::runtime_error("runs");
    }

    delete source;

    if (printLevel_ >= 2)
      std::cout << "Closed " << path << std::endl;
  }

  if (printLevel_ == 1) // last print is in the event loop
    std::cout << std::endl;

  std::cout << "Number of events in : " << nTotal << " (" << nRead << " read)" << std::endl;
  std::cout << "Number of events out: " << nWritten << std::endl;

  if (outEventTree) {
    outputFile->cd();
    outEventTree->Write();
    delete outEventTree;
  }

  for (auto* mergeable : mergeables) {
    outputFile->cd();
    mergeable->Write();
  }

  if (runSources.size() != 0) {
    TTree* outRunTree(0);

    for (auto& runSource : runSources) {
      auto* source(TFile::Open(paths_[runSource.second.first]));
      auto* inRunTree(static_cast<TTree*>(source->Get("runs")));

      if (!outRunTree) {
        outputFile->cd();
        outRunTree = new TTree("runs", "Runs");

        auto blist(utils::BranchList::makeList(*inRunTree));
        blist += branchList_[kRun];

        run.book(*outRunTree, blist);
      }

      if (applyBranchListOnRead_[kRun])
        run.setAddress(*inRunTree, branchList_[kRun]);
      else
        run.setAddress(*inRunTree);

      inRunTree->GetEntry(runSource.second.second);

      if (runToMenuMap.size() != 0)
        run.hltMenu = runToMenuMap[runSource.first];

      run.fill(*outRunTree);

      delete source;
    }

    outputFile->cd();
    outRunTree->Write();

    delete outRunTree;

    if (runToMenuMap.size() != 0) {
      outputFile->cd();
      TTree* outHLTTree(new TTree("hlt", "HLT"));

      outHLTTree->Branch("menu", "TString", &run.hlt.menu);
      outHLTTree->Branch("paths", "std::vector<TString>", &run.hlt.paths);

      for (unsigned iH(0); iH != hltMenuList.size(); ++iH) {
        *run.hlt.menu = hltMenuList[iH];
        auto& paths(hltPathsList[iH]);
        run.hlt.paths->assign(paths.begin(), paths.end());

        outHLTTree->Fill();
      }

      outputFile->cd();
      outHLTTree->Write();

      delete outHLTTree;
    }
  }

  delete outputFile;

  return nWritten;
}
