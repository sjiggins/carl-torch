// Macro for the preprocessing of inpùt for training the NN in CARL-Torch
// This macro performs the following changes to perform to the Grid output:
//   - Create lighter root files with the branches of interest
//   - Add branch with eventWeight*XS/sumW*1./R_accepted where R_accepted is the fraction of events used w.r.t. to the number of entries of the file
//   - Scale every slice to its XS ---> multiplying by XS[dsID]/(sum of XS for all dsIds) (This is not done for our case)
//   - Give two output files for each MC model
//
// This is a particular example applied to dijet samples for hadronisation uncertainty study
// The two hadronisation models are the Cluster and the String model
// - Cluster: DSIDs 364677-364685
// - String: DSIDs 364686-364694

void preprocessing_CARLTorch_input() {
  // Set input
  const int nFiles=18;// 18 DSIDs
  const int nFilesOUT=2;// 2 models
  string INPUT_PATH = Form("/eos/user/a/arubioji/CARL_GridOutput_test9");
  string INPUTFILES[nFiles];
  Float_t XS[nFiles];

  // XS*FiltEff for each DSID
  INPUTFILES[0] = "Hadronisation_Cluster_364677";
  XS[0] = 2.1406E+07*1.4419E-03*1E+03;//pb
  INPUTFILES[1] = "Hadronisation_Cluster_364678";
  XS[1] = 8.9314E+04*5.1230E-03*1E+03;
  INPUTFILES[2] = "Hadronisation_Cluster_364679";
  XS[2] = 9.2751E+03*5.6541E-04*1E+03;
  INPUTFILES[3] = "Hadronisation_Cluster_364680";
  XS[3] = 5.5101E+01*1.4974E-03*1E+03;
  INPUTFILES[4] = "Hadronisation_Cluster_364681";
  XS[4] = 1.6314E+00*2.4206E-02*1E+03;
  INPUTFILES[5] = "Hadronisation_Cluster_364682";
  XS[5] = 1.2842E-01*1.0820E-02*1E+03;
  INPUTFILES[6] = "Hadronisation_Cluster_364683";
  XS[6] = 2.7212E-02*3.6180E-03*1E+03;
  INPUTFILES[7] = "Hadronisation_Cluster_364684";
  XS[7] = 2.0583E-04*1.5966E-02*1E+03;
  INPUTFILES[8] = "Hadronisation_Cluster_364685";
  XS[8] = 3.5683E-05*3.3033E-03*1E+03;
  INPUTFILES[9] = "Hadronisation_String_364686";
  XS[9] = 2.1406E+07*1.4140E-03*1E+03;
  INPUTFILES[10] = "Hadronisation_String_364687";
  XS[10] = 8.9314E+04*5.1181E-03*1E+03;
  INPUTFILES[11] = "Hadronisation_String_364688";
  XS[11] = 9.2750E+03*5.6410E-04*1E+03;
  INPUTFILES[12] = "Hadronisation_String_364689";
  XS[12] = 5.5101E+01*1.4985E-03*1E+03;
  INPUTFILES[13] = "Hadronisation_String_364690";
  XS[13] = 1.6313E+00*2.4247E-02*1E+03;
  INPUTFILES[14] = "Hadronisation_String_364691";
  XS[14] = 1.2842E-01*1.0837E-02*1E+03;
  INPUTFILES[15] = "Hadronisation_String_364692";
  XS[15] = 2.7211E-02*3.6257E-03*1E+03;
  INPUTFILES[16] = "Hadronisation_String_364693";
  XS[16] = 2.0584E-04*1.5998E-02*1E+03;
  INPUTFILES[17] = "Hadronisation_String_364694";
  XS[17] = 3.5683E-05*3.3119E-03*1E+03;

  // Computing denominator for XS scaling
  //    This is aimed to scale the number of entries of each tree according to the XS of each DSID
  //    This is not feasible in our case since the max. and min. DSID XS differ in 13 orders of magnitude
  int total_XS_n=0;
  int total_XS_v=0;
  for (int ifile=0; ifile < nFiles; ifile++){
    if (ifile < (int)nFiles/2.) total_XS_n += XS[ifile];
    else total_XS_v += XS[ifile];
  }

  // Define variables to manage the trees, histos and files
  TFile* file[nFiles];
  TTree* tree[nFiles];
  TFile* file_light[nFiles];
  TTree* tree_light[nFiles];
  TH1D* h_sumW[nFiles];
  Float_t SumW[nFiles];
  Int_t Nentries[nFiles];

  TFile* fileOUT[nFilesOUT];
  TTree* treeOUT[nFilesOUT];

  // Defining variables for the new branches
  Float_t eventWeight;
  Float_t scaled_eventWeight;
  Float_t cache_XS, cache_sumW;
  Int_t cache_Nentries_file;
  Int_t cache_Nentries_accepted;
  eventWeight = 0.;
  scaled_eventWeight = 0.;
  cache_XS = 0.;
  cache_sumW = 0.;
  cache_Nentries_file = 0;
  cache_Nentries_accepted = 0;

  // Define TChain to store all the iput root files
  cout << "Filling TChain" <<endl;
  TChain chain("Tree");
  for (int ifile=0; ifile<nFiles; ifile++) {
    chain.Add(Form("%s/%s.root", INPUT_PATH.c_str(), INPUTFILES[ifile].c_str()));
  }

  //CREATE NEW BRANCH WITH SCALED WEIGHTS

  // Obtaining Nentries and sumW for each DSID

  TObjArray *fileElements=chain.GetListOfFiles();
  TIter next(fileElements);
  TChainElement *chEl=0;

  cout << "Obtaining Nentries and sumW for each DSID" <<endl;
  int ifile = 0;
  while (( chEl=(TChainElement*)next() )) {
    TFile file(chEl->GetTitle());
    std::cout << "Title of file " << ifile << ": " << chEl->GetTitle() << std::endl;
    // Retrieve sum of weights histo from input TFile
    h_sumW[ifile] = (TH1D*)file.Get("sumWeight");
    // Define Nentries and sumW
    Nentries[ifile] = h_sumW[ifile]->GetEntries();
    SumW[ifile] = h_sumW[ifile]->GetBinContent(1);
    std::cout << "File " << ifile << ", sumW: " << h_sumW[ifile]->GetBinContent(1) << ", Nentries: " << h_sumW[ifile]->GetEntries() << std::endl ;
    ifile++;
  }

  // Retrieving Trees from original files and cloning them into lighter ones
  cout << "Retrieving Trees from original files and cloning them into lighter ones" <<endl;
  ifile = 0;
  next.Reset();
  chEl=0;
  while (( chEl=(TChainElement*)next() )) {
    TFile file(chEl->GetTitle());
    std::cout << "File " << INPUTFILES[ifile].c_str() << std::endl;
    // Retrieve Tree from file
    tree[ifile] = (TTree*)file.Get("Tree");
    // Show total number of entries
    std::cout << "Total number of entries " << tree[ifile]->GetEntries() << std::endl;
    if (ifile < (int)nFiles/2.){std::cout << "Maximum number of entries to select according to XS: " << XS[ifile]/total_XS_n*tree[ifile]->GetEntries() << std::endl;}
    else {std::cout << "Maximum number of entries to select according to XS: " << XS[ifile]/total_XS_v*tree[ifile]->GetEntries() << std::endl;}

    // Set branches status. Set to status 1 only those branches that you want to keep
    tree[ifile]->SetBranchStatus("*",0);
    tree[ifile]->SetBranchStatus("eventWeight",1);
    tree[ifile]->SetBranchStatus("N*",1);
    tree[ifile]->SetBranchStatus("HT",1);
    tree[ifile]->SetBranchStatus("ptFracB",1);
    tree[ifile]->SetBranchStatus("ptFracC",1);
    tree[ifile]->SetBranchStatus("Jet*",1);
    tree[ifile]->SetBranchStatus("JetShape_ch",0);

    //Retrieve eventWeight branch
    tree[ifile]->SetBranchAddress("eventWeight", &eventWeight);
    // Define light TFile
    file_light[ifile] = new TFile(Form("%s/%s_light.root", INPUT_PATH.c_str(), INPUTFILES[ifile].c_str()),"recreate");
    // Clone Tree in an light Tree
    tree_light[ifile] = tree[ifile]->CloneTree(0);
    // Define new branch in the light tree
    tree_light[ifile]->Branch("scaled_eventWeight", &scaled_eventWeight);
    tree_light[ifile]->Branch("cache_XS", &cache_XS);
    tree_light[ifile]->Branch("cache_sumW", &cache_sumW);
    tree_light[ifile]->Branch("cache_Nentries_file", &cache_Nentries_file);
    tree_light[ifile]->Branch("cache_Nentries_accepted", &cache_Nentries_accepted);

    // Fill new tree entry by entry, including new branch
    Long64_t minEntries = (Long64_t)std::min(100000, (int)tree[ifile]->GetEntries());
    std::cout << "Fraction of events used wrt available entries of this file " << (float)minEntries/(float)tree[ifile]->GetEntries() << std::endl;
    for (Long64_t i=0;i<minEntries; i++) {
       tree[ifile]->GetEntry(i);

       if (((4<ifile)&&(ifile<9)) || ((13<ifile)&&(ifile<18))) scaled_eventWeight = eventWeight*XS[ifile]/SumW[ifile]*((float)tree[ifile]->GetEntries()/(float)minEntries);
       else scaled_eventWeight = eventWeight*XS[ifile]/Nentries[ifile]*((float)tree[ifile]->GetEntries()/(float)minEntries);

       cache_XS = XS[ifile];
       cache_sumW = SumW[ifile];
       cache_Nentries_file = (int)tree[ifile]->GetEntries();
       cache_Nentries_accepted = (int)minEntries;

       tree_light[ifile]->Fill();
    }
    // Storing light TFiles (ideally, this should be skipped)
    file_light[ifile]->Write();
    ifile++;
  }

  // Add Tress into Cluster and String
  cout << "Add Trees into two TList" << endl;
  TList *list0 = new TList;
  TList *list1 = new TList;
  for (int ifile=0; ifile<nFiles; ifile++) {
    if (ifile < int(nFiles/2.+0.5))
    {
      std::cout << "Adding to Cluster list the tree " << ifile << std::endl;
      list0->Add(tree_light[ifile]);
    }
    else if (ifile >= int(nFiles/2.+0.5))
    {
      std::cout << "Adding to String list the tree " << ifile << std::endl;
      list1->Add(tree_light[ifile]);
    }
    else
    {
      std::cout << "Something went wrong with adding the trees" << std::endl;
    }
  }

  // Merge them two have two output files
  std::cout << "Merge files from lists into 2 output files" << std::endl;
  std::cout << "Output file ClusterMC.root" << std::endl;
  TFile *fileOUT_0 = new TFile(Form("%s/ClusterMC.root", INPUT_PATH.c_str()),"recreate");
  TTree *treeOUT_0 = TTree::MergeTrees(list0);
  treeOUT_0->SetName("Tree");
  fileOUT_0->Write();

  std::cout << "Output file StringMC.root" << std::endl;
  TFile *fileOUT_1 = new TFile(Form("%s/StringMC.root", INPUT_PATH.c_str()),"recreate");
  TTree *treeOUT_1 = TTree::MergeTrees(list1);
  treeOUT_1->SetName("Tree");
  fileOUT_1->Write();

  for (int ifile=0; ifile<nFiles; ifile++){
    file_light[ifile]->Close();
  }

}
