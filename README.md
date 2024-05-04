# Equivariant Quantum Graph Neural Network for Small Molecules 

This repository show the code to reproduce training and testing of Quantum Graph Neural Network to predict properties of small molecules. Our model is compared with the similar classical Graph Neural Network.

Code can be run as a py file (`src/train.py`) and configured with command line arguments


There are 2 available models: quantum and classical GNN, that can be chosen by setting `--quantum` to `true` or `false`.


`--feature_idx` parameter defines which property of molecules from QM9 dataset will the model train to predict. We used `4` which is  $\Delta \epsilon$ -- difference between HOMO and LUMO energies.


Interactive run logs here https://wandb.ai/amirfvb/graphQNN


<!-- В sota статье по QM9 датасету. homo, lumo and gap (2, 3, 4)


  ---------------------------------------------------------------------------------------------------------------
  Target   Property                     Description                       Unit
  -------- ---------------------------- --------------------------------- ---------------------------------------
  0        $\mu$                        Dipole moment                     $\textrm{D}$

  1        $\alpha$                     Isotropic polarizability          ${a_0}^3$

  2        $\epsilon_{\textrm{HOMO}}$   Highest occupied molecular        $\textrm{eV}$
                                        orbital energy                    

  3        $\epsilon_{\textrm{LUMO}}$   Lowest unoccupied molecular       $\textrm{eV}$
                                        orbital energy                    

  4        $\Delta \epsilon$            Gap between                       $\textrm{eV}$
                                        $\epsilon_{\textrm{HOMO}}$ and    
                                        $\epsilon_{\textrm{LUMO}}$        

  5        $\langle R^2 \rangle$        Electronic spatial extent         ${a_0}^2$

  6        $\textrm{ZPVE}$              Zero point vibrational energy     $\textrm{eV}$

  7        $U_0$                        Internal energy at 0K             $\textrm{eV}$

  8        $U$                          Internal energy at 298.15K        $\textrm{eV}$

  9        $H$                          Enthalpy at 298.15K               $\textrm{eV}$

  10       $G$                          Free energy at 298.15K            $\textrm{eV}$

  11       $c_{\textrm{v}}$             Heat capavity at 298.15K          $\frac{\textrm{cal}}{\textrm{mol K}}$

  12       $U_0^{\textrm{ATOM}}$        Atomization energy at 0K          $\textrm{eV}$

  13       $U^{\textrm{ATOM}}$          Atomization energy at 298.15K     $\textrm{eV}$

  14       $H^{\textrm{ATOM}}$          Atomization enthalpy at 298.15K   $\textrm{eV}$

  15       $G^{\textrm{ATOM}}$          Atomization free energy at        $\textrm{eV}$
                                        298.15K                           

  16       $A$                          Rotational constant               $\textrm{GHz}$

  17       $B$                          Rotational constant               $\textrm{GHz}$

  18       $C$                          Rotational constant               $\textrm{GHz}$
  --------------------------------------------------------------------------------------------------------------- -->
