# CERN Grid

## Obtain the GRID certificate

Install the keys into `$HOME/.globus` using the Grid certificate, according to [this manual](https://twiki.cern.ch/twiki/bin/view/Main/UsingUSAtlasGrid).

## Activate the certificate

```bash
voms-proxy-init -voms atlas
```

## Upload the samples using Rucio

Activate Rucio

```bash
lsetup rucio
```

Inspect the datasets on the grid

```bash
rucio --list-dids user.<username>:*
```

To be completed...

## Submitting jobs to the Grid

```bash
prun --exec "source gridExec.sh %IN" --outDS user.<username>.<outputName> --inDS user.<username>:user.<username>.CARLtraining --outputs "output.tar" --noBuild --maxAttempt 1 --forceStaged --architecture "&nvidia" --nCore 1 --memory 10000
```