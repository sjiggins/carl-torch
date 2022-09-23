# CERN Grid

## Obtain the GRID certificate

Install the keys into `$HOME/.globus` using the Grid certificate. You can follow [this manual](https://twiki.cern.ch/twiki/bin/view/Main/UsingUSAtlasGrid).

## Activate the certificate

```bash
voms-proxy-init -voms atlas
```

## Upload the samples using Rucio

Activate Rucio by running

```bash
lsetup rucio
```

and inspect the datasets on the grid

```bash
rucio --list-dids user.<username>:*
```

You can upload files via

```bash
rucio upload <path/to/local/file.root> --rse <RSE>
```
where `<RSE>` is the Rucio Storage Element. List of all RSEs can be found by running
```bash
rucio list-rses
```
The input data can be saved on any scratchdisk. However, beaware that the data storage is not permanent.

## Submitting jobs to the Grid

The jobs can be submitted via `panda`.

```bash
lsetup panda
```

```bash
prun --exec "source gridExec.sh %RNDM:0 %IN" --outDS user.<username>.<outputName> \
--inDS user.<username>:user.<username>.<inputName> --outputs "output.tar" \
--noBuild --maxAttempt 1 --forceStaged --architecture "&nvidia" --nCore 1 \
--site UKI-LT2-QMUL --nFilesPerJob 2

```
and can be monitored via [BigPanda](https://bigpanda.cern.ch/). You should receive and email after 

## Downloading the results
The results can be downloaded using Rucio

```bash
rucio download user.<username>.<outputName>.log           # log files
rucio download user.<username>.<outputName>_output.tar    # models and data
```
