#!/bin/bash

for d in /root/workspace/swapping-autoencoder-pytorch/checkpoints/*; do

    echo \# Inspect directory: $d

    link=$(find $d | grep latest)
    target=$d/$(readlink $link)

    for f in $d/*.pth; do
        if [[ $f == $link || $f == $target ]]; then
            continue
        fi
        rm $f -v
    done
done
