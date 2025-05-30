#!/usr/bin/env bash

set -euo pipefail

thisdir=$(readlink -f $(dirname "${0}"))
proxycmd="sbatch ${thisdir}/11_bfconvert_proxy"

indir="${1}"

if [ -z "${indir}" ]; then
    echo "must supply a file!" >&2
    exit 1
fi

echo $(find -L "${indir}" -type f -name "*.nd2")
find -L "${indir}" -type f -name "*.nd2" -print0 | xargs -0 -n1 $proxycmd
