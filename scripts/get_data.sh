#!/bin/bash
set -eou pipefail
mkdir -p ../dataset
wget -e robots=off --recursive --no-clobber --page-requisites --html-extension --convert-links --restrict-file-names=windows --no-parent --accept=html -P ../dataset https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html
