#! /bin/bash

set -e

python make-fast-alg-figures.py
for i in media/bhut-*.pdf; do
  if ! [[ $i = *crop* ]]; then
    pdfcrop $i
  fi
done

if [[ "$1" = "watch" ]]; then
  git ls-files | entr ./run-org-conversion.sh
else
  ./run-org-conversion.sh
fi

echo "NOTES BUILD SCCESSFULLY COMPLETED"
