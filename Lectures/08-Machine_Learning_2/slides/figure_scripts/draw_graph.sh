#! /bin/bash

tempdir=$(mktemp -d)
trap 'rm -rf $tempdir' EXIT
pdf_file="$tempdir/graph.pdf"
dot "$1" -Tpdf > "$pdf_file"
pdfcrop --margins "0 0 0 0" "$pdf_file" "$pdf_file"
cat "$pdf_file"
