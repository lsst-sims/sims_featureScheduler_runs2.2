#!/bin/bash


find ../ -type f \( -iname "*10yrs.db" ! -path "*technical*" \) | xargs -I'{}' ln -s '{}' .
generate_ss
cat ss_script.sh > maf_ss.sh
generate_ss --pop vatiras_granvik_10k
cat ss_script.sh >> maf_ss.sh
split -l 56  maf_ss.sh sub_
python generate_slurms.py
