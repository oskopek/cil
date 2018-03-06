

cat $1 | perl -pe 's/<user>/ U /g' \
					| perl -pe 's/<url>/ L /g' \
					| sed -e 's/^[ \t]*//'
					
					#
					#| sed -r 's/_[_]*/S/g' \
					#| sed -r 's/ [ ]*/_/g' \
					#| sed -r 's/(\S)/ \1/g'\
					#| sed -e 's/^[ \t]*//'\
					#| sed -r 's/[_]*$//g' \