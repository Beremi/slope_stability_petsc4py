PDF=presentation_petsc_rewrite.pdf
SOURCE=presentation_petsc_rewrite.tex

.PHONY: pdf clean preview distclean

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=$(basename $(PDF)) $(SOURCE)

preview: pdf
	mkdir -p preview
	rm -f preview/slide-*.png
	pages=$$(pdfinfo $(PDF) 2>/dev/null | awk '/^Pages:/ {print $$2}'); \
	for i in $$(seq 1 $$pages); do \
		out=$$(printf "preview/slide-%02d.png" $$i); \
		gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -dFirstPage=$$i -dLastPage=$$i -r160 -sOutputFile=$$out $(PDF); \
	done

clean:
	rm -f $(basename $(PDF)).aux \
	      $(basename $(PDF)).fdb_latexmk \
	      $(basename $(PDF)).fls \
	      $(basename $(PDF)).log \
	      $(basename $(PDF)).nav \
	      $(basename $(PDF)).out \
	      $(basename $(PDF)).snm \
	      $(basename $(PDF)).toc \
	      $(basename $(PDF)).synctex.gz
	rm -f preview/slide-*.png

distclean: clean
	rm -f $(PDF)
