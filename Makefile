default: linter tests

all: linter tests docs dist

linter:
	flake8 encodec && mypy encodec

tests:
	python3 -m encodec.binary
	python3 -m encodec.compress
	python3 -m encodec.model
	python3 -m encodec.modules.seanet
	python3 -m encodec.msstftd
	python3 -m encodec.quantization.ac
	python3 -m encodec.balancer
	test ! -f test_24k_decompressed.wav || rm test_24k_decompressed.wav; \
		python3 -m encodec test_24k.wav test_24k.ecdc -f && \
		python3 -m encodec test_24k.ecdc test_24k_decompressed.wav -f
	test ! -f test_48k_decompressed.wav || rm test_48k_decompressed.wav; \
		python3 -m encodec test_48k.wav test_48k.ecdc -f -q && \
		python3 -m encodec test_48k.ecdc test_48k_decompressed.wav -f -q

docs:
	pdoc3 --html -o docs -f encodec

dist: docs
	python3 setup.py sdist

clean:
	rm -r docs dist *.egg-info

live:
	pdoc3 --http : encodec

.PHONY: linter tests docs dist
