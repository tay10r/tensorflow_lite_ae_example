.PHONY: all
all: deploy
	./deploy

deploy: deploy.cpp encoder.cpp decoder.cpp

encoder.cpp: encoder.tflite
	xxd -i $< >$@

decoder.cpp: decoder.tflite
	xxd -i $< >$@

encoder.tflite decoder.tflite: generate train.py
	./generate
	./train.py

generate: generate.cpp
