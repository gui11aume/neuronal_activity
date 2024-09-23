.PHONY: train
train:
	CUDA_VISIBLE_DEVICES=0 dvc repro

.PHONY: clean
clean:
	rm -rf trained.pt
