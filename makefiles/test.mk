.PHONY: test

## test_data.%: % in CIFAR10, CIFAR100, MNIST, Mini-Imagenet
test_data.%:
	pytest -k $* sparselearning/tests/test_data.py -s

## test.%: % in struct_sparse, mask_loading_saving
test.%:
	pytest sparselearning/tests/test_$*.py -s

## test: Run all tests
test: test_data.CIFAR10 test_data.CIFAR100 test.mask_loading_saving test.struct_sparse