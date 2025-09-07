reinstall_package:
	@pip uninstall -y python-parallelism || :
	@pip install -e .

test:
	pytest tests
	@echo "All tests passed!"
