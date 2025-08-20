reinstall_package:
	@pip uninstall -y python-parallelism || :
	@pip install -e .
