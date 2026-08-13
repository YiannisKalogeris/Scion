# Intentionally empty.
#
# Present so that pytest's default "prepend" import mode gives test modules a
# fully qualified name. Without it, every test file in the repository would need
# a globally unique basename -- so tests/fem/test_elements.py and
# tests/mpm/test_elements.py would collide at collection time.
