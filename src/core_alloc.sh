#! /bin/bash

# once per terminal session:
sudo sh -c "echo $$ > /sys/fs/cgroup/amir_bench/cgroup.procs"

# verify before trusting it:
grep Cpus_allowed_list /proc/self/status     # must say: 16-31