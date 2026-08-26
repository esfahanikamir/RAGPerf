#!/bin/bash

# Create a new cgroup for the benchmark (if it doesn't already exist).
sudo mkdir -p /sys/fs/cgroup/amir_bench

# Restrict this cgroup to physical CPUs 16-31.
# Adjust this range if your system uses a different CPU numbering.
echo 16-31 | sudo tee /sys/fs/cgroup/amir_bench/cpuset.cpus

# Specify the NUMA memory node(s) the cgroup is allowed to use.
# Most single-socket systems use node 0.
echo 0 | sudo tee /sys/fs/cgroup/amir_bench/cpuset.mems

# Move the current shell into the cgroup so all child processes inherit it.
echo $$ | sudo tee /sys/fs/cgroup/amir_bench/cgroup.procs

# Verify that the shell is now restricted to the requested CPUs.
grep Cpus_allowed_list /proc/self/status