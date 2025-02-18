#!/bin/bash
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e
# THIS FILE ASSUMES IT IS RUN INSIDE THE tests/tests_<package> DIRECTORY

# Batch size for testing: Determines how many standalone test invocations run in parallel
# It can be set through the env variable PL_STANDALONE_TESTS_BATCH_SIZE and defaults to 6 if not set
test_batch_size="${PL_STANDALONE_TESTS_BATCH_SIZE:-5}"
source="${PL_STANDALONE_TESTS_SOURCE:-"lightning"}"
# this is the directory where the tests are located
test_dir=$1 # parse the first argument
COLLECTED_TESTS_FILE="collected_tests.txt"

ls -lh .  # show the contents of the directory

# this environment variable allows special tests to run
export PL_RUN_STANDALONE_TESTS=1
# python arguments
defaults=" -m coverage run --source ${source} --append -m pytest --no-header -v -s --timeout 120 "
echo "Using defaults: ${defaults}"

# get the list of parametrizations. we need to call them separately. the last two lines are removed.
# note: if there's a syntax error, this will fail with some garbled output
python3 -um pytest $test_dir -q --collect-only --pythonwarnings ignore 2>&1 > $COLLECTED_TESTS_FILE
# early terminate if collection failed (e.g. syntax error)
if [[ $? != 0 ]]; then
  cat $COLLECTED_TESTS_FILE
  exit 1
fi

# removes the last line of the file
sed -i '$d' $COLLECTED_TESTS_FILE

# Get test list and run each test individually
tests=($(grep -oP '\S+::test_\S+' "$COLLECTED_TESTS_FILE"))
test_count=${#tests[@]}
# present the collected tests
printf "collected $test_count tests:\n-------------------\n"
# replace space with new line
echo "${tests[@]}" | tr ' ' '\n'
printf "\n===================\n"

# if test count is one print warning
if [[ $test_count -eq 1 ]]; then
  printf "WARNING: only one test found!\n"
elif [ $test_count -eq 0 ]; then
  printf "ERROR: no tests found!\n"
  exit 1
fi

# clear all the collected reports
rm -f parallel_test_output-*.txt  # in case it exists, remove it


status=0 # reset the script status
report="" # final report
pids=() # array of PID for running tests
test_ids=() # array of indexes of running tests
printf "Running $test_count tests in batches of $test_batch_size\n"
for i in "${!tests[@]}"; do
  # remove initial "tests/" from the test name
  test=${tests[$i]/tests\//}
  printf "Running test $((i+1))/$test_count: $test\n"

  # execute the test in the background
  # redirect to a log file that buffers test output. since the tests will run in the background,
  # we cannot let them output to std{out,err} because the outputs would be garbled together
  python3 ${defaults} "$test" 2>&1 > "standalone_test_output-$i.txt" &
  test_ids+=($i) # save the test's id in an array with running tests
  pids+=($!) # save the PID in an array with running tests

  # if we reached the batch size, wait for all tests to finish
  if (( (($i + 1) % $test_batch_size == 0) || $i == $test_count-1 )); then
    printf "Waiting for batch to finish: $(IFS=' '; echo "${pids[@]}")\n"
    # wait for running tests
    for j in "${!test_ids[@]}"; do
      i=${test_ids[$j]} # restore the global test's id
      pid=${pids[$j]} # restore the particular PID
      test=${tests[$i]} # restore the test name
      printf "Waiting for $tests >> standalone_test_output-$i.txt (PID: $pid)\n"
      wait -n $pid
      # get the exit status of the test
      test_status=$?
      # add row to the final report
      report+="Ran\t$test\t>> exit:$test_status\n"
      if [[ $test_status != 0 ]]; then
        # show the output of the failed test
        cat "standalone_test_output-$i.txt"
        # Process exited with a non-zero exit status
        status=$test_status
      fi
    done
    test_ids=()  # reset the test's id array
    pids=()  # reset the PID array
  fi
done

# echo test report
printf '=%.s' {1..80}
printf "\n$report"
printf '=%.s' {1..80}
printf '\n'

# exit with the worst test result
exit $status
