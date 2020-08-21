"""
FIFO script runner.
    1. able to modify the script you want to run during the execution.
    2. supports "export" environment variables
    3. supports execution of multiple scripts

Preliminary version.
!!! Please use 'Ctrl+\' to stop the script files.

"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
import os, subprocess
import argparse


def is_script(script_filename):
    if script_filename[-3:] == '.sh':
        return True
    else:
        return False


def run_scripts(script_filename):
    """
    you can put a file. which contains some scripts you want to run. For example,

    "a1.test"
    echo a
    echo b

    echo c

    if you run this file,
    a
    b
    c

    will be printed.

    This will modify script_file, for example,

    "a1.test"
    echo a
    echo b
    ===running above ... ===

    echo c

    This script_file means that it already executed 'echo a' and it is executing 'echo b'
    You can add script you want to run, below the "===running above===" string

    :param script_filename:
    :return:
    """
    runstr = "===running above==="

    finished = False
    while not finished:
        to_run_script = None
        script_file = open(script_filename, "r")
        scripts = script_file.readlines()
        scripts = [script.strip() for script in scripts]
        script_file.close()

        for i in range(len(scripts)):
            script = scripts[i]
            if (script == runstr):
                if i == len(scripts) - 1:
                    to_run_script = "echo test done"
                    assert scripts.pop() == runstr
                    finished = True
                else:
                    to_run_script = scripts[i + 1]
                    """
                    Below will change something like

                    python a.py
                    ===running above===
                    python b.py

                    to

                    python a.py
                    python b.py
                    ===running above===
                    """
                    scripts[i] = to_run_script
                    scripts[i + 1] = runstr
                break

        if to_run_script is None:  # When the first call of this python file. But have to allow null script
            to_run_script = scripts[0]
            scripts.insert(1, runstr)

        script_file = open(script_filename, "w")
        scripts = "\n".join(scripts)
        script_file.write(scripts)
        script_file.close()

        if to_run_script:  # Pass null script
            # TODO: os.system has many advantages over subprocess, for example can use cd dir \n , or >>, etc...
            if to_run_script.startswith('export'):
                to_run_script = to_run_script.replace('export', '')
                first_equal = to_run_script.find('=')
                env, val = to_run_script[:first_equal], to_run_script[(first_equal + 1):]

                val_with_exported_values = str(os.popen("echo %s" % val.strip()).read())
                os.environ[env.strip()] = val_with_exported_values.strip()
            else:
                to_run_script = to_run_script.replace('\"', '\'')
                subprocess.call(['python', '-c', 'import os; os.system("%s")' % to_run_script])
                # os.system(to_run_script)


def check_and_run_script(script_file):
    if is_script(script_file):
        run_scripts(script_file)
    else:
        print("given file doesn't have extension name .sh")


if __name__ == '__main__':
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description="FIFO Tester.")
    # TODO: maybe we can add to test a folder later. done scripts can be moved to other folder like 'done'
    parser.add_argument('script', nargs='+', help="Script you want to run FIFO-wise")
    parser.add_argument('-j', '--num_workers', type=int, default=-1, help='num_workers you want to use')
    args = parser.parse_args()

    if args.num_workers < 0:
        args.num_workers = len(args.script)
    pool = Pool(args.num_workers)
    print("Using %d cores, executing %d scripts.\n%s" % (args.num_workers, len(args.script), args.script))

    results = []
    results.append(pool.map_async(check_and_run_script, args.script))
    while len(results) > 0:
        result = results.pop(0)
        result.get()
    pool.close()
    pool.join()
    print("done")
