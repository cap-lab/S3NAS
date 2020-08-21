# tester

A FIFO script runner. Preliminary version.

You can run a script file and can MODIFY the script file DURING experiment.

!!! USE **Ctrl+\\** to stop the executions

## Example

For example, you can make a script file like this
```.env
a.sh

export TRAIN_EPOCHS=10
python run/main.py --train_epochs=$TRAIN_EPOCHS
python run/parse_network.py
```
And then run with
```.env
python tester.py a.sh
```
Then a.sh will be modified like this, and waits the execution to finish.
```.env
a.sh

export TRAIN_EPOCHS=10
python run/main.py --train_epochs=$TRAIN_EPOCHS
===running above===
python run/parse_network.py
```
If you want to modify the next executions, you can just modify the a.sh file
```.env
a.sh

export TRAIN_EPOCHS=10
python run/main.py --train_epochs=$TRAIN_EPOCHS
===running above===
python run/main.py --train_epochs=20
python run/parse_network.py

git checkout test2
python run/main.py --train_epochs=20
```
Then tester will run the next line, when the execution is finished.
```.env
a.sh

export TRAIN_EPOCHS=10
python run/main.py --train_epochs=$TRAIN_EPOCHS
python run/main.py --train_epochs=20
===running above===
python run/parse_network.py

git checkout test2
python run/main.py --train_epochs=20
```

## Multiprocessing
You can also run multiple script files by passing filenames by arguments. If you use zsh, you can use `python tester.py -j 4 *.sh`

number of cores will be same as number of script files, as default.



## TODO
- [x] Supports 'export' in sh
- [x] Supports using environment variable '$MODEL_NAME' in windows
- [x] Supports multiple sh files
- [ ] Support to 'cd' in sh
