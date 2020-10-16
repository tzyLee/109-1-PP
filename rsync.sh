#!/bin/bash

DIR_NAME=$(dirname $BASH_SOURCE)

nchc

rsync -avzh --exclude '.git' --exclude '*.pdf' --exclude 'docs' $DIR_NAME nchc:~

