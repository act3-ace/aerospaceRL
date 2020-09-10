#!/usr/bin/env bash

Log="/home/coder/log.txt"

cd /home/coder/aerospacerl
git remote set-url origin https://$ACE_USER:$GIT_PASSWORD@git.act3-ace.com/kyle.dunlap/aerospacerl.git
git pull
git checkout master

echo "Originally running as" > $Log
id
echo "Changing to $FIXUID:$FIXGID" >> $Log 
gosu $FIXUID:$FIXGID "$@"
