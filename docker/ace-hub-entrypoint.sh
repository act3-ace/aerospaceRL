#!/usr/bin/env bash

# cd $HOME

echo "Originally running as"
id
echo "Changing to $FIXUID:$FIXGID"
gosu $FIXUID:$FIXGID "$@"

cd /home/coder/aerospacerl
git remote set-url origin https://$ACE_USER:$GIT_PASSWORD@git.act3-ace.com/kyle.dunlap/aerospacerl.git
git pull
git checkout master

# or we can create the actual user by name
# adduser --uid $FIXUID --gid $FIGXGID $FIXUSER 
# gosu $FIXUSER "$@"
