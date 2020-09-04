#!/usr/bin/env bash

# cd $HOME

echo "Originally running as"
id
echo "Changing to $FIXUID:$FIXGID"
gosu $FIXUID:$FIXGID "$@"

# or we can create the actual user by name
# adduser --uid $FIXUID --gid $FIGXGID $FIXUSER 
# gosu $FIXUSER "$@"
