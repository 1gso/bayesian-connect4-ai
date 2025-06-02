#!/bin/bash
# get_data.sh - Download training data from SFTP server

mkdir -p ~/.ssh data/input && cp ssh_keys/bayesian-project-key ~/.ssh/id_rsa && chmod 600 ~/.ssh/id_rsa
for f in r139_a.txt r139_8.txt; do curl -k --user "Parmot:" --key ~/.ssh/id_rsa -o "data/input/$f" "sftp://24.23.247.209:22/$f"; done
