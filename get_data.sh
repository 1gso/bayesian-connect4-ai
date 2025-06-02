#!/bin/bash
# get_data.sh - Download training data from SFTP server

mkdir -p ~/.ssh data/input && cp ssh_keys/bayesian-project-key ~/.ssh/id_rsa && chmod 600 ~/.ssh/id_rsa
scp -i ~/.ssh/id_rsa Parmot@24.23.247.209:'{r139_a.txt,r139_8.txt}' data/input/
