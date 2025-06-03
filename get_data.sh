#!/bin/bash
# get_data.sh - Download training data from SFTP server

mkdir -p ~/.ssh data/input && cp ssh_keys/bayesian-project-key ~/.ssh/id_rsa && chmod 600 ~/.ssh/id_rsa

echo "Downloading r139 data files..."
scp -i ~/.ssh/id_rsa Parmot@24.23.247.209:r139_*.txt data/input/

echo "Files downloaded:"
ls -la data/input/r139*
