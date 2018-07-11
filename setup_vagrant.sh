#!/bin/bash
# Unclear if this line actually needs to be run
# sudo add-apt-repository ppa:keithw/mahimahi
sudo apt-get update
sudo apt-get install mahimahi python-pip -y
sudo apt-get install python3-pip
pip3 install jupyter mypy

# This needs to start every time we start the box
sudo sysctl -w net.ipv4.ip_forward=1
