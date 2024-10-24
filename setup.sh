# Requires Ubuntu 23.04 or later. Assuming you're starting empty.
sudo apt update
sudo apt upgrade
sudo apt install python3
sudo apt install cmake
python3 --version
sudo apt install python3-pip
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
