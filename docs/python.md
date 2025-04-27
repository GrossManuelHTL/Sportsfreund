watch this: https://www.youtube.com/watch?v=MVyb-nI4KyI
when we develop with python, we should develop like we don't have an IDE

## Needed packages/installations

sudo apt install python3-pip
sudo apt-get update


sudo apt install python3-venv


## How to start
(In ./raspyfit/processor/src)

 python3 -m venv venv
 source venv/bin/activate
 pip install -r requirements.txt
 python video_main.py [--live]