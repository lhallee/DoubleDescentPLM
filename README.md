# DoubleDescentPLM

```bash
git clone https://github.com/lhallee/DoubleDescentPLM.git
cd DoubleDescentPLM
git pull origin main
sudo docker build -t plm .
sudo docker run --gpus all --ipc=host -v ${PWD}:/workspace plm python -m train
```
