This is a docker building pipeline for orcar with pip standard.

**Build docker**
```bash
sudo PLATFORM=x86_64 OPENAI_API_KEY="your_key" TAG=latest bash ./build.sh
```

**Run docker**

```bash
sudo docker run \
--rm --gpus all \
-it \
--user user \
-v $PWD/../:/home/user/orcar \
orcar/x86_64:latest \
/bin/bash
```